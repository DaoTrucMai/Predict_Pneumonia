#!/bin/bash
# ============================================================================
# Jetson Nano AI Environment Setup Script
# ============================================================================
# Target: Jetson Nano 4GB - Ubuntu 18.04 - JetPack 4.6.x (L4T R32.7.1)
# Purpose: Pneumonia Prediction (Chest X-ray) Model Deployment
#
# IMPORTANT: Script sử dụng System Python (không dùng virtual environment)
#            vì PyTorch wheel NVIDIA không tương thích với venv trên Jetson
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# ============================================================================

set -e  # Exit on error

# === Configuration ===
SWAP_SIZE="4G"
WORK_DIR="$HOME/dr_setup_temp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PyTorch wheel URL (JetPack 4.6.x / Python 3.6)
PYTORCH_WHEEL_URL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"
PYTORCH_WHEEL_NAME="torch-1.10.0-cp36-cp36m-linux_aarch64.whl"

# TorchVision version to build from source
TORCHVISION_VERSION="v0.11.1"

# === Color Codes ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# === Helper Functions ===
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

# === Pre-flight Checks ===
check_jetson() {
    log_step "Checking Jetson Nano..."
    
    if [ -f /etc/nv_tegra_release ]; then
        JETSON_INFO=$(cat /etc/nv_tegra_release | head -1)
        log_success "Jetson detected: $JETSON_INFO"
    else
        log_warning "Not running on Jetson Nano"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_info "Python version: $PYTHON_VERSION"
    
    # Check if Python 3.6
    if [[ ! "$PYTHON_VERSION" == 3.6* ]]; then
        log_warning "Expected Python 3.6.x, got $PYTHON_VERSION"
        log_warning "PyTorch wheel may not be compatible"
    fi
}

# === Step 1: Configure Swap ===
step_configure_swap() {
    log_step "Step 1/6: Configuring Swap Space"
    
    CURRENT_SWAP=$(free -g | grep Swap | awk '{print $2}')
    log_info "Current swap: ${CURRENT_SWAP}GB"
    
    if [ "$CURRENT_SWAP" -lt 4 ]; then
        log_info "Creating $SWAP_SIZE swap file..."
        
        if [ -f /swapfile ]; then
            log_info "Removing existing swapfile..."
            sudo swapoff /swapfile 2>/dev/null || true
            sudo rm -f /swapfile
        fi
        
        sudo fallocate -l $SWAP_SIZE /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        
        # Add to fstab if not already present
        if ! grep -q "swapfile" /etc/fstab; then
            echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        fi
        
        log_success "Swap configured: $SWAP_SIZE"
    else
        log_success "Swap already sufficient: ${CURRENT_SWAP}GB"
    fi
}

# === Step 2: Install System Dependencies ===
step_install_system_deps() {
    log_step "Step 2/6: Installing System Dependencies"
    
    # Disable NVIDIA repos to avoid update errors
    if [ -f /etc/apt/sources.list.d/nvidia-l4t-apt-source.list ]; then
        log_info "Temporarily disabling NVIDIA repos..."
        sudo mv /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
               /etc/apt/sources.list.d/nvidia-l4t-apt-source.list.bak 2>/dev/null || true
    fi
    
    log_info "Updating package lists..."
    sudo apt-get update 2>/dev/null || log_warning "Some repos failed (OK)"
    
    log_info "Installing required packages..."
    sudo apt-get install -y \
        python3-pip \
        python3-dev \
        python3-setuptools \
        libopenblas-base \
        libopenblas-dev \
        libopenmpi-dev \
        libomp-dev \
        openmpi-bin \
        libjpeg-dev \
        zlib1g-dev \
        libpython3-dev \
        libatlas-base-dev \
        libblas-dev \
        liblapack-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        cmake \
        git \
        wget \
        curl \
        2>/dev/null || log_warning "Some packages may have failed"
    
    # Restore NVIDIA repos
    if [ -f /etc/apt/sources.list.d/nvidia-l4t-apt-source.list.bak ]; then
        sudo mv /etc/apt/sources.list.d/nvidia-l4t-apt-source.list.bak \
               /etc/apt/sources.list.d/nvidia-l4t-apt-source.list 2>/dev/null || true
    fi
    
    log_success "System dependencies installed"
}

# === Step 3: Upgrade pip ===
step_upgrade_pip() {
    log_step "Step 3/6: Upgrading pip"
    
    sudo pip3 install --upgrade pip setuptools wheel
    log_success "pip upgraded"
}

# === Step 4: Install PyTorch ===
step_install_pytorch() {
    log_step "Step 4/6: Installing PyTorch (NVIDIA Wheel)"
    
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    
    # Check if PyTorch already installed and working
    if python3 -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())" 2>/dev/null; then
        log_warning "PyTorch already installed"
        read -p "Reinstall? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping PyTorch installation"
            return 0
        fi
        sudo pip3 uninstall torch -y 2>/dev/null || true
    fi
    
    # Download PyTorch wheel
    log_info "Downloading PyTorch wheel from NVIDIA..."
    log_info "URL: $PYTORCH_WHEEL_URL"
    
    if [ -f "$PYTORCH_WHEEL_NAME" ]; then
        log_info "Wheel already exists, skipping download"
    else
        wget -O "$PYTORCH_WHEEL_NAME" "$PYTORCH_WHEEL_URL" || {
            log_error "Download failed!"
            log_info "Trying alternative method..."
            curl -L -o "$PYTORCH_WHEEL_NAME" "$PYTORCH_WHEEL_URL" || {
                log_error "Both wget and curl failed"
                exit 1
            }
        }
    fi
    
    # Verify file size (should be ~300MB+)
    FILE_SIZE=$(stat -c%s "$PYTORCH_WHEEL_NAME" 2>/dev/null || stat -f%z "$PYTORCH_WHEEL_NAME")
    if [ "$FILE_SIZE" -lt 100000000 ]; then
        log_error "Downloaded file too small ($FILE_SIZE bytes). Re-downloading..."
        rm -f "$PYTORCH_WHEEL_NAME"
        wget -O "$PYTORCH_WHEEL_NAME" "$PYTORCH_WHEEL_URL"
    fi
    
    log_info "Installing PyTorch..."
    sudo pip3 install "$PYTORCH_WHEEL_NAME"
    
    # Verify installation
    log_info "Verifying PyTorch installation..."
    python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())" || {
        log_error "PyTorch verification failed!"
        exit 1
    }
    
    log_success "PyTorch installed successfully"
}

# === Step 5: Install TorchVision (Build from Source) ===
step_install_torchvision() {
    log_step "Step 5/6: Installing TorchVision (Building from Source)"
    
    cd "$WORK_DIR"
    
    # Check if already installed
    if python3 -c "import torchvision; from torchvision.ops.misc import FrozenBatchNorm2d; print('OK')" 2>/dev/null; then
        log_warning "TorchVision already installed with ops module"
        read -p "Reinstall? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping TorchVision installation"
            return 0
        fi
    fi
    
    # Remove old torchvision versions
    log_info "Removing old TorchVision versions..."
    sudo pip3 uninstall torchvision -y 2>/dev/null || true
    sudo pip3 uninstall torchvision -y 2>/dev/null || true
    sudo rm -rf /usr/local/lib/python3.6/dist-packages/torchvision* 2>/dev/null || true
    
    # Clone TorchVision
    log_info "Cloning TorchVision $TORCHVISION_VERSION..."
    rm -rf torchvision
    git clone --branch "$TORCHVISION_VERSION" --depth 1 https://github.com/pytorch/vision torchvision
    cd torchvision
    
    # Build and install
    log_info "Building TorchVision (this may take 15-20 minutes)..."
    export BUILD_VERSION=0.11.1
    export TORCH_CUDA_ARCH_LIST="5.3"  # Jetson Nano GPU architecture
    
    sudo python3 setup.py install
    
    cd "$WORK_DIR"
    rm -rf torchvision
    
    # Verify installation
    log_info "Verifying TorchVision installation..."
    python3 -c "import torchvision; print('TorchVision:', torchvision.__version__)"
    python3 -c "from torchvision.ops.misc import FrozenBatchNorm2d; print('torchvision.ops: OK')"
    
    log_success "TorchVision installed successfully"
}

# === Step 6: Install Python Packages ===
step_install_python_packages() {
    log_step "Step 6/6: Installing Python Packages (Pneumonia inference)"

    # Keep packages minimal for Jetson Nano inference
    log_info "Installing numpy, pillow, tqdm..."
    sudo pip3 install --upgrade numpy pillow tqdm

    # Optional but common: OpenCV bindings (use apt on Jetson)
    log_info "Installing python3-opencv (optional)..."
    sudo apt-get install -y python3-opencv 2>/dev/null || log_warning "python3-opencv install failed (optional)"

    log_success "Python packages installed"
}

# === Verification ===
verify_installation() {
    log_step "Verifying Installation"
    
    echo ""
    log_info "Testing all packages..."
    
    python3 << 'EOF'
import sys
print(f"Python: {sys.version.split()[0]}")

def test_import(name, pkg_name=None):
    pkg_name = pkg_name or name
    try:
        module = __import__(pkg_name)
        version = getattr(module, '__version__', 'OK')
        print(f"  ✓ {name}: {version}")
        return True
    except ImportError as e:
        print(f"  ✗ {name}: FAILED ({e})")
        return False

print("\n=== Package Verification ===")
results = []
results.append(test_import("PyTorch", "torch"))
results.append(test_import("TorchVision", "torchvision"))
results.append(test_import("NumPy", "numpy"))
results.append(test_import("Pillow", "PIL"))
results.append(test_import("tqdm", "tqdm"))

print("")

# Check CUDA
import torch
if torch.cuda.is_available():
    print(f"  ✓ CUDA: {torch.version.cuda}")
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    
    # Quick GPU test
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.mm(x, x)
        print("  ✓ GPU compute: OK")
    except Exception as e:
        print(f"  ✗ GPU compute: FAILED ({e})")
else:
    print("  ⚠ CUDA: Not available")

print("")
if all(results):
    print("=== All packages installed successfully! ===")
else:
    print("=== Some packages failed ===")
    sys.exit(1)
EOF
}

# === Cleanup ===
cleanup() {
    log_info "Cleaning up..."
    rm -rf "$WORK_DIR"
    log_success "Cleanup complete"
}

# === Print Final Instructions ===
print_instructions() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              INSTALLATION COMPLETE!                        ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "Quick test:"
    echo -e "  ${CYAN}python3 -c \"import torch; import torchvision; print('CUDA:', torch.cuda.is_available()); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__)\"${NC}"
    echo ""
    echo -e "Next steps (Pneumonia Prediction):"
    echo -e "  1) Copy your trained weights to this folder (example):"
    echo -e "     ${CYAN}pneumonia_densenet161.pth${NC}"
    echo -e "  2) Run inference on an X-ray image:"
    echo -e "     ${CYAN}cd $SCRIPT_DIR${NC}"
    echo -e "     ${CYAN}python3 jetson_inference.py --image /path/to/xray.jpg --weights pneumonia_densenet161.pth${NC}"
    echo ""
    echo -e "For better performance:"
    echo -e "  ${CYAN}sudo nvpmodel -m 0${NC}     # Max power mode"
    echo -e "  ${CYAN}sudo jetson_clocks${NC}    # Lock clocks to maximum"
    echo ""
}

# === Main Execution ===
main() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   Jetson Nano AI Environment Setup                         ║${NC}"
    echo -e "${CYAN}║   Pneumonia Prediction (Chest X-ray)                       ║${NC}"
    echo -e "${CYAN}║   JetPack 4.6.x / Python 3.6 / System Python               ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}NOTE: This script uses System Python (no virtual environment)${NC}"
    echo -e "${YELLOW}      because NVIDIA PyTorch wheel is incompatible with venv${NC}"
    echo ""
    
    read -p "Continue with installation? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    
    # Pre-flight
    check_jetson
    
    # Installation steps
    step_configure_swap
    step_install_system_deps
    step_upgrade_pip
    step_install_pytorch
    step_install_torchvision
    step_install_python_packages
    
    # Verify
    verify_installation
    
    # Cleanup
    cleanup
    
    # Done
    print_instructions
}

# Run
main "$@"

