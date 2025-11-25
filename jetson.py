# inference_jetson.py - Chạy cực nhanh trên Jetson Nano
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import time
import os

# ================== CẤU HÌNH ==================
MODEL_PATH = "densenet161-8d451a50.pth"    # ← tên file model của bạn
IMG_PATH   = "test_img.jpg"              # ← ảnh bạn muốn test (để cùng thư mục)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform giống hệt lúc train
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ================== LOAD MODEL ==================
# Vì bạn dùng DenseNet161 trong train.py → load lại đúng như vậy
model = models.densenet161(pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

# Load trọng số bạn đã train
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)

class_names = ['NORMAL', 'PNEUMONIA']

# ================== HÀM DỰ ĐOÁN ==================
def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(DEVICE)  # thêm batch dimension
    
    with torch.no_grad():
        start = time.time()
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        prob = torch.softmax(outputs, dim=1)[0][predicted.item()].item()
        fps = 1.0 / (time.time() - start)
    
    return class_names[predicted.item()], prob*100, fps

# ================== CHẠY TEST ==================
if os.path.exists(IMG_PATH):
    label, confidence, fps = predict_image(IMG_PATH)
    print(f"Result: {label}")
    print(f"Accuracy: {confidence:.2f}%")
    print(f"Speed: {fps:.1f} FPS (on Jetson Nano)")
else:
    print("Can not find test_image.jpg. Please put your test image in the same folder as this script.")