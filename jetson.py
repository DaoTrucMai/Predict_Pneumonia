#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_model(num_classes: int = 2) -> nn.Module:
    # Jetson-safe: không tải pretrained weights từ internet
    model = models.densenet161(pretrained=True)
    in_features = model.classifier.in_features  # thường 2208
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def load_weights(model: nn.Module, weight_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(weight_path, map_location=device)

    # Thường là state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=True)
    else:
        # nếu save cả model object
        model = ckpt

    model.to(device)
    model.eval()
    return model


def make_transform() -> transforms.Compose:
    # Match notebook: Resize -> CenterCrop -> ToTensor (không normalize)
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def collect_images(root: Path) -> List[Path]:
    imgs = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return imgs


def infer_actual_from_path(p: Path) -> Optional[str]:
    # Actual lấy theo folder NORMAL/PNEUMONIA trong đường dẫn
    parts = [x.upper() for x in p.parts]
    for c in CLASS_NAMES:
        if c in parts:
            return c
    return None


def preprocess_image(image_path: Path, tfm: transforms.Compose) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    x = tfm(img)           # [3,224,224]
    return x.unsqueeze(0)  # [1,3,224,224]


def predict_one(model: nn.Module, x: torch.Tensor, device: torch.device) -> Tuple[str, Dict[str, float]]:
    """
    Return predicted label + probs dict
    """
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)                    # [1,2]
        probs = F.softmax(logits, dim=1)[0]  # [2]
        pred_idx = int(torch.argmax(probs).item())

    predicted = CLASS_NAMES[pred_idx]
    probs_dict = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))}
    return predicted, probs_dict


def format_probs(probs: Dict[str, float]) -> str:
    # ổn định thứ tự NORMAL rồi PNEUMONIA
    return "{ " + ", ".join([f"{k}: {probs[k]:.4f}" for k in CLASS_NAMES]) + " }"


def pick_device(choice: str) -> torch.device:
    choice = choice.lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Bạn chọn --device cuda nhưng CUDA không available.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_one_line(device: torch.device):
    if device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best-model-weighted.pt",
                        help="Path to trained weights (.pt)")

    # 1 ảnh cụ thể
    parser.add_argument("--image", type=str, default=None,
                        help="Đường dẫn 1 ảnh cụ thể để dự đoán (optional)")

    # batch random từ Data/test
    parser.add_argument("--test_dir", type=str, default="Data/test",
                        help="Path to Data/test folder (dùng khi không có --image)")
    parser.add_argument("--sample", type=int, default=10,
                        help="Số ảnh random để test (5/10/20...)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed (tuỳ chọn). Nếu không set thì mỗi lần chạy random khác nhau.")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Chọn device: auto/cpu/cuda")

    args = parser.parse_args()

    device = pick_device(args.device)
    print_device_one_line(device)

    tfm = make_transform()

    model = build_model(num_classes=2)
    model = load_weights(model, args.weights, device=device)

    # Chọn danh sách ảnh cần dự đoán
    if args.image is not None:
        img_path = Path(args.image)
        if not img_path.exists():
            raise SystemExit(f"Không tìm thấy ảnh: {img_path}")
        sampled = [img_path]
    else:
        test_root = Path(args.test_dir)
        if not test_root.exists():
            raise SystemExit(f"Không tìm thấy thư mục test: {test_root}")

        all_images = collect_images(test_root)
        if len(all_images) == 0:
            raise SystemExit("Không tìm thấy ảnh trong thư mục test.")

        if args.seed is not None:
            random.seed(args.seed)

        n = min(args.sample, len(all_images))
        sampled = random.sample(all_images, n)

    # Metrics
    total_correct = 0
    total_with_gt = 0
    sum_trueprob_percent = 0.0

    for i, img_path in enumerate(sampled, start=1):
        x = preprocess_image(img_path, tfm)
        predicted, probs_dict = predict_one(model, x, device)

        actual = infer_actual_from_path(img_path)
        filename = img_path.name

        if actual is None:
            acc_text = "N/A"
            mark = "?"
        else:
            # "Accuracy per-image" = Prob(True class) * 100
            true_prob_percent = 100.0 * probs_dict[actual]
            sum_trueprob_percent += true_prob_percent

            correct = 1 if predicted == actual else 0
            total_correct += correct
            total_with_gt += 1

            acc_text = f"{true_prob_percent:.2f}%"
            mark = "✓" if correct == 1 else "✗"

        print("-" * 70)
        print(f"[{i}/{len(sampled)}] File: {filename}")
        print(f"  Actual   : {actual if actual else 'UNKNOWN'}")
        print(f"  Predicted: {predicted}")
        print(f"  Probs    : {format_probs(probs_dict)}")
        print(f"  Accuracy : {acc_text} {mark} ")

    # Summary
    print("=" * 70)
    if total_with_gt > 0:
        avg_acc_percent = sum_trueprob_percent / total_with_gt
        std_acc_percent = 100.0 * (total_correct / total_with_gt)

        print(f"Correct count         : {total_correct}/{total_with_gt}")
        print(f"Average Accuracy (%)  : {avg_acc_percent:.2f}% ")
        # print(f"Standard Accuracy (%) : {std_acc_percent:.2f}%  (tỉ lệ đúng)")
    else:
        print("Correct count         : N/A")
        print("Average Accuracy (%)  : N/A")
        # print("Standard Accuracy (%) : N/A")
    print("=" * 70)


if __name__ == "__main__":
    main()

