import torch
import torchvision
from ultralytics import YOLO
import cv2
import numpy as np

print("=" * 50)
print("AI STREET VISION - Environment Test")
print("=" * 50)

# 1. Test PyTorch and MPS
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ Torchvision version: {torchvision.__version__}")
print(f"✓ MPS (Metal) available: {torch.backends.mps.is_available()}")
print(f"✓ MPS built: {torch.backends.mps.is_built()}")

# 2. Test device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(5, 5, device=device)
    print(f"✓ Successfully created tensor on MPS")
    print(f"  Device: {x.device}")
else:
    device = torch.device("cpu")
    print("⚠ MPS not available, using CPU")

# 3. Test YOLOv8
print("\n" + "=" * 50)
print("Testing YOLOv8n...")
print("=" * 50)
try:
    model = YOLO('yolov8n.pt')  # This will auto-download if not present
    print("✓ YOLOv8n loaded successfully")
    print(f"  Model type: {type(model)}")
except Exception as e:
    print(f"✗ Error loading YOLOv8: {e}")

# 4. Test DeepLabv3+
print("\n" + "=" * 50)
print("Testing DeepLabv3+ (pretrained)...")
print("=" * 50)
try:
    segmentation_model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights='DEFAULT'
    )
    segmentation_model.eval()
    segmentation_model.to(device)
    print("✓ DeepLabv3+ loaded successfully")
    print(f"  Model on device: {next(segmentation_model.parameters()).device}")
except Exception as e:
    print(f"✗ Error loading DeepLabv3+: {e}")

print("\n" + "=" * 50)
print("🎉 All systems ready!")
print("=" * 50)