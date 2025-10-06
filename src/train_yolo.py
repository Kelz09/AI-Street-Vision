from ultralytics import YOLO
import torch
from pathlib import Path

print("=" * 60)
print("YOLOv8 Training - Self Driving Car Dataset")
print("=" * 60)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"\nUsing device: {device}")

print("\nLoading YOLOv8n pretrained model...")
model = YOLO('yolov8n.pt')

data_yaml = Path('data/processed/self_driving_yolo/data.yaml')

if not data_yaml.exists():
    print(f"\nError: Dataset not found at {data_yaml}")
    print("Run: python src/prepare_roboflow.py first!")
    exit(1)

print(f"Dataset config: {data_yaml}")

print("\n" + "=" * 60)
print("Training Configuration")
print("=" * 60)

config = {
    'data': str(data_yaml),
    'epochs': 50,
    'imgsz': 640,
    'batch': 16,
    'device': device,
    'project': 'runs/detect',
    'name': 'self_driving_yolo',
    'patience': 15,
    'save': True,
    'plots': True,
    'verbose': True,
    'workers': 4,
}

for key, value in config.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("Starting Training...")
print("=" * 60)
print("\nExpected time: 4-5 hours on M3 MacBook Air")
print("You can monitor progress below.\n")

try:
    results = model.train(**config)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nResults saved to: runs/detect/self_driving_yolo/")
    print(f"Best model: runs/detect/self_driving_yolo/weights/best.pt")
    print(f"Training plots: runs/detect/self_driving_yolo/")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    print("Partial results saved in: runs/detect/self_driving_yolo/")
    
except Exception as e:
    print(f"\n\nError during training: {e}")