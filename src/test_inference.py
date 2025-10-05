import torch
import torchvision
from torchvision import transforms
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}\n")

# Load models
print("Loading models...")
yolo_model = YOLO('yolov8n.pt')
segmentation_model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
segmentation_model.eval()
segmentation_model.to(device)
print("✓ Models loaded\n")

# Segmentation classes (COCO classes used by DeepLabv3+)
CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def segment_image(image_path):
    """Run semantic segmentation"""
    img = Image.open(image_path).convert('RGB')
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = segmentation_model(input_tensor)['out'][0]
    
    output_predictions = output.argmax(0).cpu().numpy()
    return output_predictions, img

def detect_objects(image_path):
    """Run object detection with YOLOv8"""
    results = yolo_model(image_path, device='mps')
    return results

# Test on all images
test_dir = "data/test_images"
image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

print(f"Testing on {len(image_files)} images...\n")
print("=" * 60)

for img_file in image_files:
    img_path = os.path.join(test_dir, img_file)
    print(f"\n📸 Testing: {img_file}")
    print("-" * 60)
    
    # YOLOv8 Detection
    print("Running YOLOv8 detection...")
    detect_results = detect_objects(img_path)
    
    # Count detections
    boxes = detect_results[0].boxes
    print(f"  ✓ Detected {len(boxes)} objects:")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = yolo_model.names[cls_id]
        print(f"    • {cls_name}: {conf:.2f}")
    
    # Semantic Segmentation
    print("\nRunning DeepLabv3+ segmentation...")
    seg_mask, original_img = segment_image(img_path)
    
    unique_classes = np.unique(seg_mask)
    print(f"  ✓ Found {len(unique_classes)} semantic classes:")
    for cls_id in unique_classes:
        if cls_id < len(CLASSES):
            print(f"    • {CLASSES[cls_id]}")
    
    # Save visualizations
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save YOLO result
    yolo_output = f"{output_dir}/yolo_{img_file}"
    detect_results[0].save(yolo_output)
    print(f"\n  💾 YOLO result saved: {yolo_output}")
    
    print("=" * 60)

print("\n✅ Testing complete! Check the 'results' folder for visualizations.")