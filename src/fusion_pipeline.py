import torch
import torchvision
from torchvision import transforms
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}\n")

print("Loading models...")
yolo_model = YOLO('yolov8n.pt')

# Load Cityscapes-trained model (has road/sidewalk classes)
seg_model = torchvision.models.segmentation.deeplabv3_resnet101(weights='DEFAULT')
seg_model.eval()
seg_model.to(device)
print("Models loaded\n")

# Cityscapes color map
CITYSCAPES_COLORS = {
    0: (128, 64, 128),   # road - purple
    1: (244, 35, 232),   # sidewalk - pink  
    2: (70, 70, 70),     # building - dark gray
    3: (102, 102, 156),  # wall
    4: (190, 153, 153),  # fence
    5: (153, 153, 153),  # pole
    6: (250, 170, 30),   # traffic light - orange
    7: (220, 220, 0),    # traffic sign - yellow
    8: (107, 142, 35),   # vegetation - green
    9: (152, 251, 152),  # terrain - light green
    10: (70, 130, 180),  # sky - blue
    11: (220, 20, 60),   # person - red
    12: (255, 0, 0),     # rider - bright red
    13: (0, 0, 142),     # car - dark blue
    14: (0, 0, 70),      # truck - darker blue
    15: (0, 60, 100),    # bus - blue
}

def segment_image(image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = seg_model(input_tensor)['out'][0]
    
    seg_mask = output.argmax(0).cpu().numpy()
    return seg_mask

def create_segmentation_overlay(image, seg_mask, alpha=0.6):
    overlay = np.zeros_like(np.array(image))
    
    for class_id, color in CITYSCAPES_COLORS.items():
        mask = seg_mask == class_id
        overlay[mask] = color
    
    blended = cv2.addWeighted(np.array(image), 1-alpha, overlay, alpha, 0)
    return blended

def detect_and_segment(image_path, output_path):
    img_pil = Image.open(image_path).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # YOLOv8 detection
    results = yolo_model(image_path, device='mps')[0]
    
    # Segmentation
    seg_mask = segment_image(img_pil)
    
    # Create segmentation overlay (more visible now with alpha=0.6)
    seg_overlay = create_segmentation_overlay(img_pil, seg_mask, alpha=0.6)
    seg_overlay_bgr = cv2.cvtColor(seg_overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Draw YOLO boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{yolo_model.names[cls_id]} {conf:.2f}"
        
        cv2.rectangle(seg_overlay_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(seg_overlay_bgr, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite(str(output_path), seg_overlay_bgr)
    
    print(f"Processed: {image_path.name}")
    print(f"  Detected {len(results.boxes)} objects")
    print(f"  Saved to: {output_path}\n")

test_dir = Path("data/raw/test_images")
output_dir = Path("results/fusion_cityscapes")
output_dir.mkdir(parents=True, exist_ok=True)

print("Processing with Cityscapes segmentation...\n")

for img_path in test_dir.glob("*.jpg"):
    output_path = output_dir / f"fusion_{img_path.name}"
    detect_and_segment(img_path, output_path)

print(f"Done! Check {output_dir}")