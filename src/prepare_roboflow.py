import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

print("Organizing Roboflow Dataset for YOLOv8")
print("=" * 60)

# Paths
SOURCE_DIR = Path("data/raw/roboflow/export")
IMAGES_DIR = SOURCE_DIR / "images"
LABELS_DIR = SOURCE_DIR / "labels"

OUTPUT_DIR = Path("data/processed/self_driving_yolo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Get all image files
image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.jpeg"))
print(f"\nFound {len(image_files)} images")

# Shuffle for random split
random.seed(42)
random.shuffle(image_files)

# Split ratios: 70% train, 20% val, 10% test
# But we'll use smaller subset for faster training
total_samples = 10000  # I am using 10k samples for quick iteration 
image_files = image_files[:total_samples]

train_split = int(0.7 * len(image_files))
val_split = int(0.9 * len(image_files))

train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]

print(f"\nUsing {total_samples} total samples:")
print(f"  Train: {len(train_files)}")
print(f"  Val:   {len(val_files)}")
print(f"  Test:  {len(test_files)}")

def copy_split(files, split_name):
    print(f"\nProcessing {split_name} split...")
    
    img_out = OUTPUT_DIR / split_name / "images"
    lbl_out = OUTPUT_DIR / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    skipped = 0
    
    for img_path in tqdm(files, desc=f"  Copying {split_name}"):
        img_name = img_path.name
        img_stem = img_path.stem
        
        # Find matching label
        lbl_path = LABELS_DIR / f"{img_stem}.txt"
        
        if not lbl_path.exists():
            skipped += 1
            continue
        
        # Copy image and label
        shutil.copy(img_path, img_out / img_name)
        shutil.copy(lbl_path, lbl_out / f"{img_stem}.txt")
        copied += 1
    
    print(f"  Copied: {copied}, Skipped: {skipped}")

# Process all splits
copy_split(train_files, "train")
copy_split(val_files, "valid")
copy_split(test_files, "test")

# Create updated data.yaml
yaml_content = f"""path: {OUTPUT_DIR.absolute()}
train: train/images
val: valid/images
test: test/images

nc: 11
names: ['biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green', 'trafficLight-GreenLeft', 'trafficLight-Red', 'trafficLight-RedLeft', 'trafficLight-Yellow', 'trafficLight-YellowLeft', 'truck']
"""

yaml_path = OUTPUT_DIR / "data.yaml"
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print("\n" + "=" * 60)
print("Dataset preparation complete!")
print("=" * 60)
print(f"\nOutput: {OUTPUT_DIR.absolute()}")
print(f"Config: {yaml_path}")
print("\nDataset ready for training!")
print("Next: python src/train_yolo.py")