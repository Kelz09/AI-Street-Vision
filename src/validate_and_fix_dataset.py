import os
from pathlib import Path
from tqdm import tqdm

print("Validating dataset...")

data_dir = Path("data/processed/self_driving_yolo")
nc = 11  # number of classes

def validate_label(label_path, nc):
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        valid_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            if class_id >= nc:
                continue
            
            coords = [float(x) for x in parts[1:5]]
            if any(c < 0 or c > 1 for c in coords):
                continue
            
            valid_lines.append(line)
        
        return valid_lines
    except:
        return None

def fix_split(split_name):
    print(f"\nChecking {split_name}...")
    
    img_dir = data_dir / split_name / "images"
    lbl_dir = data_dir / split_name / "labels"
    
    images = list(img_dir.glob("*.jpg"))
    
    removed = 0
    fixed = 0
    
    for img_path in tqdm(images):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        
        if not lbl_path.exists():
            img_path.unlink()
            removed += 1
            continue
        
        valid_lines = validate_label(lbl_path, nc)
        
        if valid_lines is None or len(valid_lines) == 0:
            img_path.unlink()
            lbl_path.unlink()
            removed += 1
        elif len(valid_lines) != len(open(lbl_path).readlines()):
            with open(lbl_path, 'w') as f:
                f.writelines(valid_lines)
            fixed += 1
    
    print(f"  Removed: {removed}, Fixed: {fixed}")

fix_split("train")
fix_split("valid")
fix_split("test")

print("\nDataset validation complete!")