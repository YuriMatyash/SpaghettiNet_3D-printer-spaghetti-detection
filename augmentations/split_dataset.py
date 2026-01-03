import os
import shutil
import random
from pathlib import Path

# ================= CONFIGURATION =================
# Define your source folders here
# Set 1: Files containing the toolhead
SRC_TOOLHEAD_IMAGES = "datasets/detection/resized_toolhead/"
SRC_TOOLHEAD_LABELS = "datasets/detection/labels_toolhead/"

# Set 2: Empty files (Background images)
SRC_EMPTY_IMAGES = "datasets/detection/resized_clean/"
SRC_EMPTY_LABELS = "datasets/detection/labels_clean/"

# Where you want the final dataset to be created
DEST_ROOT = "datasets/detection/dataset/"

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10

# Image extensions to look for
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
# =================================================

def create_dirs(base_path):
    """Creates the standard YOLO folder structure."""
    for split in ['train', 'val', 'test']:
        for dtype in ['images', 'labels']:
            path = Path(base_path) / dtype / split
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {path}")

def get_file_pairs(img_dir, lbl_dir):
    """
    Matches images to text files by stem (filename without extension).
    Returns a list of tuples: (path_to_image, path_to_label)
    """
    img_path = Path(img_dir)
    lbl_path = Path(lbl_dir)
    
    pairs = []
    
    # Get all images
    images = [f for f in img_path.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
    
    for img in images:
        # Construct expected label path
        lbl = lbl_path / f"{img.stem}.txt"
        
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            print(f"Warning: Missing label for {img.name}, skipping.")
            
    return pairs

def main():
    # 1. Setup
    print("--- Starting Dataset Split ---")
    random.seed(42) # Set seed for reproducibility (optional, remove for pure chaos)
    
    # 2. Gather Data
    print("Gathering files...")
    toolhead_pairs = get_file_pairs(SRC_TOOLHEAD_IMAGES, SRC_TOOLHEAD_LABELS)
    empty_pairs = get_file_pairs(SRC_EMPTY_IMAGES, SRC_EMPTY_LABELS)
    
    print(f"Found {len(toolhead_pairs)} toolhead pairs.")
    print(f"Found {len(empty_pairs)} empty pairs.")
    
    # 3. Combine and Shuffle
    all_pairs = toolhead_pairs + empty_pairs
    random.shuffle(all_pairs) # <--- This ensures random order, not by filename
    
    total_files = len(all_pairs)
    print(f"Total files to process: {total_files}")
    
    if total_files == 0:
        print("Error: No files found. Check your paths.")
        return

    # 4. Calculate Split Indices
    train_end = int(total_files * TRAIN_RATIO)
    val_end = train_end + int(total_files * VAL_RATIO)
    
    # Slicing the shuffled list
    train_set = all_pairs[:train_end]
    val_set = all_pairs[train_end:val_end]
    test_set = all_pairs[val_end:]
    
    print(f"Split breakdown: Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    
    # 5. Create Directories
    create_dirs(DEST_ROOT)
    
    # 6. Copy Files
    def copy_set(dataset, split_name):
        print(f"Copying {len(dataset)} files to {split_name}...")
        for img_src, lbl_src in dataset:
            # Define destinations
            img_dest = Path(DEST_ROOT) / 'images' / split_name / img_src.name
            lbl_dest = Path(DEST_ROOT) / 'labels' / split_name / lbl_src.name
            
            # Copy files
            shutil.copy2(img_src, img_dest)
            shutil.copy2(lbl_src, lbl_dest)

    copy_set(train_set, 'train')
    copy_set(val_set, 'val')
    copy_set(test_set, 'test')

    # 7. Create data.yaml (Bonus)
    yaml_content = f"""
path: {Path(DEST_ROOT).absolute()} 
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: toolhead
"""
    with open(Path(DEST_ROOT) / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("--- Done! data.yaml created in root folder. ---")

if __name__ == "__main__":
    main()