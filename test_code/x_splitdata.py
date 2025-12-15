import os
import shutil
import random

# --- CONFIGURATION ---
# Where your current images are
source_clean = "data/processed/resolution_scaling/good/images"       
source_spaghetti = "data/processed/resolution_scaling/spaghetti/images" 

# Where you want the YOLO dataset to be created
dataset_root = "datasets/spaghetti_classifier"
split_ratio = 0.8  # 80% train, 20% val

def setup_dirs():
    for split in ['train', 'val']:
        for cls in ['clean', 'spaghetti']:
            os.makedirs(os.path.join(dataset_root, split, cls), exist_ok=True)

def copy_images(source_dir, class_name):
    # Get all images
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)
    
    # Calculate split index
    split_idx = int(len(files) * split_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    print(f"Processing {class_name}: {len(train_files)} training, {len(val_files)} validation")
    
    # Copy files
    for f in train_files:
        shutil.copy2(os.path.join(source_dir, f), os.path.join(dataset_root, 'train', class_name, f))
        
    for f in val_files:
        shutil.copy2(os.path.join(source_dir, f), os.path.join(dataset_root, 'val', class_name, f))

if __name__ == "__main__":
    setup_dirs()
    if os.path.exists(source_clean) and os.path.exists(source_spaghetti):
        copy_images(source_clean, "clean")
        copy_images(source_spaghetti, "spaghetti")
        print(f"Dataset ready at: {dataset_root}")
    else:
        print("Error: Check your source paths!")