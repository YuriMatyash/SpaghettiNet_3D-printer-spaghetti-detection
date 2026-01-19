import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re

# Import your custom modules
import hyperparams as hp
import utils

def main():
    print("--- Starting EDA Generation ---")
    
    # 1. Class Balance Visualization
    plot_class_balance()
    
    # 2. Temporal Motion Analysis (The "Spike" Graph)
    plot_temporal_motion()
    
    # 3. Data Diversity Mosaic
    plot_data_diversity()

    print("\n--- All EDA charts saved! ---")


def plot_class_balance():
    print("\n1. Generating Class Balance Chart...")
    
    # Use your existing utils function to get the exact training windows
    sequences, labels = utils.get_data_lists() #
    
    clean_count = labels.count(0)
    detached_count = labels.count(1)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Clean (0)', 'Detached (1)'], [clean_count, detached_count], color=['#2ecc71', '#e74c3c'])
    
    # Add numbers on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom', fontweight='bold')
    
    plt.title(f"Dataset Class Balance (Window Size: {hp.SEQ_LEN})")
    plt.ylabel("Number of 16-Frame Sequences")
    plt.grid(axis='y', alpha=0.3)
    
    save_path = "eda_class_balance.png"
    plt.savefig(save_path, dpi=300)
    print(f"   Saved to {save_path}")
    plt.close()


def plot_temporal_motion():
    print("\n2. Generating Temporal Motion Analysis...")
    
    # Find a video folder that actually has a detachment
    video_folders = glob.glob(os.path.join(hp.DATASET_ROOT, "*"))
    target_folder = None
    
    # Look for a folder containing a "detach" file
    for folder in video_folders:
        if any("detach" in f.lower() for f in os.listdir(folder)):
            target_folder = folder
            break
            
    if not target_folder:
        print("   No detachment video found for analysis.")
        return

    print(f"   Analyzing folder: {os.path.basename(target_folder)}")
    
    # 1. Load and Sort Images (Reusing logic from utils.py)
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(target_folder, ext)))

    # Sort numerically
    def sort_key(fname):
        base = os.path.basename(fname)
        numbers = re.findall(r'\d+', base)
        return int(numbers[0]) if numbers else 0
    images = sorted(images, key=sort_key)
    
    # 2. Calculate Frame-to-Frame MSE (Mean Squared Error)
    motion_scores = []
    frames_indices = []
    detachment_idx = -1
    
    prev_frame = None
    
    for i, img_path in enumerate(images):
        # Mark the detachment frame index
        if "detach" in os.path.basename(img_path).lower():
            detachment_idx = i
            
        # Read image in grayscale for simple pixel difference
        curr_frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if curr_frame is None: continue
        
        # Resize for speed (optional)
        curr_frame = cv2.resize(curr_frame, (224, 224))
        
        if prev_frame is not None:
            # Calculate pixel difference (Motion Energy)
            diff = cv2.absdiff(curr_frame, prev_frame)
            score = np.mean(diff) # Average pixel change
            motion_scores.append(score)
            frames_indices.append(i)
        
        prev_frame = curr_frame

    # 3. Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(frames_indices, motion_scores, color='#3498db', linewidth=2, label='Pixel Motion Energy')
    
    # Add vertical line for detachment event
    if detachment_idx != -1:
        plt.axvline(x=detachment_idx, color='#e74c3c', linestyle='--', linewidth=2, label='Detachment Event')
        plt.text(detachment_idx + 0.5, max(motion_scores)*0.8, 'Failure', color='#e74c3c', fontweight='bold')

    plt.title("Temporal Motion Analysis (Frame-to-Frame Difference)")
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Pixel Difference (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "eda_temporal_motion.png"
    plt.savefig(save_path, dpi=300)
    print(f"   Saved to {save_path}")
    plt.close()


def plot_data_diversity():
    print("\n3. Generating Data Diversity Mosaic...")
    
    video_folders = glob.glob(os.path.join(hp.DATASET_ROOT, "*"))
    
    # Limit to 9 videos for a 3x3 grid
    video_folders = video_folders[:9]
    
    if not video_folders:
        print("   No data found.")
        return

    # Calculate grid size (e.g., 2x2, 3x3)
    grid_size = int(np.ceil(np.sqrt(len(video_folders))))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, folder in enumerate(video_folders):
        # Get the first image from each folder
        images = glob.glob(os.path.join(folder, "*"))
        
        # Sort to find the first frame
        def sort_key(fname):
            base = os.path.basename(fname)
            numbers = re.findall(r'\d+', base)
            return int(numbers[0]) if numbers else 0
            
        if images:
            first_img_path = sorted(images, key=sort_key)[0]
            img = cv2.imread(first_img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert for Matplotlib
            
            axes[i].imshow(img)
            axes[i].set_title(os.path.basename(folder), fontsize=8)
            axes[i].axis('off')
    
    # Hide empty subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Data Source Diversity (First Frame per Video)", fontsize=14)
    plt.tight_layout()
    
    save_path = "eda_diversity_mosaic.png"
    plt.savefig(save_path, dpi=300)
    print(f"   Saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    main()