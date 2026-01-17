import os
import shutil
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import classification_report
import matplotlib.patches as patches
import yaml
import matplotlib.image as mpimg
import hyperparams as hp

#################################################################
# GENERAL
#################################################################

# fixes .yaml file to have a relative path per user
def fix_dataset_yaml(yaml_path):
    """
    Updates the dataset.yaml file to include the absolute 'path' key.
    This fixes the 'images not found' error by telling YOLO exactly where
    the dataset root is, regardless of where the script is run from.
    """
    print(f"Fixing paths in: {yaml_path}")
    
    # 1. Resolve absolute path to the dataset directory (where yaml is)
    abs_dataset_root = os.path.dirname(os.path.abspath(yaml_path))
    
    # 2. Read the current YAML
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    # 3. Inject the absolute 'path' key
    # YOLO uses this 'path' as the base for 'train' and 'val'
    data['path'] = abs_dataset_root
    
    # Optional: Ensure train/val are relative to this root
    # (e.g., if they were '../images/train', we strip the '../')
    for key in ['train', 'val', 'test']:
        if key in data and isinstance(data[key], str):
            # If path starts with relative chars like ./ or ../, clean it
            # But usually 'images/train' is what we want if 'path' is set.
            pass 

    # 4. Write back the fixed YAML
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    
    print(f"✅ Updated 'path' in yaml to: {abs_dataset_root}")

#################################################################
# EDA FUNCTIONS
#################################################################
def eda_detect_class_distribution(yaml_path):
    """
    Counts how many objects per class AND how many empty (clean) images exist.
    """
    print("Generating Detection Class Distribution (including Background/Clean)...")
    
    try:
        img_dir, lbl_dir, data = _get_dataset_paths(yaml_path)
    except Exception as e:
        print(f"⚠️ Error resolving paths: {e}")
        return

    if not os.path.exists(lbl_dir):
        print(f"⚠️ Error: Labels folder not found at {lbl_dir}")
        return

    class_counts = {}
    total_boxes = 0
    empty_images = 0  # Counter for clean images
    
    label_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
    
    if not label_files:
        print(f"⚠️ No .txt label files found in {lbl_dir}")
        return

    for l_file in label_files:
        path = os.path.join(lbl_dir, l_file)
        
        # We need to check if file is empty or contains actual boxes
        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        if not lines:
            # File exists but is empty -> This is a "Clean" image
            empty_images += 1
        else:
            # File has boxes -> Count them
            for line in lines:
                parts = line.split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    total_boxes += 1
                
    # Prepare Data for Plotting
    names = data.get('names', {})
    
    # 1. Get standard class names and counts
    labels = [names.get(k, str(k)) for k in sorted(class_counts.keys())]
    counts = [class_counts[k] for k in sorted(class_counts.keys())]
    
    # 2. Add "Background/Clean" to the chart
    labels.append("Background (Clean)")
    counts.append(empty_images)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Create colors: Use standard palette for classes, and Grey/Green for Background
    colors = sns.color_palette("viridis", len(class_counts)) + [(0.5, 0.5, 0.5)] # Grey for background
    
    bars = plt.bar(labels, counts, color=colors)
    plt.bar_label(bars)
    
    plt.title(f"Dataset Balance: {total_boxes} Objects vs {empty_images} Clean Images")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

def eda_detect_visualize_boxes(yaml_path, num_samples=6):
    """
    Visualizes random training images with Ground Truth bounding boxes.
    """
    print(f"Visualizing {num_samples} random samples with Ground Truth boxes...")
    
    try:
        img_dir, lbl_dir, data = _get_dataset_paths(yaml_path)
    except Exception as e:
        print(f"⚠️ Error resolving paths: {e}")
        return
    
    class_names = data.get('names', {})
    
    if not os.path.exists(img_dir):
        print(f"⚠️ Images folder not found at {img_dir}")
        return

    # Get paired files
    all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not all_images:
        print("No images found to visualize.")
        return

    samples = random.sample(all_images, min(len(all_images), num_samples))
    
    # Calculate grid size
    cols = 3
    rows = (len(samples) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, img_file in enumerate(samples):
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(lbl_dir, label_file)
        
        ax = axes[i]
        ax.imshow(img_rgb)
        ax.axis('off')
        ax.set_title(img_file, fontsize=9)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                boxes = f.readlines()
                
            for box in boxes:
                parts = list(map(float, box.strip().split()))
                cls = int(parts[0])
                xc, yc, bw, bh = parts[1:]
                
                # Convert normalized (0-1) to pixels
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                box_w = bw * w
                box_h = bh * h
                
                # Draw
                rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='#00ff00', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-5, class_names.get(cls, str(cls)), color='#00ff00', fontsize=10, fontweight='bold')
    
    # Turn off unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def eda_detect_box_aspect_ratio(yaml_path):
    """
    Analyzes width vs height of objects.
    """
    print("Analyzing Box Aspect Ratios...")
    
    try:
        _, lbl_dir, _ = _get_dataset_paths(yaml_path)
    except Exception as e:
        print(f"⚠️ Error resolving paths: {e}")
        return
    
    widths = []
    heights = []
    
    if not os.path.exists(lbl_dir):
        print(f"⚠️ Labels dir not found: {lbl_dir}")
        return

    files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
    for file in files:
        with open(os.path.join(lbl_dir, file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    w = float(parts[3])
                    h = float(parts[4])
                    widths.append(w)
                    heights.append(h)
                
    plt.figure(figsize=(8, 8))
    plt.scatter(widths, heights, alpha=0.3, color='purple')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0,1], [0,1], 'k--', alpha=0.5) 
    plt.xlabel("Box Width (Normalized)")
    plt.ylabel("Box Height (Normalized)")
    plt.title("Object Size Distribution")
    plt.grid(True, alpha=0.3)
    plt.show()
    
def eda_detect_heatmap(yaml_path):
    """
    Generates object location heatmap using Hexagonal Binning.
    Shows where objects are concentrated in the frame.
    """
    print("Generating Object Location Heatmap (Hexbin)...")
    try:
        _, lbl_dir, _ = _get_dataset_paths(yaml_path)
    except Exception as e:
        print(f"⚠️ Error resolving paths: {e}")
        return
    
    if not os.path.exists(lbl_dir):
        print(f"⚠️ Labels dir not found: {lbl_dir}")
        return

    x_centers = []
    y_centers = []
    
    files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
    for file in files:
        with open(os.path.join(lbl_dir, file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    xc = float(parts[1])
                    yc = float(parts[2])
                    x_centers.append(xc)
                    y_centers.append(yc)

    if not x_centers:
        print("No labels found to plot heatmap.")
        return

    plt.figure(figsize=(9, 7))
    
    # Gridsize: Controls number of hexagons (higher = smaller hexagons)
    # cmap: 'inferno' or 'magma' looks great for heatmaps
    hb = plt.hexbin(x_centers, y_centers, gridsize=30, cmap='inferno', mincnt=1, extent=[0, 1, 0, 1])
    
    plt.colorbar(hb, label='Count')
    plt.gca().invert_yaxis()  # Image coordinates: (0,0) is top-left
    plt.title("Object Location Heatmap (Hexbin Style)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()


#################################################################
# DETECTION EVALUATION
#################################################################

def eval_detect_curves():
    """
    Plots the training loss and mAP metrics from results.csv.
    Crucial to see if the model converged or overfitted.
    """
    results_path = os.path.join(hp.PATH_BASE, hp.FINAL_WEIGHTS_NAME, 'results.csv')
    
    if not os.path.exists(results_path):
        print(f"⚠️ Error: results.csv not found at {results_path}")
        return

    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip() # Clean column names
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Box Loss (How well it draws the box)
    # YOLOv8 usually has 'train/box_loss' and 'val/box_loss'
    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
        ax1.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', marker='.')
        ax1.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', marker='.')
        ax1.set_title("Box Regression Loss (Lower is better)")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
    # 2. mAP Metrics (How accurate the detection is)
    # mAP50 is standard, mAP50-95 is strict
    if 'metrics/mAP50(B)' in df.columns:
        ax2.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', color='green', marker='.')
        if 'metrics/mAP50-95(B)' in df.columns:
            ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95', color='orange', linestyle='--')
        
        ax2.set_title("Mean Average Precision (Higher is better)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score (0-1)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def eval_detect_performance_plots():
    """
    Displays the standard YOLO analysis plots generated during training:
    1. Confusion Matrix (What is it confusing the toolhead with?)
    2. F1-Confidence Curve (Helps pick the best threshold)
    3. Precision-Recall Curve
    """
    base_dir = os.path.join(hp.PATH_BASE, hp.FINAL_WEIGHTS_NAME)
    
    # List of plots we want to see if they exist
    plots_to_show = [
        ("Confusion Matrix", "confusion_matrix.png"),
        ("F1 Curve (Best Confidence Threshold)", "F1_curve.png"),
        ("Precision-Recall Curve", "PR_curve.png")
    ]
    
    plt.figure(figsize=(20, 6))
    
    plot_idx = 1
    found_any = False
    
    for title, filename in plots_to_show:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            found_any = True
            img = mpimg.imread(path)
            plt.subplot(1, 3, plot_idx)
            plt.imshow(img)
            plt.axis('off')
            plt.title(title)
            plot_idx += 1
            
    if not found_any:
        print("⚠️ No standard plots found. They might be generated only after a full validation run.")
    else:
        plt.show()

def eval_detect_visual_preds(num_samples=4):
    """
    Runs inference on random Validation images and displays the result.
    Uses the model's built-in plotter for clean bounding boxes.
    """
    print(f"Running inference on {num_samples} validation images...")
    
    # 1. Load Model
    best_weight_path = os.path.join(hp.PATH_BASE, hp.FINAL_WEIGHTS_NAME, 'weights', 'best.pt')
    if not os.path.exists(best_weight_path):
        print(f"⚠️ Model weights not found at {best_weight_path}")
        return
    model = YOLO(best_weight_path)

    # 2. Get Validation Images (Using our helper to resolve the path)
    try:
        _, lbl_dir, _ = _get_dataset_paths(hp.DATA_YAML)
        # Assuming images are parallel to labels, e.g., .../labels/val -> .../images/val
        # Or we can read from yaml 'val' key
        with open(hp.DATA_YAML, 'r') as f:
            data = yaml.safe_load(f)
        
        # Resolve 'val' path
        # If 'val' is relative, join with yaml root. This mimics our _get_dataset_paths logic roughly
        yaml_root = data.get('path', os.path.dirname(os.path.abspath(hp.DATA_YAML)))
        val_img_dir = os.path.join(yaml_root, data.get('val'))
        
    except Exception as e:
        print(f"⚠️ Could not resolve validation path automatically: {e}")
        return

    if not os.path.exists(val_img_dir):
        print(f"⚠️ Validation image folder not found: {val_img_dir}")
        return

    all_images = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not all_images:
        print("No images found.")
        return
        
    samples = random.sample(all_images, min(len(all_images), num_samples))
    
    # 3. Predict and Plot
    # We calculate grid size (e.g., 2x2)
    cols = 2
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    if isinstance(axes, np.ndarray): axes = axes.flatten()
    else: axes = [axes]

    for i, img_name in enumerate(samples):
        img_path = os.path.join(val_img_dir, img_name)
        
        # Run Predict
        results = model.predict(img_path, conf=hp.CONF_THRESHOLD, verbose=False)
        
        # Result.plot() returns a BGR numpy array
        res_plot = results[0].plot() 
        res_plot = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(res_plot)
        axes[i].axis('off')
        axes[i].set_title(f"Pred: {img_name}", fontsize=10)

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def eval_detect_metrics():
    """
    Runs a full validation pass to get the official numbers (mAP50, mAP50-95).
    """
    best_weight_path = os.path.join(hp.PATH_BASE, hp.FINAL_WEIGHTS_NAME, 'weights', 'best.pt')
    if not os.path.exists(best_weight_path):
        print("⚠️ Weights not found.")
        return

    print("Running Official Validation...")
    model = YOLO(best_weight_path)
    metrics = model.val(data=hp.DATA_YAML, verbose=False)
    
    print("\n=== FINAL DETECTION METRICS ===")
    print(f"mAP@50    (PASCAL VOC metric): {metrics.box.map50:.4f}")
    print(f"mAP@50-95 (COCO metric)      : {metrics.box.map:.4f}")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")


##################################################################
# PRIVATE HELPERS
##################################################################

def _get_dataset_paths(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    # 1. Determine the base directory of the dataset
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    
    # 'path' in yaml is often '.' or explicit. If missing, assume yaml_dir.
    base_path_rel = data.get('path', '.')
    
    # Resolve absolute base path
    if os.path.isabs(base_path_rel):
        base_path = base_path_rel
    else:
        base_path = os.path.normpath(os.path.join(yaml_dir, base_path_rel))

    # 2. Resolve Train Images Path
    train_rel = data.get('train') # e.g., "images/train"
    if os.path.isabs(train_rel):
        img_dir = train_rel
    else:
        img_dir = os.path.normpath(os.path.join(base_path, train_rel))
        
    # 3. Resolve Train Labels Path
    # Standard YOLO structure assumes 'labels' is parallel to 'images'
    # e.g. .../dataset/images/train -> .../dataset/labels/train
    if 'images' in img_dir:
        lbl_dir = img_dir.replace('images', 'labels')
    else:
        # Fallback if folder isn't named 'images', try specific replacement or parallel folder
        parent = os.path.dirname(img_dir)
        dir_name = os.path.basename(img_dir)
        # Assuming structure root/images/train -> root/labels/train
        lbl_dir = os.path.join(parent.replace('images', 'labels'), dir_name)

    return img_dir, lbl_dir, data