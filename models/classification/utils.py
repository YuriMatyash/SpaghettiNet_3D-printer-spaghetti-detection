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

import hyperparams as hp

##################################################################
# DATASET PREPARATION FUNCTIONS
##################################################################

# Setup final dataset directories for the yolo format
def setup_dirs():
    # Clean the destination to avoid mixing old data
    if os.path.exists(hp.PATH_FINAL_DATASET):
        print(f"Refreshing final dataset directory: {hp.PATH_FINAL_DATASET}")
        shutil.rmtree(hp.PATH_FINAL_DATASET)
    
    # Create the new folders
    for split in ['train', 'val']:
        for class_name in ['clean', 'spaghetti']:
            # constructs: datasets/classification/final/train/clean, etc.
            dir_path = os.path.join(hp.PATH_FINAL_DATASET, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created: {dir_path}")


# split images from "resized" folders(final data) to final dataset folder with split
def split_dataset():
    # Map the class names to their source folders (from your hyperparams.py)
    sources = {
        'clean': hp.PATH_CLEAN_RESIZED,
        'spaghetti': hp.PATH_SPAGHETTI_RESIZED
    }

    for class_name, source_dir in sources.items():
        if not os.path.exists(source_dir):
            print(f"Warning: Source folder missing {source_dir}")
            continue

        # Get all valid images
        images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Shuffle images for randomness
        random.shuffle(images)
        
        # Calculate split point
        split_idx = int(len(images) * hp.SPLIT_RATIO)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        print(f"Processing '{class_name}': {len(train_imgs)} Train, {len(val_imgs)} Val")

        # Copy function to reduce repetition
        # takes a list of images and a split type ('train' or 'val')
        def copy_to_split(image_list, split_type):
            destination_folder = os.path.join(hp.PATH_FINAL_DATASET, split_type, class_name)
            for img_name in image_list:
                src_path = os.path.join(source_dir, img_name)
                dst_path = os.path.join(destination_folder, img_name)
                shutil.copy2(src_path, dst_path)

        # Execute Copy
        copy_to_split(train_imgs, 'train')
        copy_to_split(val_imgs, 'val')


#################################################################
# EDA FUNCTIONS
#################################################################

def eda_distribution():
    """
    Uses explicit hyperparam paths to count files and plot distribution.
    Includes a safety check for empty datasets.
    """
    print("Generating Class Distribution Plots...")
    
    # 1. Fetch counts
    n_clean_train = _count_images(hp.PATH_CLEAN_TRAIN)
    n_spag_train = _count_images(hp.PATH_SPAGHETTI_TRAIN)
    n_clean_val = _count_images(hp.PATH_CLEAN_VAL)
    n_spag_val = _count_images(hp.PATH_SPAGHETTI_VAL)

    total_imgs = n_clean_train + n_spag_train + n_clean_val + n_spag_val

    # --- SAFETY CHECK ---
    if total_imgs == 0:
        print("⚠️ ERROR: No images found in the 'final' dataset folders.")
        print(f"Checked path: {hp.PATH_FINAL_DATASET}")
        print("Did you run 'setup_dirs()' and 'split_dataset()' yet?")
        return
    # --------------------

    # 2. Prepare Data
    classes = ['Clean', 'Spaghetti']
    train_counts = [n_clean_train, n_spag_train]
    val_counts = [n_clean_val, n_spag_val]

    # 3. Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar Chart
    x = np.arange(len(classes))
    width = 0.35
    ax1.bar(x - width/2, train_counts, width, label='Train', color='#3498db')
    ax1.bar(x + width/2, val_counts, width, label='Validation', color='#e74c3c')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_title("Train vs Validation Split")
    ax1.legend()
    
    # Pie Chart
    total_clean = n_clean_train + n_clean_val
    total_spag = n_spag_train + n_spag_val
    ax2.pie([total_clean, total_spag], labels=classes, autopct='%1.1f%%', 
            colors=['#2ecc71', '#f1c40f'], startangle=90)
    ax2.set_title("Total Class Balance")
    
    plt.tight_layout()
    plt.show()
    
def eda_samples(num_samples=5):
    """
    Shows random samples specifically from the TRAIN folders defined in hyperparams.
    """
    print(f"Showing {num_samples} random samples per class from training set...")
    
    # Explicitly map name to the HP variable
    class_paths = {
        'Clean': hp.PATH_CLEAN_TRAIN,
        'Spaghetti': hp.PATH_SPAGHETTI_TRAIN
    }
    
    fig, axes = plt.subplots(len(class_paths), num_samples, figsize=(15, 6))
    fig.suptitle(f"Random Training Samples", fontsize=16)

    for i, (name, path) in enumerate(class_paths.items()):
        files = _get_files(path)
        if not files: continue
        
        samples = random.sample(files, min(len(files), num_samples))
        
        for j, img_path in enumerate(samples):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0: axes[i, j].set_title(name, fontweight='bold', x=-0.2, y=0.5)

    plt.tight_layout()
    plt.show()

def eda_average_image(limit=100):
    """
    Computes 'Average Image' using images from hp.PATH_CLEAN_TRAIN and hp.PATH_SPAGHETTI_TRAIN.
    """
    print("Generating Average (Mean) Image analysis...")
    
    # Explicit mapping
    class_paths = {
        'Clean': hp.PATH_CLEAN_TRAIN,
        'Spaghetti': hp.PATH_SPAGHETTI_TRAIN
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for idx, (name, path) in enumerate(class_paths.items()):
        files = _get_files(path)
        # Limit to avoid slow processing
        files = files[:limit] 
        
        if not files: continue
        
        # Initialize
        avg_img = np.zeros((hp.IMG_SIZE, hp.IMG_SIZE, 3), float)
        count = 0
        
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                img = cv2.resize(img, (hp.IMG_SIZE, hp.IMG_SIZE))
                avg_img += img
                count += 1
            
        if count > 0:
            avg_img = avg_img / count
            avg_img = np.array(avg_img, dtype=np.uint8)
            avg_img = cv2.cvtColor(avg_img, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(avg_img)
            axes[idx].set_title(f"Average '{name}' Structure")
            axes[idx].axis('off')

    plt.show()

def eda_brightness():
    """
    Plots brightness density using hp.PATH_CLEAN_TRAIN and hp.PATH_SPAGHETTI_TRAIN.
    """
    print("Calculating brightness distribution...")
    plt.figure(figsize=(10, 6))
    
    # Explicit mapping
    data_sources = [
        ('Clean', hp.PATH_CLEAN_TRAIN, '#2ecc71'),
        ('Spaghetti', hp.PATH_SPAGHETTI_TRAIN, '#e74c3c')
    ]
    
    for name, path, color in data_sources:
        files = _get_files(path)
        # Sample max 300 images
        files = random.sample(files, min(len(files), 300))
        
        brightness_values = []
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # Append average Value (brightness)
                brightness_values.append(hsv[:,:,2].mean())
            
        sns.kdeplot(brightness_values, fill=True, label=name, color=color, alpha=0.3)
        
    plt.title("Brightness Distribution")
    plt.xlabel("Average Pixel Intensity (0-255)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    
##################################################################
# EVALUATION FUNCTIONS
##################################################################

def eval_training_curves():
    """
    Reads the 'results.csv' generated by YOLO training and plots:
    1. Train/Val Loss
    2. Top-1 Accuracy over epochs
    """
    # Construct path to results.csv
    # Path: models/classification/Final_Classification_Results/results.csv
    results_path = os.path.join(hp.PATH_BASE, hp.FINAL_WEIGHTS_NAME, 'results.csv')
    
    if not os.path.exists(results_path):
        print(f"⚠️ Error: Could not find results file at {results_path}")
        print("Did you finish training successfully?")
        return

    # Read CSV (YOLO CSVs sometimes have weird spacing, so we strip names)
    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()

    print(f"Plotting training results from: {results_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Loss Plot
    # different versions of YOLO name columns differently, we try to catch both
    train_loss_col = 'train/loss' if 'train/loss' in df.columns else 'train/box_loss' # fallback
    val_loss_col = 'val/loss' if 'val/loss' in df.columns else 'val/box_loss'
    
    # Classification usually has 'train/loss' or 'train/ce_loss' (Cross Entropy)
    # Let's verify columns for classification specifically
    possible_loss_cols = [c for c in df.columns if 'loss' in c and 'train' in c]
    possible_val_loss_cols = [c for c in df.columns if 'loss' in c and 'val' in c]
    
    if possible_loss_cols and possible_val_loss_cols:
        ax1.plot(df['epoch'], df[possible_loss_cols[0]], label='Train Loss', marker='.')
        ax1.plot(df['epoch'], df[possible_val_loss_cols[0]], label='Val Loss', marker='.')
        ax1.set_title("Loss over Epochs (Lower is better)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Accuracy Plot
    if 'metrics/accuracy_top1' in df.columns:
        ax1.set_title("Loss Curves")
        ax2.plot(df['epoch'], df['metrics/accuracy_top1'] * 100, label='Top-1 Accuracy', color='green', marker='o')
        ax2.set_title("Validation Accuracy (Higher is better)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def eval_validate_model():
    """
    Loads the BEST saved model and runs a full validation pass 
    to get the final official metrics.
    """
    # Path: models/classification/Final_Classification_Results/weights/best.pt
    best_weight_path = os.path.join(hp.PATH_BASE, hp.FINAL_WEIGHTS_NAME, 'weights', 'best.pt')
    
    if not os.path.exists(best_weight_path):
        print(f"⚠️ Error: Model weights not found at {best_weight_path}")
        return

    print(f"Loading best model from: {best_weight_path}")
    model = YOLO(best_weight_path)
    
    print("\nStarting Validation on Validation Set...")
    metrics = model.val(data=hp.PATH_FINAL_DATASET)
    
    print("\n=== FINAL METRICS ===")
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")
    return metrics

def eval_visual_predictions(num_samples=6):
    """
    Runs the model on random images from the Validation set and displays
    the image, the true label, and the model's prediction + confidence.
    """
    # Path: models/classification/Final_Classification_Results/weights/best.pt
    best_weight_path = os.path.join(hp.PATH_BASE, hp.FINAL_WEIGHTS_NAME, 'weights', 'best.pt')
    model = YOLO(best_weight_path)
    
    # Gather validation images
    # We want a mix of clean and spaghetti
    clean_files = _get_files(hp.PATH_CLEAN_VAL)
    spag_files = _get_files(hp.PATH_SPAGHETTI_VAL)
    
    # Pick random samples (half clean, half spaghetti ideally)
    samples = []
    if clean_files: samples.extend(random.sample(clean_files, min(3, len(clean_files))))
    if spag_files: samples.extend(random.sample(spag_files, min(3, len(spag_files))))
    random.shuffle(samples)
    
    # Predict and Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Model Predictions on Unseen Validation Data", fontsize=16)
    axes = axes.flatten()
    
    for i, img_path in enumerate(samples[:6]): # Limit to 6 plots
        # Run inference
        results = model.predict(img_path, verbose=False)[0]
        
        # Get True Label from path (folder name)
        # e.g. .../val/spaghetti/img.jpg -> 'spaghetti'
        true_label = os.path.basename(os.path.dirname(img_path))
        
        # Get Prediction
        probs = results.probs
        pred_index = probs.top1
        pred_label = results.names[pred_index]
        confidence = probs.top1conf.item() * 100
        
        # Color code: Green if correct, Red if wrong
        color = 'green' if pred_label == true_label else 'red'
        
        # Plot
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)", 
                          color=color, fontweight='bold')

    plt.tight_layout()
    plt.show()

def eval_confusion_matrix():
    """
    Displays the Confusion Matrix generated by YOLO during training.
    """
    # YOLO automatically saves 'confusion_matrix.png' in the run folder
    # Path: models/classification/Final_Classification_Results/confusion_matrix.png
    matrix_path = os.path.join(hp.PATH_BASE, hp.FINAL_WEIGHTS_NAME, 'confusion_matrix.png')
    
    # Sometimes it saves as 'confusion_matrix_normalized.png', let's check both
    if not os.path.exists(matrix_path):
        matrix_path = os.path.join(hp.PATH_BASE, hp.FINAL_WEIGHTS_NAME, 'confusion_matrix_normalized.png')

    if os.path.exists(matrix_path):
        print(f"Loading Confusion Matrix from: {matrix_path}")
        img = cv2.imread(matrix_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Confusion Matrix")
        plt.show()
    else:
        print(f"⚠️ Warning: Confusion matrix image not found at {matrix_path}")
        print("It might be generated only after a full validation run.")

def eval_metrics_table():
    """
    Generates a detailed Classification Report (Precision, Recall, F1-Score)
    by running the model on the Validation set.
    """
    print("Generating detailed metrics table (Precision, Recall, F1)...")
    
    # Load Model
    best_weight_path = os.path.join(hp.PATH_BASE, hp.FINAL_WEIGHTS_NAME, 'weights', 'best.pt')
    if not os.path.exists(best_weight_path):
        print("⚠️ Error: Model weights not found.")
        return
    model = YOLO(best_weight_path)

    # 1. Collect Predictions and True Labels
    # We iterate through the validation folders defined in hyperparams
    y_true = []
    y_pred = []
    
    # Map class names to IDs (usually clean=0, spaghetti=1, but we check model.names)
    # YOLO stores class names in model.names dictionary
    class_names = model.names
    # Invert mapping: {'clean': 0, 'spaghetti': 1}
    class_map = {v: k for k, v in class_names.items()}

    # List of validation paths to iterate
    val_dirs = {
        'clean': hp.PATH_CLEAN_VAL,
        'spaghetti': hp.PATH_SPAGHETTI_VAL
    }

    for class_label, dir_path in val_dirs.items():
        if not os.path.exists(dir_path): continue
        
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png'))]
        
        # Run inference in batches (faster) or one by one
        # For simplicity in evaluation, we run predict on the list of files
        if len(files) > 0:
            results = model.predict(files, verbose=False)
            
            for res in results:
                # Appending True Label ID
                y_true.append(class_map[class_label])
                # Appending Predicted Label ID
                y_pred.append(res.probs.top1)

    # 2. Generate Report
    report_dict = classification_report(
        y_true, 
        y_pred, 
        target_names=[class_names[i] for i in range(len(class_names))], 
        output_dict=True
    )
    
    # 3. Convert to Pandas DataFrame for a pretty table
    df = pd.DataFrame(report_dict).transpose()
    
    # Formatting: Round numbers and nice columns
    df = df.round(4)
    print("\n=== Detailed Performance Metrics ===")
    display(df) # Jupyter magic to show table

##################################################################
# PRIVATE HELPERS
##################################################################

def _count_images(path):
    """Helper: Counts images in a specific directory path."""
    if not os.path.exists(path): return 0
    return len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))])

def _get_files(path):
    """Helper: Returns full file paths from a directory."""
    if not os.path.exists(path): return []
    return [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))]