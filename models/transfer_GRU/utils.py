import os
import glob
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import hyperparams as hp


# iterates through the dataset folders, creates sliding windows, and labels them
def get_data_lists() -> tuple[list[list[str]], list[int]]:
    
    all_sequences = []
    all_labels = []

    # Find all video folders (video_1, video_2, etc.)
    video_folders = glob.glob(os.path.join(hp.DATASET_ROOT, "*"))
    video_folders = [f for f in video_folders if os.path.isdir(f)]
    
    print(f"Scanning {len(video_folders)} video folders in '{hp.DATASET_ROOT}'...")

    for folder in video_folders:
        seqs, lbls = _create_smart_sliding_windows(folder)
        all_sequences.extend(seqs)
        all_labels.extend(lbls)
        
        # Optional: Print stats per folder
        detach_count = sum(lbls)
        print(f"  Folder '{os.path.basename(folder)}': {len(seqs)}  sequences ({detach_count} detached)")

    return all_sequences, all_labels

def plot_training_graph(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Left Axis: Loss
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    l1 = ax1.plot(epochs, train_losses, color=color, linestyle='--', label='Train Loss')
    l2 = ax1.plot(epochs, val_losses, color='tab:orange', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Right Axis: Accuracy
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    l3 = ax2.plot(epochs, val_accuracies, color=color, linewidth=2, label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title('Training Performance: Loss & Accuracy')
    plt.tight_layout()
    plt.savefig('training_graph.png', dpi=300)
    print("   Saved 'training_graph.png'")

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Detached'], 
                yticklabels=['Normal', 'Detached'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Final Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("   Saved 'confusion_matrix.png'")


#############################################################################
# Private methods
##############################################################################

"""
Scans a folder, finds the 'detach' frame, and generates labeled windows.
Returns:
    samples: list[list[str]] - It is a list containing windows, where each window is a list of file paths
    labels: list[int] - it is a list of integer labels(0 or 1) corresponding to each window
"""
def _create_smart_sliding_windows(folder_path) -> tuple[list[list[str]], list[int]]:

    # 1. Get all images
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # 2. Sort numerically (1.jpg, 2.jpg, 26-detach.jpg, 100.jpg)
    # We extract the FIRST number we see in the filename
    def sort_key(fname):
        base = os.path.basename(fname)
        numbers = re.findall(r'\d+', base)
        return int(numbers[0]) if numbers else 0
        
    images = sorted(images, key=sort_key)
    
    if len(images) < hp.SEQ_LEN:
        return [], []

    # 3. Find the Detachment Index
    detachment_start_index = float('inf') # Default: never detaches
    
    for idx, img_path in enumerate(images):
        filename = os.path.basename(img_path).lower()
        if "detach" in filename:
            detachment_start_index = idx
            break
            
    # 4. Generate Windows
    samples = []
    labels = []
    num_sequences = len(images) - hp.SEQ_LEN + 1

    for i in range(num_sequences):
        # Window of 16 frames
        window_paths = images[i : i + hp.SEQ_LEN]
        
        # Determine the index of the LAST frame in this window
        last_frame_idx = i + hp.SEQ_LEN - 1
        
        # Label Logic:
        # If the window includes or passes the detachment frame, it's a failure.
        if last_frame_idx >= detachment_start_index:
            label = 1
        else:
            label = 0
            
        samples.append(window_paths)
        labels.append(label)
        
    return samples, labels

