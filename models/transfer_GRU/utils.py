import os
import glob
import re

import hyperparams as hp


"""
Iterates over dataset/video_1, dataset/video_2... 
and extracts mixed labels from each.
"""
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

