import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import os
import glob
import re
import hyperparams as hp

class PrinterFrameDataset(Dataset):
    def __init__(self, sequences, labels, is_train=True):
        self.sequences = sequences
        self.labels = labels
        
        # Select the correct transform based on mode
        if is_train:
            self.transform = hp.TRAIN_TRANSFORMS
        else:
            self.transform = hp.VAL_TRANSFORMS

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        image_paths = self.sequences[idx]
        label = self.labels[idx]
        
        tensors = []
        for img_path in image_paths:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                tensors.append(torch.zeros(3, hp.IMG_SIZE, hp.IMG_SIZE))
                continue
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensors.append(self.transform(pil_img))
            
        return torch.stack(tensors), torch.tensor(label, dtype=torch.float32)

# SMART SLIDING WINDOW GENERATOR
def create_smart_sliding_windows(folder_path):
    """
    Scans a folder, finds the 'detach' frame, and generates labeled windows.
    Returns:
        samples: List of list of paths
        labels: List of ints (0 or 1)
    """
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