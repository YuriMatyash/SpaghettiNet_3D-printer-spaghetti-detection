import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import random
import torchvision.transforms.functional as TF 
import hyperparams as hp

class PrinterFrameDataset(Dataset):
    def __init__(self, sequences, labels, is_train=True):
        self.sequences = sequences
        self.labels = labels
        self.is_train = is_train
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        image_paths = self.sequences[idx]
        label = self.labels[idx]
        
        # defaults
        do_flip = False
        do_blur = False
        angle = 0.0
        bright_factor = 1.0
        contrast_factor = 1.0
        sat_factor = 1.0
        hue_factor = 0.0

        if self.is_train:
            # Flip?
            if random.random() < hp.AUG_FLIP_PROB:
                do_flip = True
            
            # Blur? (Simulate out-of-focus camera)
            if random.random() < hp.AUG_BLUR_PROB:
                do_blur = True
                
            # Rotation
            angle = random.uniform(-hp.AUG_ROTATION_DEGREES, hp.AUG_ROTATION_DEGREES)
            
            # Color Jitter
            bright_factor = random.uniform(1 - hp.AUG_BRIGHTNESS, 1 + hp.AUG_BRIGHTNESS)
            contrast_factor = random.uniform(1 - hp.AUG_CONTRAST, 1 + hp.AUG_CONTRAST)
            sat_factor = random.uniform(1 - hp.AUG_SATURATION, 1 + hp.AUG_SATURATION)
            hue_factor = random.uniform(-hp.AUG_HUE, hp.AUG_HUE)

        # apply to all images in the sequence
        tensors = []
        for img_path in image_paths:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                tensors.append(torch.zeros(3, hp.IMG_SIZE, hp.IMG_SIZE))
                continue
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            pil_img = TF.resize(pil_img, (hp.IMG_SIZE, hp.IMG_SIZE))
            
            if self.is_train:
                # Apply consistent augmentations
                if do_flip:
                    pil_img = TF.hflip(pil_img)
                    
                if do_blur:
                    # kernel_size must be odd (3, 5, 7)
                    pil_img = TF.gaussian_blur(pil_img, kernel_size=hp.AUG_BLUR_KERNEL)
                
                # Geometry
                pil_img = TF.rotate(pil_img, angle)
                
                # Color
                pil_img = TF.adjust_brightness(pil_img, bright_factor)
                pil_img = TF.adjust_contrast(pil_img, contrast_factor)
                pil_img = TF.adjust_saturation(pil_img, sat_factor)
                pil_img = TF.adjust_hue(pil_img, hue_factor)

            # Convert to Tensor & Normalize
            img_tensor = TF.to_tensor(pil_img)
            img_tensor = TF.normalize(img_tensor, mean=hp.IMG_MEAN, std=hp.IMG_STD)
            
            tensors.append(img_tensor)
            
        return torch.stack(tensors), torch.tensor(label, dtype=torch.float32)