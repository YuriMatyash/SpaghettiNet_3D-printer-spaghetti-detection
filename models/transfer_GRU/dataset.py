import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
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