import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import hyperparams as hp

class SingleFrameFeatureExtractor:
    def __init__(self, device=None) -> None:
        self.device = hp.DEVICE
        self.transform = hp.IMG_TRANSFORMS

        # load model
        print(f"Loading MobileNetV3-Small on {self.device}")
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=weights)

        # remove output layer, to get features only
        self.model.classifier = nn.Identity()
        
        # freeze weights so to not train
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.to(self.device)
        self.model.eval()

    # input a single frame (OpenCV BGR format), output feature tensor
    def extract(self, frame_bgr) -> torch.Tensor:
        # convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # apply transforms
        img_tensor = self.transform(pil_img)
        
        # add batch dimension because model expects (N, C, H, W)
        input_batch = img_tensor.unsqueeze(0).to(self.device)
        
        # forward pass
        with torch.no_grad():
            features = self.model(input_batch)
            
        # Output is (1, 576), contains feature vector for the frame
        return features