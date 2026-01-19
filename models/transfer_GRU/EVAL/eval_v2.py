import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torchvision.transforms.functional as TF
import random
import os

# Custom modules
from dataset import PrinterFrameDataset
from model import SpaghettiNet
import hyperparams as hp
import utils

def stress_test():
    device = hp.DEVICE
    print(f"--- 🏋️ Starting Robustness Stress Test ---")

    # 1. Load Data (Use specific folder or all data)
    sequences, labels = utils.get_data_lists() #
    
    # Optional: Limit to a smaller subset for speed
    if len(sequences) > 200:
        combined = list(zip(sequences, labels))
        random.shuffle(combined)
        sequences, labels = zip(*combined[:200]) # Test on 200 random windows
    
    print(f"Testing on {len(sequences)} sequences per scenario.\n")

    # 2. Load Model
    model = SpaghettiNet() #
    try:
        model.load_state_dict(torch.load(hp.MODEL_SAVE_PATH, map_location=device))
    except FileNotFoundError:
        print("Model weights not found.")
        return
    model.to(device)
    model.eval()

    # 3. Define Scenarios (The consistent augmentations)
    scenarios = {
        "Baseline (No Change)": lambda img: img,
        "🌑 Low Light (0.5x)":  lambda img: TF.adjust_brightness(img, 0.5), # 50% darker
        "☀️ High Contrast":     lambda img: TF.adjust_contrast(img, 2.0),   # Double contrast
        "🔄 Tilted (+15°)":     lambda img: TF.rotate(img, 15)              # Camera rotated
    }

    # 4. Run the Gauntlet
    results = {}

    for name, transform_func in scenarios.items():
        print(f"Running Scenario: {name}...", end=" ")
        
        y_true = []
        y_pred = []
        
        # We process manually to inject the transform BEFORE the model sees it
        dataset = PrinterFrameDataset(sequences, labels, is_train=False) #
        loader = DataLoader(dataset, batch_size=hp.BATCH_SIZE, shuffle=False)
        
        with torch.no_grad():
            for videos, targets in loader:
                # videos shape: (Batch, 16, 3, 224, 224)
                
                # --- APPLY CONSISTENT TRANSFORM ---
                # We must apply the function to every frame in the batch equally
                # Flatten to (Batch*16, 3, 224, 224) for easy processing
                b, seq, c, h, w = videos.shape
                flat_videos = videos.view(-1, c, h, w)
                
                # Apply transform to all frames at once
                aug_videos = transform_func(flat_videos)
                
                # Reshape back to (Batch, 16, 3, 224, 224)
                videos = aug_videos.view(b, seq, c, h, w).to(device)
                
                # --- INFERENCE ---
                outputs = model(videos)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                y_true.extend(targets.numpy())
                y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        results[name] = acc
        print(f"Done. Accuracy: {acc:.1%}")

    # 5. Print Final Report
    print("\n" + "="*40)
    print("📊 ROBUSTNESS SCORECARD")
    print("="*40)
    print(f"{'Scenario':<25} | {'Accuracy':<10} | {'Drop'}")
    print("-" * 45)
    
    baseline = results["Baseline (No Change)"]
    
    for name, score in results.items():
        drop = baseline - score
        print(f"{name:<25} | {score:.1%}    | -{drop:.1%}")
    print("="*40)

if __name__ == "__main__":
    stress_test()