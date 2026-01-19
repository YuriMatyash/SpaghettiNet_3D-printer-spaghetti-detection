import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

# Custom modules
from dataset import PrinterFrameDataset
from model import SpaghettiNet
import hyperparams as hp
from utils import get_data_lists

def evaluate():
    # 1. Setup
    device = hp.DEVICE
    print(f"--- Evaluating Model on {device} ---")
    
    # Load Data
    sequences, labels = get_data_lists()
    
    # NOTE: To rigorously test, we should use data the model has NEVER seen.
    # Since train.py shuffled randomly, we'll just use a random 20% split here 
    # to simulate a test set.
    combined = list(zip(sequences, labels))
    random.shuffle(combined)
    
    split_idx = int(len(combined) * 0.2) # Use 20% for testing
    test_data = combined[:split_idx]
    
    test_seqs, test_lbls = zip(*test_data)
    test_dataset = PrinterFrameDataset(test_seqs, test_lbls, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=hp.BATCH_SIZE, shuffle=False)
    
    print(f"Testing on {len(test_dataset)} sequences...")

    # 2. Load Model
    model = SpaghettiNet()
    try:
        model.load_state_dict(torch.load(hp.MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded weights from {hp.MODEL_SAVE_PATH}")
    except FileNotFoundError:
        print("Model file not found! Train the model first.")
        return

    model.to(device)
    model.eval()

    # 3. Inference Loop
    y_true = []
    y_scores = [] # Raw probabilities
    y_pred = []   # Thresholded predictions (0 or 1)

    with torch.no_grad():
        for videos, targets in test_loader:
            videos = videos.to(device)
            
            # Forward pass
            outputs = model(videos)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Store results
            y_scores.extend(probs)
            y_pred.extend((probs > 0.5).astype(int))
            y_true.extend(targets.numpy())

    # Flatten lists
    y_scores = np.array(y_scores).flatten()
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()

    # 4. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n" + "="*30)
    print(f"✅ Accuracy:  {acc:.2%}")
    print(f"🎯 Precision: {prec:.4f} (Low False Alarms)")
    print(f"🔍 Recall:    {rec:.4f} (Caught Failures)")
    print(f"⚖️ F1-Score:  {f1:.4f}")
    print("="*30)

    # 5. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Detached'], yticklabels=['Normal', 'Detached'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('eval_confusion_matrix.png')
    print("Saved confusion matrix to eval_confusion_matrix.png")

    # 6. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('eval_roc_curve.png')
    print("Saved ROC curve to eval_roc_curve.png")

if __name__ == "__main__":
    evaluate()