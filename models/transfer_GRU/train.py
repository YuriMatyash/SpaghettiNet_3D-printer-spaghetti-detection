import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import glob
import os
import time

# Custom modules
from dataset import PrinterFrameDataset, create_smart_sliding_windows
from model import SpaghettiNet
import hyperparams as hp

def get_data_lists():
    """
    Iterates over dataset/video_1, dataset/video_2... 
    and extracts mixed labels from each.
    """
    all_sequences = []
    all_labels = []

    # Find all video folders (video_1, video_2, etc.)
    video_folders = glob.glob(os.path.join(hp.DATASET_ROOT, "*"))
    video_folders = [f for f in video_folders if os.path.isdir(f)]
    
    print(f"Scanning {len(video_folders)} video folders in '{hp.DATASET_ROOT}'...")

    for folder in video_folders:
        seqs, lbls = create_smart_sliding_windows(folder)
        all_sequences.extend(seqs)
        all_labels.extend(lbls)
        
        # Optional: Print stats per folder
        detach_count = sum(lbls)
        print(f"  Folder '{os.path.basename(folder)}': {len(seqs)} sequences ({detach_count} detached)")

    return all_sequences, all_labels

def main():
    device = hp.DEVICE
    print(f"Training on device: {device}")
    
    # 1. Prepare Data
    sequences, labels = get_data_lists()
    
    total_len = len(sequences)
    train_len = int(hp.TRAIN_SPLIT * total_len)
    
    # Zip them together to shuffle, then unzip
    combined = list(zip(sequences, labels))
    import random
    random.shuffle(combined)
    
    train_data = combined[:train_len]
    val_data = combined[train_len:]
    
    # Unzip back into lists
    train_seqs, train_lbls = zip(*train_data)
    val_seqs, val_lbls = zip(*val_data)
    
    # Create Two Distinct Datasets
    train_dataset = PrinterFrameDataset(train_seqs, train_lbls, is_train=True)
    val_dataset = PrinterFrameDataset(val_seqs, val_lbls, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hp.BATCH_SIZE, shuffle=False)

    # 2. Model & Training Setup
    model = SpaghettiNet().to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)

    # 3. Training Loop
    for epoch in range(hp.EPOCHS):
        start_time = time.time()
        model.train() 
        train_loss = 0
        
        print(f"\n--- Epoch {epoch+1}/{hp.EPOCHS} ---")
        
        for batch_idx, (videos, targets) in enumerate(train_loader):
            videos, targets = videos.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos) 
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Validation
        model.eval() 
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for videos, targets in val_loader:
                videos, targets = videos.to(device), targets.to(device)
                outputs = model(videos)
                loss = criterion(outputs, targets.unsqueeze(1))
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                total += targets.size(0)
                correct += (predicted.squeeze() == targets).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        accuracy = 100 * correct / total if total > 0 else 0
        
        print(f"Epoch Time: {time.time() - start_time:.1f}s")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.2f}%")

    print(f"Saving model to {hp.MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), hp.MODEL_SAVE_PATH)
    print("Done!")

if __name__ == "__main__":
    main()