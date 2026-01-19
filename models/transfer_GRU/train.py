import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import os
import time

# Custom modules
from dataset import PrinterFrameDataset
from model import SpaghettiNet
import hyperparams as hp
from utils import get_data_lists, plot_training_graph, plot_confusion_matrix

def main():
    device = hp.DEVICE
    print(f"Training on device: {device}")
    
    # 1. Prepare Data
    sequences, labels = get_data_lists()
    
    total_len = len(sequences)
    train_len = int(hp.TRAIN_SPLIT * total_len)
    
    # Zip, shuffle, unzip
    combined = list(zip(sequences, labels))
    import random
    random.shuffle(combined)
    
    train_data = combined[:train_len]
    val_data = combined[train_len:]
    
    train_seqs, train_lbls = zip(*train_data)
    val_seqs, val_lbls = zip(*val_data)
    
    # Create Datasets & Loaders
    train_dataset = PrinterFrameDataset(train_seqs, train_lbls, is_train=True)
    val_dataset = PrinterFrameDataset(val_seqs, val_lbls, is_train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=hp.BATCH_SIZE, 
        shuffle=True,
        num_workers=hp.NUM_WORKERS,
        pin_memory=hp.PIN_MEMORY,
        persistent_workers=hp.PERSISTENT_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=hp.BATCH_SIZE, 
        shuffle=False,
        num_workers=hp.NUM_WORKERS,
        pin_memory=hp.PIN_MEMORY,
        persistent_workers=hp.PERSISTENT_WORKERS
    )

    # 2. Model Setup
    model = SpaghettiNet().to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)
    
    # Modern PyTorch 2.x GradScaler
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # History Tracking
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 3. Training Loop
    print(f"Starting training for {hp.EPOCHS} epochs...")
    
    for epoch in range(hp.EPOCHS):
        start_time = time.time()
        model.train() 
        running_train_loss = 0
        
        # --- TRAIN STEP ---
        for batch_idx, (videos, targets) in enumerate(train_loader):
            videos, targets = videos.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(videos) 
                loss = criterion(outputs, targets.unsqueeze(1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_train_loss += loss.item()
            
        # --- VALIDATION STEP ---
        model.eval() 
        running_val_loss = 0
        correct = 0
        total = 0
        
        # Lists to store predictions for this epoch (optional, but good for debug)
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for videos, targets in val_loader:
                videos, targets = videos.to(device), targets.to(device)
                
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model(videos)
                    loss = criterion(outputs, targets.unsqueeze(1))
                
                running_val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                
                # Store for metrics
                total += targets.size(0)
                correct += (predicted.squeeze() == targets).sum().item()
                
                # If this is the LAST epoch, save them for the Confusion Matrix
                if epoch == hp.EPOCHS - 1:
                    val_preds.extend(predicted.cpu().numpy().flatten())
                    val_targets.extend(targets.cpu().numpy().flatten())

        # Calculate Averages
        avg_train_loss = running_train_loss / len(train_loader)
        avg_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        accuracy = 100 * correct / total if total > 0 else 0
        
        # Store History
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)
        
        # Print Stats
        epoch_dur = time.time() - start_time
        print(f"Epoch {epoch+1}/{hp.EPOCHS} ({epoch_dur:.1f}s) | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {accuracy:.2f}%")

    # 4. Save Model & Graphs
    print(f"\nSaving model to {hp.MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), hp.MODEL_SAVE_PATH)
    
    print("Generating Training Graph...")
    plot_training_graph(train_losses, val_losses, val_accuracies)
    
    print("Generating Confusion Matrix...")
    # We use the predictions from the final validation pass
    plot_confusion_matrix(val_targets, val_preds)
    
    print("All Done!")

if __name__ == "__main__":
    main()