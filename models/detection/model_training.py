from ultralytics import YOLO
import torch
import os

# ================= CONFIGURATION =================
# 1. Model Selection
MODEL_NAME = 'yolov12n.pt'  

# 2. Data Path (Absolute path is safest)
DATA_YAML = "datasets/detection/dataset/dataset.yaml"

# 3. Training Parameters
IMG_SIZE = 224          # Matching your resized images
EPOCHS = 100            # High epochs because dataset is small
BATCH_SIZE = 16         # Adjust based on your VRAM (16 or 32 is usually safe for 224x224)
PROJECT_NAME = "SpaghettiNet_Toolhead"
# =================================================

def main():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nFound GPU: {gpu_name}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
        device_id = 0
    else:
        print("\nGPU not found. Training will use CPU (Very Slow)!\n")
        device_id = 'cpu'
    
    try:
        print(f"Attempting to load {MODEL_NAME}...")
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"Could not load {MODEL_NAME} (might be too new). Falling back to yolo11n.pt")
        model = YOLO('yolo11n.pt')

    print("Starting training...")
    
    # Train the model
    results = model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name='finetune_onnx_v1',
        device=device_id,
        
        # === Augmentations for Sparse Data ===
        mosaic=1.0,        # vital for small datasets
        mixup=0.1,         
        degrees=15.0,      
        translate=0.1,     
        scale=0.5,         
        fliplr=0.5,
        
        # === Optimization ===
        patience=20,       
        exist_ok=True,     
        verbose=True
    )

    print("Training complete.")
    
    # Evaluate performance
    print("Evaluating on test set...")
    metrics = model.val(split='test')
    print(f"mAP50: {metrics.box.map50}")

    # === EXPORT TO ONNX ===
    print("Exporting to ONNX...")
    # format='onnx': The target format
    # dynamic=True: Allows inputting images of different sizes (optional, but good practice)
    # opset=12: Good compatibility standard (optional)
    path = model.export(format='onnx', dynamic=True)
    
    print(f"ONNX model saved to: {path}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()