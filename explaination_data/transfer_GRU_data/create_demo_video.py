import torch
import cv2
import os
import glob
import re
from PIL import Image

# Custom modules
from model import SpaghettiNet
import hyperparams as hp

def create_demo():
    # --- CONFIGURATION ---
    TARGET_FOLDER = "datasets/transfer_GRU/dataset/video_2"
    OUTPUT_FILE = "demo_inference.mp4"
    DETACHMENT_FRAME_NUM = 26
    FRAMES_TO_RECORD = 16
    OUTPUT_FPS = 1
    
    # Calculate start frame to center the detachment
    # We want detachment (26) to be roughly in the middle of the 16 frames
    # Start = 26 - 8 = 18
    START_RECORDING_AT_NUM = DETACHMENT_FRAME_NUM - (FRAMES_TO_RECORD // 2)
    
    device = hp.DEVICE
    print(f"--- Creating Demo Video from {TARGET_FOLDER} ---")
    print(f"Target Window: Frames {START_RECORDING_AT_NUM} to {START_RECORDING_AT_NUM + FRAMES_TO_RECORD - 1}")

    # 1. Load Model
    model = SpaghettiNet()
    try:
        model.load_state_dict(torch.load(hp.MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded weights from {hp.MODEL_SAVE_PATH}")
    except FileNotFoundError:
        print("Error: Model weights not found.")
        return

    model.to(device)
    model.eval()

    # 2. Get and Sort Images
    # We use the same sorting logic as utils.py to ensure correct order
    def sort_key(fname):
        base = os.path.basename(fname)
        numbers = re.findall(r'\d+', base)
        return int(numbers[0]) if numbers else 0

    extensions = ["*.jpg", "*.jpeg", "*.png"]
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(TARGET_FOLDER, ext)))
    
    images = sorted(images, key=sort_key)
    
    if not images:
        print("No images found in folder!")
        return

    # 3. Setup Video Writer
    # Read first image to get dimensions
    first_frame = cv2.imread(images[0])
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID'
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, OUTPUT_FPS, (width, height))

    # 4. Inference Loop
    print("Processing frames...")
    
    # We must reset memory before starting
    model.reset_memory()

    for img_path in images:
        # Extract frame number from filename for logic
        frame_num = sort_key(img_path)
        
        # A. Preprocess
        frame_bgr = cv2.imread(img_path)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Apply transforms (Resize -> Tensor -> Normalize)
        input_tensor = hp.IMG_TRANSFORMS(pil_img).unsqueeze(0).to(device)

        # B. Run Inference (Stateful)
        # We run this on EVERY frame to keep the GRU memory updated
        with torch.no_grad():
            prob = model.predict_live_frame(input_tensor).item()

        # C. Recording Logic
        # Only save frames within our target 16-frame window
        if frame_num >= START_RECORDING_AT_NUM and frame_num < (START_RECORDING_AT_NUM + FRAMES_TO_RECORD):
            
            # --- VISUALIZATION ---
            display_frame = frame_bgr.copy()
            
            is_detached = prob > hp.ALARM_THRESHOLD
            color = (0, 0, 255) if is_detached else (0, 255, 0) # Red or Green
            status_text = "DETACHED" if is_detached else "Normal"
            
            # 1. Thick Border
            thickness = 15
            cv2.rectangle(display_frame, (0, 0), (width-1, height-1), color, thickness)
            
            # 2. Text Overlay
            # Probability
            text = f"Fail Prob: {prob:.1%}"
            cv2.putText(display_frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Frame Number (Optional, for debug)
            # cv2.putText(display_frame, f"Frame: {frame_num}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            print(f"Recording Frame {frame_num} | Prob: {prob:.4f} | {status_text}")
            out.write(display_frame)

    out.release()
    print(f"\nDone! Video saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_demo()