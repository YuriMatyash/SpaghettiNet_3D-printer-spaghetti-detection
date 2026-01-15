import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np

from model import SpaghettiNet
import hyperparams as hp

transform = hp.IMG_TRANSFORMS

def main():
    device = hp.DEVICE 
    
    model = SpaghettiNet()
    try:
        model.load_state_dict(torch.load(hp.MODEL_SAVE_PATH, map_location=device))
        print(f"Weights loaded from {hp.MODEL_SAVE_PATH}")
    except FileNotFoundError:
        print(f"Error: Could not find {hp.MODEL_SAVE_PATH}. Did you run train.py?")
        return

    model.to(device)
    model.eval() # Set to evaluation mode

    # 2. Setup Camera
    cap = cv2.VideoCapture(hp.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {hp.CAMERA_INDEX}.")
        return

    print("Monitoring started. Press 'q' to quit.")

    # State variables
    last_process_time = 0
    frame_counter = 0
    current_prob = 0.0
    
    # Calculate wait time based on sample rate (e.g., if rate is 1fps, wait 1.0s)
    wait_interval = 1.0 / hp.FPS_SAMPLE_RATE

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        current_time = time.time()
        
        # --- THE SAMPLING LOGIC ---
        # We only feed the model once every X seconds (defined in hyperparams)
        if current_time - last_process_time >= wait_interval:
            
            # 1. Preprocess
            # OpenCV (BGR) -> PIL (RGB) -> Transform -> Tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            # Add Batch Dimension: (1, 3, 224, 224)
            input_tensor = transform(pil_img).unsqueeze(0).to(device) 
            
            # 2. Inference
            with torch.no_grad():
                # This function handles the hidden state internally
                current_prob = model.predict_live_frame(input_tensor).item()
            
            # 3. Logic & Reset
            frame_counter += 1
            last_process_time = current_time
            
            # Reset memory when sequence length is reached (e.g., 16 frames)
            if frame_counter >= hp.SEQ_LEN:
                print(f"[{time.strftime('%H:%M:%S')}] Window Complete. Resetting GRU Memory.")
                model.reset_memory()
                frame_counter = 0
            
            # Console Log
            status = "DETACHED" if current_prob > hp.ALARM_THRESHOLD else "Normal"
            print(f"Frame {frame_counter}/{hp.SEQ_LEN} | Prob: {current_prob:.4f} | Status: {status}")

        # --- VISUALIZATION (Runs at full 30fps) ---
        # Draw status on the video feed
        display_frame = frame.copy()
        
        # Color: Green if normal, Red if detached
        color = (0, 255, 0) if current_prob < hp.ALARM_THRESHOLD else (0, 0, 255)
        
        # Text 1: Probability
        cv2.putText(display_frame, f"Fail Prob: {current_prob:.1%}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Text 2: Frame Counter
        cv2.putText(display_frame, f"Seq: {frame_counter}/{hp.SEQ_LEN}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Big Alarm Text
        if current_prob > hp.ALARM_THRESHOLD:
            # Optional: Add a grace period check here if needed (e.g. if frame_counter > 5)
            cv2.putText(display_frame, "ALARM: DETACHMENT!", (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow('SpaghettiNet Monitor', display_frame)

        # Quit check
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()