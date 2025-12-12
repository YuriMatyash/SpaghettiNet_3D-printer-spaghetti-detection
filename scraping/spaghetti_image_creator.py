import cv2
import os
import glob
import re
import math

# --- CONFIGURATION ---
SOURCE_VIDEO_DIR = os.path.join('scraping', 'videos')
OUTPUT_IMG_DIR = os.path.join('data', 'raw', 'spaghetti', 'images')
OUTPUT_LBL_DIR = os.path.join('data', 'raw', 'sapghetti', 'lables')

# CAPTURE SETTINGS
CAPTURE_INTERVAL = 0.5  # Take a photo every x of a second

# Helper Function
# finds the next available ID for naming
def get_next_start_id(target_dir):
    if not os.path.exists(target_dir):
        return 1
    
    max_id = 0
    pattern = re.compile(r'spaghetti_(\d+)')
    
    for filename in os.listdir(target_dir):
        match = pattern.search(filename)
        if match:
            val = int(match.group(1))
            if val > max_id:
                max_id = val
                
    return max_id + 1

def extract_frames_by_time():
    # verify paths exist, if not create them
    if not os.path.exists(SOURCE_VIDEO_DIR):
        print(f"Error: Video folder not found at: {SOURCE_VIDEO_DIR}")
        os.makedirs(SOURCE_VIDEO_DIR, exist_ok=True)
        return

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

    # acceptable video extensions, capcut mainly uses mp4, but just in case
    # collect all video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(SOURCE_VIDEO_DIR, ext)))
    
    if not video_files:
        print("No videos found in", SOURCE_VIDEO_DIR)
        return

    # get starting ID for naming
    current_id = get_next_start_id(OUTPUT_IMG_DIR)
    print(f"Starting numbering at: good_{current_id}")

    total_extracted = 0

    # start processing each video
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path)
        
        if not cap.isOpened():
            print(f"Could not open {video_name}")
            continue
            
        # Get Video Properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            print(f"Skipping {video_name} (Invalid FPS or Frame Count)")
            continue

        # calculate step size in frames, based on desired time interval
        step = int(fps * CAPTURE_INTERVAL)
        if step < 1: step = 1 # interval too small for fps
        
        duration = total_frames / fps
        expected_images = int(total_frames / step)

        print(f"Processing '{video_name}'")
        print(f"  -> Length: {duration:.1f}s | FPS: {fps:.1f}")
        print(f"  -> Interval: {CAPTURE_INTERVAL}s (Every {step} frames)")
        print(f"  -> Expected Output: ~{expected_images} images")
        
        saved_count = 0
        current_frame = 0

        while True:
            # Set the position of the next frame to capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if not ret:
                break # end of video or read error
            
            # Save image
            img_filename = f"spaghetti_{current_id}.jpg"
            img_save_path = os.path.join(OUTPUT_IMG_DIR, img_filename)
            cv2.imwrite(img_save_path, frame)
            
            current_id += 1
            saved_count += 1
            total_extracted += 1
            
            current_frame += step
            
            # if we overshoot total frames, break
            if current_frame >= total_frames:
                break

        cap.release()
        print(f"  -> Extracted {saved_count} frames.")

    print("-" * 30)
    print(f"Done! Total new images: {total_extracted}")
    print(f"Next available ID: good_{current_id}")

if __name__ == "__main__":
    extract_frames_by_time()