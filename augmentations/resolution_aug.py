import os
from PIL import Image

SIZE_WIDTH = 224
SIZE_HEIGHT = 224

# Paths
folder_path = "datasets/detection/raw"
save_path = "datasets/detection/resized"

def process_all_images(source_folder, destination_folder, target_size=(SIZE_WIDTH, SIZE_HEIGHT)):
    # 1. Create destination directory if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created directory: {destination_folder}")

    # 2. Get list of files
    try:
        files = os.listdir(source_folder)
    except FileNotFoundError:
        print(f"Error: Source folder not found at {source_folder}")
        return

    count = 0
    print(f"Starting processing of images from {source_folder}...")

    for filename in files:
        # 3. Check if file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(source_folder, filename)
            
            # --- FIX: Ensure extension is .jpg ---
            # Split filename and extension, then force .jpg
            base_name = os.path.splitext(filename)[0]
            new_filename = f"{base_name}.jpg"
            save_file_path = os.path.join(destination_folder, new_filename)

            try:
                with Image.open(file_path) as img:
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # Use "JPEG" instead of "JPG" (standard PIL format name)
                    resized_img.convert('RGB').save(save_file_path, "JPEG", quality=95)
                    
                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} images...")
                        
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    print(f"Done! Successfully processed {count} images.")
    print(f"Saved to: {destination_folder}")

if __name__ == "__main__":
    process_all_images(folder_path, save_path)