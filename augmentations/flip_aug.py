import os
from PIL import Image
from pathlib import Path

def flip_images_in_directory(root_dir_path):
   
    valid_images = [".jpg", ".jpeg", ".png", ".bmp"]
    count = 0

    for subdir, dirs, files in os.walk(root_dir_path):
        for file in files:
            filepath = Path(subdir) / file
            
            if filepath.suffix.lower() in valid_images and "_flipped" not in file:
                try:

                    with Image.open(filepath) as img:
                   
                        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        new_filename = f"{filepath.stem}_flipped{filepath.suffix}"
                        new_filepath = Path(subdir) / new_filename
                        flipped_img.save(new_filepath)
                        count += 1
                        print(f"Created: {new_filename}")
                        
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    print(f"\nDone! Created {count} new flipped images.")


dataset_path = "temp" 
flip_images_in_directory(dataset_path)