import PIL.Image

# --- FIX START: Patch Pillow 10+ Compatibility ---
# Pillow 10 removed ANTIALIAS, but MoviePy still uses it. 
# We manually re-add it as an alias to LANCZOS.
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
# --- FIX END ---

from moviepy.editor import VideoFileClip

def convert_to_gif(input_file, output_file):
    print(f"Converting {input_file} to {output_file}...")
    
    # Load the clip
    clip = VideoFileClip(input_file)
    
    # Resize to width=600 to keep file size small for GitHub
    # This is where the error previously occurred
    clip = clip.resize(width=600) 
    
    # Write GIF (1 FPS matches your demo generation)
    clip.write_gif(output_file, fps=1) 
    print("Conversion complete!")

if __name__ == "__main__":
    convert_to_gif("explaination_data/transfer_GRU_data/demo_inference.mp4", "explaination_data/transfer_GRU_data/demo.gif")