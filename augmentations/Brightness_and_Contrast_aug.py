from PIL import Image, ImageEnhance

# Load an image
img_path = "temp/spaghetti_32.jpg"
original_img = Image.open(img_path)

# Creating a darker version of the image
enhancer = ImageEnhance.Brightness(original_img)
dark_img = enhancer.enhance(0.5) 
dark_img.show(title="Darker") # Opening the window to show the darker image

# Creating a verison with higher contrast
enhancer = ImageEnhance.Contrast(original_img)
high_contrast_img = enhancer.enhance(2.0)
high_contrast_img.show(title="High Contrast")

# Creating a brighter image with higher contrast
enhancer_b = ImageEnhance.Brightness(original_img)
bright_img = enhancer_b.enhance(1.7)

enhancer_c = ImageEnhance.Contrast(bright_img) # Using the brightened image
final_washed_out_img = enhancer_c.enhance(0.4) # Lowering contrast to create washed-out effect
final_washed_out_img.show(title="Bright and Washed Out")