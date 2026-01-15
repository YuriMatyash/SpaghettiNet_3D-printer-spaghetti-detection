import cv2
import matplotlib.pyplot as plt

def visualize_detection(image_path, txt_path):
    # 1. Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape # Should be 224, 224 based on your preprocessing

    # 2. Read the coordinates from the .txt file
    with open(txt_path, 'r') as f:
        line = f.readline().split()
        if not line:
            return
        
        # YOLO format: class x_center y_center width height (normalized)
        _, x_center, y_center, width, height = map(float, line)

    # 3. Convert normalized YOLO coordinates to pixel coordinates
    # Center-based to Corner-based (xmin, ymin, xmax, ymax)
    x1 = int((x_center - width / 2) * w)
    y1 = int((y_center - height / 2) * h)
    x2 = int((x_center + width / 2) * w)
    y2 = int((y_center + height / 2) * h)

    # 4. Draw the bounding box
    # Green box with thickness of 2
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 5. Add a label label
    cv2.putText(image, "Toolhead", (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 6. Plot the result
    plt.imshow(image)
    plt.title(f"Manual Annotation: {w}x{h}")
    plt.axis('off')
    plt.show()

# Use the function
visualize_detection('explaination_data/detection_data/1_toolhead.jpg' ,'explaination_data/detection_data/1_toolhead.txt')