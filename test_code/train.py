import os
from ultralytics import YOLO

def train_classifier():
    # Get the absolute path to the folder where this script lives
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {current_dir}")  # Debug print
    
    # Force the full path to the weights file
    weights_path = os.path.join(current_dir, 'yolov12n-cls.pt')

    print(f"Looking for weights at: {weights_path}") # Debug print

    # 1. Load model with explicit path
    model = YOLO(weights_path) 

    # 2. Train
    results = model.train(
        data='datasets/spaghetti_classifier', 
        epochs=20,
        imgsz=224,
        batch=16,
        project='spaghetti_project',
        name='run1'
    )

if __name__ == '__main__':
    train_classifier()