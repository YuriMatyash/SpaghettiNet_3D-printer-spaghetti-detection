from ultralytics import YOLO

# Load your custom trained model
# It will be saved in spaghetti_project/run1/weights/best.pt
model = YOLO('spaghetti_project/run1/weights/best.pt')

# Predict
results = model('data/processed/resolution_scaling/good/images/1_good_111.jpg')

# Print the top result
# names is a dictionary like {0: 'clean', 1: 'spaghetti'}
top_class_id = results[0].probs.top1
class_name = results[0].names[top_class_id]
confidence = results[0].probs.top1conf.item()

print(f"Prediction: {class_name} ({confidence:.2%})")