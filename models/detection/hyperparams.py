# HYPERPARAMETERS FOR DETECTION MODEL
import os

FINAL_WEIGHTS_NAME = "Final_Detection_Results"

# GENERAL PATHS
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../"))
def get_path(*args):
    return os.path.join(PROJECT_ROOT, *args)

PATH_BASE = os.path.join(PROJECT_ROOT, "models", "detection")

# PATHS
DATA_YAML = get_path("datasets", "detection", "dataset", "dataset.yaml")
PATH_WEIGHTS_YOLO = get_path("models", "detection", "yolo26n.pt")
PATH_WEIGHTS_FINAL = get_path("models", "detection", "weights", f"{FINAL_WEIGHTS_NAME}.pt")


# TRAINING
IMG_SIZE = 224          # Matching your resized images
EPOCHS = 200            # High epochs because dataset is small
BATCH_SIZE = 16         # Adjust based on your VRAM (16 or 32 is usually safe for 224x224)

AUG_MOSAIC=1.0
AUG_MIXUP=0.1
AUG_ROTATION_DEGREES=15.0
AUG_TRANSLATE=0.1
AUG_SCALE=0.5
AUG_FLIP_LR=0.5

OPT_PAITENCE=20

# EVALUATION
# EVALUATION
CONF_THRESHOLD = 0.25   # Only show boxes with >25% confidence.