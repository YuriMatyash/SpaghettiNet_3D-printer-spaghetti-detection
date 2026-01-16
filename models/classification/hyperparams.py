# HYPERPARAMETERS FOR CLASSIFICATION MODEL
import os

# GENERAL
FINAL_WEIGHTS_NAME = "Final_Classification_Results"


# GENERAL PATHS
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../"))
def get_path(*args):
    return os.path.join(PROJECT_ROOT, *args)

PATH_BASE = os.path.join(PROJECT_ROOT, "models", "classification")

# DATASET PATHS
PATH_CLEAN_RAW = get_path("datasets", "classification", "raw", "clean")
PATH_SPAGHETTI_RAW = get_path("datasets", "classification", "raw", "spaghetti")

PATH_CLEAN_RESIZED = get_path("datasets", "classification", "resized", "clean")
PATH_SPAGHETTI_RESIZED = get_path("datasets", "classification", "resized", "spaghetti")

PATH_FINAL_DATASET = get_path("datasets", "classification", "final")

PATH_CLEAN_TRAIN = get_path("datasets", "classification", "final", "train", "clean")
PATH_SPAGHETTI_TRAIN = get_path("datasets", "classification", "final", "train", "spaghetti")

PATH_CLEAN_VAL = get_path("datasets", "classification", "final", "val", "clean")
PATH_SPAGHETTI_VAL = get_path("datasets", "classification", "final", "val", "spaghetti")

# WEIGHTS PATHS
PATH_WEIGHTS_YOLO = get_path("models", "classification", "yolo26n-cls.pt")
PATH_WEIGHTS_FINAL = get_path("models", "classification", "weights", f"{FINAL_WEIGHTS_NAME}.pt")

# DATA
SPLIT_RATIO = 0.8       # 80% train, 20% val split ratio

# TRAIN
BATCH_SIZE = 16         # Number of images per batch
EPOCHS = 20             # Total training passes
IMG_SIZE = 224          # Image size for training