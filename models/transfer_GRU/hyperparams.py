import torch
import torchvision.transforms as transforms

# COMPUTE DEVICE
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"using device: {DEVICE}")


# DATASET
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]  # Standard ImageNet mean
IMG_STD = [0.229, 0.224, 0.225]   # Standard ImageNet std

# AUGMENTATION PARAMETERS
AUG_FLIP_PROB = 0.5         # 50% chance to flip horizontally
AUG_ROTATION_DEGREES = 10   # Rotate +/- 10 degrees (simulates crooked camera)
AUG_BRIGHTNESS = 0.3        # Increased to 0.3 (Printers often have harsh LED lighting)
AUG_CONTRAST = 0.3          # Increased to 0.3
AUG_SATURATION = 0.1        # Keep low (Filament color doesn't matter much)
AUG_HUE = 0.05              # Slight hue shift
AUG_BLUR_PROB = 0.2         # 20% chance to have a blurry lens
AUG_BLUR_KERNEL = 5         # Kernel size (Must be odd: 3, 5, 7...)

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
])

IMG_TRANSFORMS = VAL_TRANSFORMS

# SEQUENCE
SEQ_LEN = 16          # Number of frames per video clip
FPS_SAMPLE_RATE = 1   # We grab 1 frame every X seconds


# MODEL
HIDDEN_SIZE = 128     # GRU hidden units
NUM_LAYERS = 1        # GRU layers
DROPOUT_PROB = 0.4    # Dropout probability
NUM_CLASSES = 1       # Binary classification (0=Normal, 1=Detached)


# TRAINING
BATCH_SIZE = 4        # Number of videos per batch
EPOCHS = 200          # Total training passes
LEARNING_RATE = 1e-4  # Adam optimizer learning rate
TRAIN_SPLIT = 0.8     # 80% Training, 20% Validation

# PERFORMANCE OPTIMIZATIONS
NUM_WORKERS = 8             # Number of CPU processes loading data
PIN_MEMORY = True           # Faster CPU->GPU transfer
PERSISTENT_WORKERS = True   # Keeps workers alive between epochs (reduces startup lag)

# PATHS
DATASET_ROOT = "datasets/transfer_GRU/dataset"      # Folder for clean prints
MODEL_SAVE_PATH = "models/transfer_GRU/spaghettinet_v3.pth"     # Where to save model weights


# INFERENCE / LIVE MONITORING
CAMERA_INDEX = 0         # 0 = Default Webcam
ALARM_THRESHOLD = 0.75   # Trigger alarm if probability > X
STARTUP_GRACE_PERIOD = 5 # Ignore alarms for first X frames