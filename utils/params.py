import os

# Training parameters
EPOCHS = 50
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Define root directory (base of the project)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Input directories
DIRECTORY_Y = os.path.join(ROOT_DIR, "data", "raw", "targets")
DIRECTORY_X = os.path.join(ROOT_DIR, "data", "raw", "images")

# Output directories
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
VISUALIZATION_DIR = os.path.join(ROOT_DIR, "logs", "visualizations")

# Data preprocessing parameters
SUBSTRING = "2016_08"
NUM_FILES = 1000
IMAGE_SIZE = (85, 85)

# Ensure all output directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
