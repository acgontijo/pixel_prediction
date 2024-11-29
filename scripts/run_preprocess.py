import numpy as np
import os
from scripts.preprocess import small_data
from utils.params import DIRECTORY_Y, DIRECTORY_X, SUBSTRING, NUM_FILES, PROCESSED_DIR

print(f"DIRECTORY_Y resolved to: {DIRECTORY_Y}")
print(f"DIRECTORY_X resolved to: {DIRECTORY_X}")

# Validate directories
if not os.path.exists(DIRECTORY_Y):
    raise FileNotFoundError(f"Directory Y not found: {DIRECTORY_Y}")
if not os.path.exists(DIRECTORY_X):
    raise FileNotFoundError(f"Directory X not found: {DIRECTORY_X}")

# Run preprocessing
print("Running data preprocessing...")
X_train, y_train = small_data(DIRECTORY_Y, DIRECTORY_X, SUBSTRING, NUM_FILES)

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Save processed data
np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)

print("Data preprocessing completed and saved.")
