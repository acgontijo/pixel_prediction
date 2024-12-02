# CALCULATING THE DELTA

import numpy as np
from PIL import Image
import os
import re
import matplotlib.pyplot as plt

# =============================
# Helper Functions
# =============================

# Function to load an image using PIL
def load_image(image_path):
    """
    Load an image using PIL and return as a numpy array.
    """
    image = Image.open(image_path)
    return np.array(image)

# Extract latitude and longitude from filenames
def extract_lat_lon(filename):
    """
    Extract latitude and longitude from the filename.
    """
    match = re.search(r"(-?\d+\.\d+)_(-?\d+\.\d+)", filename)
    if match:
        return match.groups()
    return None

# =============================
# Preprocessing and Delta Calculation
# =============================

def calculate_delta(images_2021_dir, targets_2021_dir, output_dir, resize_to=(85, 85)):
    """
    Calculate deltas between 2021 predictions (X) and 2021 targets (Y).
    Resulting delta images will be in black and white.
    """
    # Map 2021 images based on latitude/longitude
    lat_lon_to_images = {}
    for filename in os.listdir(images_2021_dir):
        lat_lon = extract_lat_lon(filename)
        if lat_lon:
            lat_lon_to_images[lat_lon] = os.path.join(images_2021_dir, filename)

    # Process 2021 target images and match with 2021 images
    os.makedirs(output_dir, exist_ok=True)
    for target_filename in os.listdir(targets_2021_dir):
        lat_lon = extract_lat_lon(target_filename)
        if lat_lon in lat_lon_to_images:
            # File paths
            image_2021_path = lat_lon_to_images[lat_lon]
            target_2021_path = os.path.join(targets_2021_dir, target_filename)

            # Load and resize the images
            image_2021 = load_image(image_2021_path)
            target_2021 = load_image(target_2021_path)

            image_2021_resized = np.array(Image.fromarray(image_2021).resize(resize_to))
            target_2021_resized = np.array(Image.fromarray(target_2021).resize(resize_to))

            # Normalize the images
            image_2021_normalized = image_2021_resized.astype(np.float32) / 255.0
            target_2021_normalized = target_2021_resized.astype(np.float32) / 255.0

            # Convert RGB to grayscale
            if len(image_2021_normalized.shape) == 3 and image_2021_normalized.shape[2] == 3:
                image_2021_normalized = np.mean(image_2021_normalized, axis=-1)  # Average RGB channels

            # Calculate delta (absolute difference)
            delta = np.abs(image_2021_normalized - target_2021_normalized)
            delta_binary = (delta > 0.5).astype(np.uint8)  # Threshold to binary black-and-white

            # Save the delta image
            delta_image = Image.fromarray((delta_binary * 255).astype(np.uint8))  # Scale to [0, 255]
            delta_image.save(os.path.join(output_dir, f"delta_{lat_lon[0]}_{lat_lon[1]}.png"))
            print(f"Delta saved for {lat_lon} at {output_dir}")
