import os
import shutil
import numpy as np
from PIL import Image

# Function to load and resize image
def resize_image(image_path, resize_to=(85, 85)):
    image = Image.open(image_path)  # Open image
    image_resized = image.resize(resize_to)  # Resize image
    return image_resized

# Define paths and filter criteria
source_folder = "../data/raw/targets"
destination_folder = "../data/filtered/targets_2021"
filter_string = "2021_05_01"
file_extension = ".tiff"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Filter files based on criteria
filtered_files = [
    file for file in os.listdir(source_folder)
    if filter_string in file and file.endswith(file_extension)
]

# Process and copy resized files to the destination folder
for file in filtered_files:
    src_path = os.path.join(source_folder, file)
    dst_path = os.path.join(destination_folder, file)

    # Resize the image
    resized_image = resize_image(src_path)

    # Save resized image to the destination folder
    resized_image.save(dst_path)

# Check write permissions for the destination folder
if not os.access(destination_folder, os.W_OK):
    print(f"No write permission for the folder: {destination_folder}")
else:
    print(f"Write permissions are OK for the folder: {destination_folder}")
