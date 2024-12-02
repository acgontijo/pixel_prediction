import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # Progress bar library

def normalize_image(image):
    """Normalize the image to the range [0, 255] and convert to uint8."""
    max_value = np.max(image)
    if max_value > 255:
        image = (image / max_value) * 255
    return np.clip(image, 0, 255).astype(np.uint8)

def load_image(image_path):
    """Load an image as a NumPy array."""
    return np.array(Image.open(image_path))

def filter_rgb_images(source_folder, destination_folder, substring):
    """Stack RGB images and save them as a new image."""
    os.makedirs(destination_folder, exist_ok=True)  # Ensure destination folder exists

    # Iterate over all files and organize RGB bands
    bands_dict = {}  # To store paths of R, G, B images keyed by matching substring
    for filename in os.listdir(source_folder):
        if filename.endswith('16.tiff') and len(filename) >= 27:
            key = filename[-27:-8]
            if substring in key:
                if 'SR_B4' in filename:
                    band_type = 'R'
                elif 'SR_B3' in filename:
                    band_type = 'G'
                elif 'SR_B2' in filename:
                    band_type = 'B'
                else:
                    continue
                bands_dict.setdefault(key, {})[band_type] = os.path.join(source_folder, filename)

    # Process and save stacked images with progress bar
    with tqdm(total=len(bands_dict), desc="Processing Images", unit="image") as pbar:
        for key, bands in bands_dict.items():
            if all(band in bands for band in ['R', 'G', 'B']):  # Ensure all bands are present
                r_image = load_image(bands['R'])
                g_image = load_image(bands['G'])
                b_image = load_image(bands['B'])

                # Stack and normalize the image
                rgb_image = np.stack([r_image, g_image, b_image], axis=-1)
                rgb_image_normalized = normalize_image(rgb_image)

                # Create and save the output image
                output_filename = os.path.basename(bands['R']).replace('SR_B4', 'SR_RGB')
                output_path = os.path.join(destination_folder, output_filename)
                Image.fromarray(rgb_image_normalized).save(output_path)

            pbar.update(1)  # Update the progress bar for each processed image

# Set source and destination folders, and substring filter
source_folder = "../data/raw/images"
destination_folder = "../data/filtered/images_2016"
substring = "2016_08"

# Execute the processing function
filter_rgb_images(source_folder, destination_folder, substring)
print(f"Processed images saved to {destination_folder}")
