import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def normalize_image(image):
    max_value = np.max(image)
    if max_value > 255:
        image = (image / max_value) * 255
    return np.clip(image, 0, 255).astype(np.uint8)

def load_image(image_path):
    return np.array(Image.open(image_path))

def filter_rgb_QA_images(source_folder, destination_folder, substring):
    os.makedirs(destination_folder, exist_ok=True)

    bands_dict = {}
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
                elif 'QA_PIXEL' in filename:
                    band_type = 'QA'
                else:
                    continue
                bands_dict.setdefault(key, {})[band_type] = os.path.join(source_folder, filename)

    with tqdm(total=len(bands_dict), desc="Processing Images", unit="image") as pbar:
        for key, bands in bands_dict.items():
            if all(band in bands for band in ['R', 'G', 'B', 'QA']):
                r_image = load_image(bands['R'])
                g_image = load_image(bands['G'])
                b_image = load_image(bands['B'])
                qa_image = load_image(bands['QA'])  # Load QA_PIXEL image as a NumPy array

                # Create mask for valid pixels where QA_PIXEL == 21824 (valid, non-cloud pixels)
                valid_mask = (qa_image == 21824)

                # Apply mask to the RGB images (set invalid pixels to black)
                r_image[~valid_mask] = 0
                g_image[~valid_mask] = 0
                b_image[~valid_mask] = 0

                # Stack the RGB image and normalize
                stacked_image = np.stack([r_image, g_image, b_image], axis=-1)
                stacked_image_normalized = normalize_image(stacked_image)

                # Save the result
                output_filename = os.path.basename(bands['R']).replace('SR_B4', 'SR_RGB_noClouds')
                output_path = os.path.join(destination_folder, output_filename)
                Image.fromarray(stacked_image_normalized).save(output_path)

            pbar.update(1)  # Update the progress bar for each processed image

# Set source and destination folders, and substring filter
source_folder = "/Users/marcuslotte/code/acgontijo/pixel_prediction/data/raw/images"
destination_folder = "../data/filtered/images_2016_QAfiltered"
substring = "2016_08"  # Example substring

# Execute the processing function
filter_rgb_QA_images(source_folder, destination_folder, substring)
print(f"Processed images saved to {destination_folder}")
