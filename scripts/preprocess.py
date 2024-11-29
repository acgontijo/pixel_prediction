import numpy as np
from PIL import Image
import os
import random

def normalize_image(image):
    image = image.astype(np.float32)
    max_value = np.max(image)
    if max_value > 255:
        image = (image / max_value) * 255
    return np.clip(image, 0, 255).astype(np.uint8)

def load_image(image_path):
    return np.array(Image.open(image_path))

def small_data(directory_y, directory_x, substring, num_files, resize_to=(85, 85)):
    filtered_filenames_y = [f for f in os.listdir(directory_y) if substring in f]
    random.seed(42)
    selected_filenames_y = random.sample(filtered_filenames_y, num_files)
    selected_filepaths_y = [os.path.join(directory_y, filename) for filename in selected_filenames_y]

    substring_dict_x = {}
    for filename in os.listdir(directory_x):
        matching_substring = filename[-27:-8]
        if 'SR_B4' in filename:
            band = 'R'
        elif 'SR_B3' in filename:
            band = 'G'
        elif 'SR_B2' in filename:
            band = 'B'
        else:
            continue
        if matching_substring not in substring_dict_x:
            substring_dict_x[matching_substring] = {'R': None, 'G': None, 'B': None}
        substring_dict_x[matching_substring][band] = filename

    selected_filepaths_x = []
    for selected_filename in selected_filenames_y:
        matching_substring = selected_filename[-27:-8]
        if matching_substring in substring_dict_x:
            corresponding_files = substring_dict_x[matching_substring]
            if all(corresponding_files[band] is not None for band in ['R', 'G', 'B']):
                selected_filepaths_x.extend(
                    [os.path.join(directory_x, corresponding_files[band]) for band in ['R', 'G', 'B']]
                )

    y_images = [np.array(Image.open(f).resize(resize_to)) for f in selected_filepaths_y]
    x_images = []
    for i in range(0, len(selected_filepaths_x), 3):
        r, g, b = [load_image(selected_filepaths_x[i + j]) for j in range(3)]
        x_images.append(np.stack([r, g, b], axis=-1))
    return np.array(x_images), np.array(y_images)
