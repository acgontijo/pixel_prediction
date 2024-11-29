import numpy as np
from PIL import Image
import os
import random

##################################


# INPUT INFO:

#     substring_test = "2021_05"
#     substring_test_day = "26"


##################################



# Function to normalize an image to [0, 255]
def normalize_image(image):
    image = image.astype(np.float32)  # Convert to float for scaling
    max_value = np.max(image)
    if max_value > 255:
        image = (image / max_value) * 255  # Scale to 0-255 range
    image = np.clip(image, 0, 255)  # Ensure values are in [0, 255]
    return image.astype(np.uint8)  # Convert back to uint8 for image display


def load_image(image_path):
    image = Image.open(image_path)  # Open image using PIL
    return np.array(image)  # Convert to numpy array


def small_data_test(directory_y, directory_x, substring, num_files, resize_to=(85, 85)):
    # Function to load an image and convert it to a numpy array
    # def load_image(image_path):
    #     image = Image.open(image_path)  # Open image using PIL
    #     return np.array(image)  # Convert to numpy array

    # Select filenames from y directory based on the substring
    filtered_filenames_y = [f for f in os.listdir(directory_y) if substring in f]
    random.seed(42)
    selected_filenames_y = random.sample(filtered_filenames_y, num_files)
    selected_filepaths_y = [os.path.join(directory_y, filename) for filename in selected_filenames_y]

    # Preprocess directory_x to match the filenames by RGB bands
    substring_dict_x = {}
    for filename in os.listdir(directory_x):
        if len(filename) < 27:
            continue

        matching_substring = filename[-27:-8]

            # Check if the file ends with '14' before processing it
        if filename.endswith('26.tiff'):
            # Determine the band type
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

    # Now, for each selected filename, find corresponding RGB files in directory_x
    selected_filepaths_x = []

    for selected_filename in selected_filenames_y:
        if len(selected_filename) < 27:
            continue

        matching_substring = selected_filename[-27:-8]

        if matching_substring in substring_dict_x:
            corresponding_files = substring_dict_x[matching_substring]

            # Check if all three bands are present
            if all(corresponding_files[band] is not None for band in ['R', 'G', 'B']):
                # Add the file paths to the list, maintaining the order
                selected_filepaths_x.extend([
                    os.path.join(directory_x, corresponding_files['R']),
                    os.path.join(directory_x, corresponding_files['G']),
                    os.path.join(directory_x, corresponding_files['B']),
            ])


    # Process y images (resize to 85x85)
    y_images = []
    for filepath_y in selected_filepaths_y:
        image_y = load_image(filepath_y)
        image_y_resized = np.array(Image.fromarray(image_y).resize(resize_to))  # Resize to (85, 85)
        y_images.append(image_y_resized)
    y_images_resized = np.array(y_images)

    # Process x images (RGB stacks in steps of 3)
    x_images = []
    for i in range(0, len(selected_filepaths_x), 3):
        r_image_path = selected_filepaths_x[i]
        g_image_path = selected_filepaths_x[i+1]
        b_image_path = selected_filepaths_x[i+2]

        r_image = load_image(r_image_path)
        g_image = load_image(g_image_path)
        b_image = load_image(b_image_path)

        rgb_image = np.stack([r_image, g_image, b_image], axis=-1)
        x_images.append(rgb_image)
    x_images = np.array(x_images)

    # Normalize all x_images
    x_images_normalized = np.array([normalize_image(image) for image in x_images])

    return x_images_normalized, y_images_resized
