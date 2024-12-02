import numpy as np
from PIL import Image

def preprocess_single_image(image_paths, resize_to=(85, 85)):
    """
    Preprocess a set of three images (R, G, B bands) to create a single RGB image
    for U-Net model prediction.

    Parameters:
    - image_paths (list): List of file paths for R, G, and B bands (in order).
    - resize_to (tuple): Target size for resizing (width, height).

    Returns:
    - np.array: Preprocessed RGB image ready for U-Net model prediction, with shape (1, height, width, 3).
    """
    if len(image_paths) != 3:
        raise ValueError("Expected 3 image paths for R, G, and B bands, got: {}".format(len(image_paths)))

    # Step 1: Load and resize each band (R, G, B)
    r_image = np.array(Image.open(image_paths[0]).resize(resize_to, Image.Resampling.LANCZOS))
    g_image = np.array(Image.open(image_paths[1]).resize(resize_to, Image.Resampling.LANCZOS))
    b_image = np.array(Image.open(image_paths[2]).resize(resize_to, Image.Resampling.LANCZOS))

    # Step 2: Stack the bands to create an RGB image
    rgb_image = np.stack([r_image, g_image, b_image], axis=-1)  # Shape: (height, width, 3)

    # Step 3: Normalize the image to [0, 1]
    rgb_image = rgb_image  # Ensure all values are between 0 and 1

    # Step 4: Add batch dimension for model input
    rgb_image = np.expand_dims(rgb_image, axis=0)  # Shape: (1, height, width, 3)

    return rgb_image
