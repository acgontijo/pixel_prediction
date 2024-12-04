import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_data(X_train, y_train):
    """
    Apply data augmentation to images and their corresponding masks.

    Parameters:
        X_train (np.ndarray): Training images of shape (N, H, W, C).
        y_train (np.ndarray): Training labels of shape (N, H, W).

    Returns:
        generator: A generator yielding augmented images and masks.
    """
    image_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    mask_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Expand the labels to have 4 dimensions
    y_train_expanded = np.expand_dims(y_train, axis=-1)  # (N, H, W) -> (N, H, W, 1)

    # Generate augmented data
    image_generator = image_datagen.flow(X_train, batch_size=32, seed=42)
    mask_generator = mask_datagen.flow(y_train_expanded, batch_size=32, seed=42)

    return zip(image_generator, mask_generator)
