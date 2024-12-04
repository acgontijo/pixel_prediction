import os
import numpy as np
from models.build_unet import build_unet
from models.build_unet import load_tiff_data


def test_build_unet():
    """
    Tests if the U-Net model is built successfully with the correct input shape.
    """
    input_shape = (85, 85, 3)  # Example input shape
    model = build_unet(input_shape)

    # Check input shape
    assert model.input_shape == (None, *input_shape), (
        f"Input shape mismatch: expected (None, {input_shape}), "
        f"got {model.input_shape}"
    )

    # Check if layers are present
    assert len(model.layers) > 0, "No layers found in the model"
    print("Model build test passed!")


def test_training_data_loading():
    """
    Tests if the dataset loads correctly for training.
    """
    # Set paths dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "../data/filtered/images_2016")
    label_dir = os.path.join(script_dir, "../data/filtered/targets_2016")

    # Load data
    images, labels = load_tiff_data(image_dir, label_dir)

    # Validate data loading
    assert len(images) > 0, "No images loaded"
    assert len(labels) > 0, "No labels loaded"

    # Validate shapes
    assert images.shape[1:] == (85, 85, 3), (
        f"Image shape mismatch: expected (85, 85, 3), got {images.shape[1:]}"
    )
    assert labels.shape[1:] == (85, 85), (
        f"Label shape mismatch: expected (85, 85), got {labels.shape[1:]}"
    )
    print("Data loading test passed!")


def main():
    """
    Run all tests.
    """
    print("Running tests...\n")
    test_build_unet()
    test_training_data_loading()
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
