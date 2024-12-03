from models.build_unet import build_unet
import numpy as np

def test_build_unet():
    """
    Tests if the U-Net model is built successfully with the correct input shape.
    """
    input_shape = (85, 85, 3)  # Example input shape
    model = build_unet(input_shape)
    assert model.input_shape == (None, *input_shape), "Input shape mismatch"
    assert len(model.layers) > 0, "No layers found in the model"
    print("Model build test passed!")

def test_training_data_loading():
    """
    Tests if the dataset loads correctly for training.
    """
    from training.train_model import load_tiff_data

    image_dir = "../data/filtered/images_2016"
    label_dir = "../data/filtered/targets_2016"
    images, labels = load_tiff_data(image_dir, label_dir)

    assert len(images) > 0, "No images loaded"
    assert len(labels) > 0, "No labels loaded"
    assert images.shape[1:] == (85, 85, 3), "Image shape mismatch"
    assert labels.shape[1:] == (85, 85), "Label shape mismatch"
    print("Data loading test passed!")

if __name__ == "__main__":
    test_build_unet()
    test_training_data_loading()
