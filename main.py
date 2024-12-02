from scripts.preprocess import small_data
from models.unet import build_unet
from scripts.train import train_model
from scripts.visualize import visualize_predictions
from utils.params import DIRECTORY_Y, DIRECTORY_X, PROCESSED_DIR, MODEL_DIR, VISUALIZATION_DIR, SUBSTRING, NUM_FILES, IMAGE_SIZE
import tensorflow as tf
import numpy as np
import os

if __name__ == "__main__":
    print("Starting the pipeline...")

    # Preprocessing step
    print("Loading and preprocessing data...")
    X_train, y_train = small_data(DIRECTORY_Y, DIRECTORY_X, SUBSTRING, NUM_FILES)

    # Ensure correct shapes for y_train
    if len(y_train.shape) == 3:
        y_train = y_train[..., tf.newaxis]

    # Normalize and binarize data
    X_train = X_train.astype('float32') / X_train.max()
    y_train = y_train.astype('float32')

    # Save preprocessed data
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    print("Preprocessed data saved.")

    # Build the model
    print("Building the U-Net model...")
    model = build_unet(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    # Train the model
    print("Training the model...")
    history, X_val, y_val = train_model(model, X_train, y_train)

    # Save the model and validation data
    print("Saving the model and validation data...")
    model.save(os.path.join(MODEL_DIR, "trained_model.keras"))
    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)

    # Visualize predictions
    print("Visualizing predictions...")
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(8)
    visualize_predictions(model, val_dataset, save_dir=VISUALIZATION_DIR)

    print("Pipeline completed successfully!")

