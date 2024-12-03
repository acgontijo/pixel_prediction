import os
import warnings
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import rasterio
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

# Load data
def load_tiff_data(image_dir, label_dir):
    """
    Load imagens and labels from data filtered directories.
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tiff')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.tiff')])

    images, labels = [], []
    for img_file, lbl_file in zip(image_files, label_files):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=rasterio.errors.NotGeoreferencedWarning)
            with rasterio.open(os.path.join(image_dir, img_file)) as img:
                images.append(img.read().transpose(1, 2, 0))  # HxWxC
            with rasterio.open(os.path.join(label_dir, lbl_file)) as lbl:
                labels.append(lbl.read(1))  # Single band

    return np.array(images), np.array(labels)

# Data directories
image_dir_2016 = '../data/filtered/images_2016'
label_dir_2016 = '../data/filtered/targets_2016'

# Load data
images_2016, labels_2016 = load_tiff_data(image_dir_2016, label_dir_2016)

# Normalization
images_2016 = images_2016 / 255.0

# Train test split
X_train, X_val, y_train, y_val = train_test_split(images_2016, labels_2016, test_size=0.2, random_state=42)

# Model building
def build_unet(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    u1 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = tf.keras.layers.concatenate([u1, c2])
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = tf.keras.layers.concatenate([u2, c1])

    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    return tf.keras.Model(inputs, outputs)

# Compile and train model
input_shape = X_train.shape[1:]
model = build_unet(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save model
model.save("trained_model.h5")
