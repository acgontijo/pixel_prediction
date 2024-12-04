import os
import warnings
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("Current working directory:", os.getcwd())

# Load data
def load_tiff_data(image_dir, label_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tiff')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.tiff')])

    images, labels = [], []

    for img_file, lbl_file in zip(image_files, label_files):
        with Image.open(os.path.join(image_dir, img_file)) as img:
            images.append(np.array(img) / 255.0)
        with Image.open(os.path.join(label_dir, lbl_file)) as lbl:
            labels.append(np.array(lbl.convert('L')))

    return np.array(images), np.array(labels)

# Paths
image_dir_2016 = 'data/filtered/images_2016'
label_dir_2016 = 'data/filtered/targets_2016'
HISTORY_PATH = "./logs/training_history.json"
MODEL_PATH = "trained_model.keras"

# Check if model is already trained
if not os.path.exists(MODEL_PATH):
    # Load and split data
    images, labels = load_tiff_data(image_dir_2016, label_dir_2016)
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build U-Net model
    input_shape = X_train.shape[1:]
    model = build_unet(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

    # Save model and history
    model.save(MODEL_PATH)
    os.makedirs("./logs", exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history.history, f)
else:
    print("Model already trained. Skipping training step.")

# Load training history
if os.path.exists(HISTORY_PATH):
    print(f"Loading existing training history from {HISTORY_PATH}...")
    with open(HISTORY_PATH, "r") as f:
        history_data = json.load(f)
else:
    raise FileNotFoundError(f"Training history not found at {HISTORY_PATH}")

# Plot training history
def plot_training_logs(history_data):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_data["loss"], label="Train Loss")
    plt.plot(history_data["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_data["accuracy"], label="Train Accuracy")
    plt.plot(history_data["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_logs(history_data)

# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# def build_unet(input_shape):
#     inputs = tf.keras.layers.Input(shape=input_shape)
#     c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
#     c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
#     p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

#     c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
#     c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
#     p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

#     c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
#     c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

#     u1 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)

#     # Adjust shapes for concatenation
#     u1, c2 = match_shapes(u1, c2)

#     u1 = tf.keras.layers.concatenate([u1, c2])
#     c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
#     c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

#     u2 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)

#     # Adjust shapes for concatenation
#     u2, c1 = match_shapes(u2, c1)

#     u2 = tf.keras.layers.concatenate([u2, c1])
#     c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
#     c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

#     outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

#     return tf.keras.Model(inputs, outputs)

# def match_shapes(layer_a, layer_b):
#     """
#     Align the shapes of two layers to ensure compatibility for concatenation.

#     Parameters:
#         layer_a: Keras tensor or layer.
#         layer_b: Keras tensor or layer.

#     Returns:
#         tuple: Adjusted layers with matching dimensions.
#     """
#     def crop_to_match(layers):
#         """
#         Crop the larger layer to match the smaller one.
#         """
#         layer_a, layer_b = layers

#         # Infer shapes dynamically
#         shape_a = tf.shape(layer_a)[1:3]  # (height, width) of layer_a
#         shape_b = tf.shape(layer_b)[1:3]  # (height, width) of layer_b

#         # Calculate cropping
#         height_diff = shape_a[0] - shape_b[0]
#         width_diff = shape_a[1] - shape_b[1]

#         crop_a = [[0, height_diff // 2], [0, width_diff // 2]]
#         crop_b = [[0, -height_diff // 2], [0, -width_diff // 2]]

#         if height_diff > 0:
#             layer_a = tf.keras.layers.Cropping2D(cropping=((0, height_diff), (0, 0)))(layer_a)
#         elif height_diff < 0:
#             layer_b = tf.keras.layers.Cropping2D(cropping=((0, -height_diff), (0, 0)))(layer_b)

#         if width_diff > 0:
#             layer_a = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, width_diff)))(layer_a)
#         elif width_diff < 0:
#             layer_b = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, -width_diff)))(layer_b)

#         return layer_a, layer_b

#     # Use Keras Lambda layer to ensure compatibility with KerasTensors
#     cropped_layer_a, cropped_layer_b = tf.keras.layers.Lambda(crop_to_match)([layer_a, layer_b])

#     return cropped_layer_a, cropped_layer_b

# def dice_loss(y_true, y_pred):
#     smooth = 1.0
#     y_true_f = tf.keras.backend.flatten(y_true)
#     y_pred_f = tf.keras.backend.flatten(y_pred)
#     intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
#     return 1 - ((2. * intersection + smooth) /
#                 (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))

# def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
#     input_shape = X_train.shape[1:]
#     model = build_unet(input_shape)

#     model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

#     early_stopping = EarlyStopping(
#         monitor="val_loss", patience=5, restore_best_weights=True
#     )

#     checkpoint = ModelCheckpoint(
#         filepath="model_checkpoint.h5", monitor="val_loss", save_best_only=True, verbose=1
#     )

#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=[early_stopping, checkpoint],
#         verbose=1
#     )

#     # Save final model
#     model.save("trained_model.keras")
#     return history
