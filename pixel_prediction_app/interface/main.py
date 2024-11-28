# =============================
# 1. Import Libraries
# =============================
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import glob
import matplotlib.pyplot as plt

# =============================
# 2. Define Preprocessing Functions
# =============================

def load_image(image_path, size=(256, 256)):
    """
    Load and preprocess Landsat 8 image (assume 3-channel RGB).
    Resizes to the required size and normalizes to [0, 1].
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # Adjust for your Landsat 8 format
    image = tf.image.resize(image, size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image

def load_mask(mask_path, size=(256, 256)):
    """
    Load and preprocess binary segmentation mask.
    Resizes to the required size and ensures binary values (0 or 1).
    """
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)  # Single-channel mask
    mask = tf.image.resize(mask, size)
    mask = tf.cast(mask > 128, tf.float32)  # Binarize mask (thresholding)
    return mask

def load_data(image_path, mask_path, size=(256, 256)):
    """
    Load image and mask together for TensorFlow dataset.
    """
    image = load_image(image_path, size)
    mask = load_mask(mask_path, size)
    return image, mask

# =============================
# 3. Create Dataset Pipeline
# =============================

def create_dataset(image_dir, mask_dir, batch_size=8, size=(256, 256)):
    """
    Create a TensorFlow dataset from directories of images and masks.
    """
    image_paths = sorted(glob.glob(f"{image_dir}/*.png"))  # Adjust for your file format
    mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))    # Adjust for your file format

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(lambda img, msk: load_data(img, msk, size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# =============================
# 4. Define U-Net Model
# =============================

def build_unet(input_shape=(256, 256, 3)):
    """
    Build U-Net model for binary segmentation.
    """
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Dropout(0.2)(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Dropout(0.2)(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    b1 = layers.Dropout(0.3)(b1)
    b1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(b1)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(b1)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    u2 = layers.UpSampling2D((2, 2))(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    # Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c4)

    return Model(inputs, outputs)

# =============================
# 5. Train the Model
# =============================

# Directories for images and masks
image_dir = '/path/to/images'
mask_dir = '/path/to/masks'

# Create datasets
train_dataset = create_dataset(f"{image_dir}/train", f"{mask_dir}/train")
val_dataset = create_dataset(f"{image_dir}/val", f"{mask_dir}/val")

# Build and compile the model
model = build_unet()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', MeanIoU(num_classes=2)])

# Set callbacks
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[early_stopping, model_checkpoint]
)

# =============================
# 6. Visualize Results
# =============================

def visualize_predictions(model, dataset, num_samples=3):
    """
    Visualize predictions from the trained model.
    """
    for images, masks in dataset.take(num_samples):
        preds = model.predict(images)
        preds = tf.round(preds)

        for i in range(len(images)):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(images[i].numpy())
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("True Mask")
            plt.imshow(masks[i].numpy().squeeze(), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.imshow(preds[i].numpy().squeeze(), cmap="gray")
            plt.axis("off")
            plt.show()

# Visualize predictions
visualize_predictions(model, val_dataset)
