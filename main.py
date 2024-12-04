# # from scripts.preprocess import small_data
# # from models.unet import build_unet
# # from scripts.train import train_model
# # from scripts.visualize import visualize_predictions
# # from utils.params import DIRECTORY_Y, DIRECTORY_X, PROCESSED_DIR, MODEL_DIR, VISUALIZATION_DIR, SUBSTRING, NUM_FILES, IMAGE_SIZE
# # import tensorflow as tf
# # import numpy as np
# # import os

# # if __name__ == "__main__":
# #     print("Starting the pipeline...")

# #     # Preprocessing step
# #     print("Loading and preprocessing data...")
# #     X_train, y_train = small_data(DIRECTORY_Y, DIRECTORY_X, SUBSTRING, NUM_FILES)

# #     # Ensure correct shapes for y_train
# #     if len(y_train.shape) == 3:
# #         y_train = y_train[..., tf.newaxis]

# #     # Normalize and binarize data
# #     X_train = X_train.astype('float32') / X_train.max()
# #     y_train = y_train.astype('float32')

# #     # Save preprocessed data
# #     np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
# #     np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
# #     print("Preprocessed data saved.")

# #     # Build the model
# #     print("Building the U-Net model...")
# #     model = build_unet(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
# #     model.compile(
# #             optimizer='adam',
# #             loss='binary_crossentropy',
# #             metrics=['accuracy']
# #         )

# #     # Train the model
# #     print("Training the model...")
# #     history, X_val, y_val = train_model(model, X_train, y_train)

# #     # Save the model and validation data
# #     print("Saving the model and validation data...")
# #     model.save(os.path.join(MODEL_DIR, "trained_model.keras"))
# #     np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val)
# #     np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)

# #     # Visualize predictions
# #     print("Visualizing predictions...")
# #     val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(8)
# #     visualize_predictions(model, val_dataset, save_dir=VISUALIZATION_DIR)

# #     print("Pipeline completed successfully!")

# # import os
# # from models.preprocess_data import augment_data
# # from models.build_unet import train_model, dice_loss
# # from models.build_unet import build_unet
# # from logs.logs.training_logs import save_training_logs, plot_training_logs
# # from models.build_unet import load_tiff_data
# # from sklearn.model_selection import train_test_split
# # import numpy as np

# # # Paths
# # DATA_DIR = "./data/filtered"
# # IMAGES_2016 = os.path.join(DATA_DIR, "images_2016")
# # TARGETS_2016 = os.path.join(DATA_DIR, "targets_2016")

# # # Load data
# # print("Loading data...")
# # images, labels = load_tiff_data(IMAGES_2016, TARGETS_2016)
# # print(f"Data loaded: {images.shape} images, {labels.shape} labels")

# # # Normalize images
# # images = images / 255.0

# # # Split data into training and validation sets
# # X_train, X_val, y_train, y_val = train_test_split(
# #     images, labels, test_size=0.2, random_state=42
# # )

# # # Apply data augmentation
# # print("Applying data augmentation...")
# # augmented_data = augment_data(X_train, y_train)

# # # Train model
# # print("Training model...")
# # history = train_model(
# #     X_train, y_train, X_val, y_val, epochs=50, batch_size=32
# # )

# # # Save training logs and plot results
# # print("Saving logs and plots...")
# # save_training_logs(history)
# # plot_training_logs(history)

# # print("Training completed and saved.")

# import os
# from models.preprocess_data import augment_data
# from models.build_unet import train_model, dice_loss, build_unet, load_tiff_data
# from logs.logs.training_logs import save_training_logs, plot_training_logs
# from sklearn.model_selection import train_test_split
# import numpy as np

# # Paths
# DATA_DIR = "./data/filtered"
# IMAGES_2016 = os.path.join(DATA_DIR, "images_2016")
# TARGETS_2016 = os.path.join(DATA_DIR, "targets_2016")
# MODEL_SAVE_PATH = "./models/trained_model.keras"

# # Load data
# print("Loading data...")
# images, labels = load_tiff_data(IMAGES_2016, TARGETS_2016)
# print(f"Data loaded: {images.shape} images, {labels.shape} labels")

# # Normalize images
# images = images / 255.0

# # Split data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(
#     images, labels, test_size=0.2, random_state=42
# )

# # Apply data augmentation
# print("Applying data augmentation...")
# X_train_augmented, y_train_augmented = [], []

# for images_batch, masks_batch in augment_data(X_train, y_train):
#     X_train_augmented.append(images_batch)
#     y_train_augmented.append(masks_batch)
#     # Stop after augmenting enough batches for one epoch
#     if len(X_train_augmented) >= len(X_train) // 32:
#         break

# # Concatenate augmented data
# X_train_augmented = np.concatenate(X_train_augmented)
# y_train_augmented = np.concatenate(y_train_augmented).squeeze(-1)

# print(f"Data augmentation completed: {X_train_augmented.shape} images.")

# # Train model
# print("Building and training model...")
# input_shape = X_train_augmented.shape[1:]
# model = build_unet(input_shape)
# history = train_model(
#     X_train_augmented, y_train_augmented, X_val, y_val, epochs=50, batch_size=32
# )

# # Save model
# print("Saving trained model...")
# model.save(MODEL_SAVE_PATH)

# # Save training logs and plot results
# print("Saving logs and plots...")
# save_training_logs(history)
# plot_training_logs(history)

# print("Training completed and saved.")
