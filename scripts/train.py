from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np

def train_model(model, X_train, y_train):
    """
    Train the U-Net model.
    """
    # Ensure the model is compiled
    if not model.compiled_metrics:  # Check if the model is compiled
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    # Split the dataset into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Define callbacks
    early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

    # Train the model
    history = model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Return the training history and validation data
    return history, X_val, y_val
