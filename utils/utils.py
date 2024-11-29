import numpy as np

def normalize_image(image):
    """
    Normalize image to [0, 255] range and return as uint8.
    """
    image = image.astype(np.float32)
    max_value = np.max(image)
    if max_value > 255:
        image = (image / max_value) * 255
    return np.clip(image, 0, 255).astype(np.uint8)

def log_message(message):
    """
    Print a formatted log message.
    """
    print(f"[LOG]: {message}")
