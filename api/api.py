# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from tensorflow.keras.models import load_model
# import os
# #import rasterio
# import numpy as np
# from PIL import Image

# app = FastAPI()

# # Base directory and model path
# BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data", "filtered")
# MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"trained_model.keras")
# model = None  # Placeholder for the model

# class Coordinates(BaseModel):
#     latitude: float
#     longitude: float

# @app.on_event("startup")
# def load_model_on_startup():
#     """
#     Load the model during API startup.
#     """
#     global model
#     model = load_model(MODEL_PATH)
#     print("Model loaded successfully!")

# def find_image_by_coordinates(lat, lon, year, base_dir, prefix):
#     """
#     Finds the file corresponding to the provided latitude, longitude, and year.
#     """
#     if prefix == "Deforestation_":
#         folder = f"targets_{year}"
#     elif prefix == "Landsat8_SR_RGB_":
#         folder = f"images_{year}"
#     else:
#         raise ValueError(f"Unknown prefix: {prefix}")

#     search_pattern = f"{prefix}{lon:.2f}_{lat:.2f}_{year}_"
#     folder_path = os.path.join(base_dir, folder)
#     for file_name in os.listdir(folder_path):
#         if file_name.startswith(search_pattern):
#             return os.path.join(folder_path, file_name)

#     raise FileNotFoundError(f"File not found for coordinates: {lat}, {lon} in {year} with prefix {prefix}")

# def calculate_deforestation(lat, lon, base_dir, model):
#     """
#     Calculate the deforestation percentage based on the returned image.
#     """
#     # Locate the corresponding files
#     image_2021_path = find_image_by_coordinates(lat, lon, 2021, base_dir, "Landsat8_SR_RGB_")
#     target_2016_path = find_image_by_coordinates(lat, lon, 2016, base_dir, "Deforestation_")

#     # # Load the images
#     # with rasterio.open(image_2021_path) as img_2021:
#     #     img_2021_array = img_2021.read().transpose(1, 2, 0) / 255.0  # Normalize

#     # with rasterio.open(target_2016_path) as target_2016:
#     #     target_2016_array = target_2016.read(1)  # Single band

#     # Load the images
#     with Image.open(image_2021_path) as img_2021:
#         # Convert to a NumPy array and normalize
#         img_2021_array = np.array(img_2021) / 255.0  # Assuming it's already RGB

#     with Image.open(target_2016_path) as target_2016:
#         # Convert to a grayscale NumPy array
#         target_2016_array = np.array(target_2016.convert('L'))  # Single-band (grayscale)

#     # Prediction
#     prediction = model.predict(np.expand_dims(img_2021_array, axis=0))
#     prediction_binary = (prediction.squeeze() > 0.5).astype(int)

#     # Calculate deforestation percentage
#     area_2016 = np.sum(target_2016_array == 1)
#     area_2021 = np.sum(prediction_binary == 1)
#     deforestation_percentage = ((area_2021 - area_2016) / area_2016) * 100

#     return deforestation_percentage

# @app.post("/deforestation")
# def get_deforestation_percentage(coords: Coordinates):
#     """
#     API endpoint to calculate deforestation percentage.
#     """
#     lat, lon = coords.latitude, coords.longitude
#     try:
#         result = calculate_deforestation(lat, lon, BASE_DIR, model)
#         return {"latitude": lat, "longitude": lon, "deforestation_percentage": result}
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=str(e))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import os
import numpy as np
from PIL import Image
from math import radians, sin, cos, sqrt, atan2
import re

app = FastAPI()

# Base directory and model path
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "filtered")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.keras")
model = None  # Placeholder for the model

# Coordinate extraction patterns
PATTERNS = {
    "Deforestation": r'Deforestation_(-?\d+\.\d+)_(-?\d+\.\d+)_\d{4}_\d{2}_\d{2}\.tiff',
    "Landsat8_SR_RGB": r'Landsat8_SR_RGB_(-?\d+\.\d+)_(-?\d+\.\d+)_\d{4}_\d{2}_\d{2}\.tiff'
}

class Coordinates(BaseModel):
    latitude: float
    longitude: float

@app.on_event("startup")
def load_model_on_startup():
    """
    Load the model during API startup.
    """
    global model
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")

def extract_coordinates(filename, prefix):
    """
    Extract coordinates from a filename based on the provided prefix.
    """
    pattern = PATTERNS.get(prefix)
    if not pattern:
        raise ValueError(f"Unknown prefix: {prefix}")

    match = re.search(pattern, filename)
    if match:
        return tuple(map(float, match.groups()))
    return None

def create_image_coordinate_dict(folder_path, prefix):
    """
    Create a mapping of filenames to their coordinates in the given folder.
    """
    image_coordinates = {}
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    for filename in os.listdir(folder_path):
        if filename.endswith('.tiff'):
            coordinates = extract_coordinates(filename, prefix)
            if coordinates:
                image_coordinates[filename] = coordinates
    return image_coordinates

def haversine_distance(coord1, coord2):
    """
    Calculate the distance between two points on Earth using the Haversine formula.
    """
    R = 6371.0  # Earth's radius in kilometers
    lon1, lat1 = radians(coord1[0]), radians(coord1[1])
    lon2, lat2 = radians(coord2[0]), radians(coord2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def find_closest_image(input_coordinates, folder_path, prefix):
    """
    Find the closest image file to the given input coordinates in the folder.
    """
    image_coordinates = create_image_coordinate_dict(folder_path, prefix)
    closest_file = None
    min_distance = float('inf')

    for filename, coordinates in image_coordinates.items():
        distance = haversine_distance(input_coordinates, coordinates)
        if distance < min_distance:
            min_distance = distance
            closest_file = filename

    if not closest_file:
        raise FileNotFoundError(f"No valid images found in {folder_path} for prefix {prefix}")

    return os.path.join(folder_path, closest_file)

def calculate_deforestation(lat, lon, base_dir, model):
    """
    Calculate the deforestation percentage based on the closest image.

    Parameters:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        base_dir (str): Base directory containing the image data.
        model (tf.keras.Model): Trained model for predictions.

    Returns:
        dict: A dictionary with deforestation percentage and metadata.
    """
    # Find closest 2016 and 2021 images
    input_coordinates = (lon, lat)
    image_2021_path = find_closest_image(input_coordinates, os.path.join(base_dir, "images_2021"), "Landsat8_SR_RGB")
    target_2016_path = find_closest_image(input_coordinates, os.path.join(base_dir, "targets_2016"), "Deforestation")

    # Check if images were found
    if not image_2021_path or not target_2016_path:
        return {
            "status": "error",
            "message": f"No valid images found for coordinates: {lat}, {lon}",
            "deforestation_percentage": None
        }

    # Load images
    with Image.open(image_2021_path) as img_2021:
        img_2021_array = np.array(img_2021) / 255.0  # Normalize

    with Image.open(target_2016_path) as target_2016:
        target_2016_array = np.array(target_2016.convert('L'))  # Grayscale

    # Prediction
    prediction = model.predict(np.expand_dims(img_2021_array, axis=0))
    prediction_binary = (prediction.squeeze() > 0.5).astype(int)

    # Calculate areas
    area_2016 = np.sum(target_2016_array == 1)
    area_2021 = np.sum(prediction_binary == 1)

    if area_2016 == 0:  # Handle zero area in 2016
        return {
            "status": "warning",
            "message": "2016 area is zero, deforestation percentage cannot be calculated.",
            "deforestation_percentage": 0  # Or a fallback value
        }

    # Calculate deforestation percentage
    deforestation_percentage = (-(area_2021 - area_2016) / area_2016) * 100

    return {
        "status": "success",
        "message": "Calculation completed successfully.",
        "deforestation_percentage": deforestation_percentage
    }

@app.post("/deforestation")
def get_deforestation_percentage(coords: Coordinates):
    """
    API endpoint to calculate deforestation percentage.
    """
    lat, lon = coords.latitude, coords.longitude
    try:
        result = calculate_deforestation(lat, lon, BASE_DIR, model)
        return {"latitude": lat, "longitude": lon, "deforestation_percentage": result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

