from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import os
import rasterio
import numpy as np

app = FastAPI()

# Base directory and model path
BASE_DIR = "../data/filtered"
MODEL_PATH = "trained_model.keras"
model = None  # Placeholder for the model

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

def find_image_by_coordinates(lat, lon, year, base_dir, prefix):
    """
    Finds the file corresponding to the provided latitude, longitude, and year.
    """
    if prefix == "Deforestation_":
        folder = f"targets_{year}"
    elif prefix == "Landsat8_SR_RGB_":
        folder = f"images_{year}"
    else:
        raise ValueError(f"Unknown prefix: {prefix}")

    search_pattern = f"{prefix}{lon:.2f}_{lat:.2f}_{year}_"
    folder_path = os.path.join(base_dir, folder)
    for file_name in os.listdir(folder_path):
        if file_name.startswith(search_pattern):
            return os.path.join(folder_path, file_name)

    raise FileNotFoundError(f"File not found for coordinates: {lat}, {lon} in {year} with prefix {prefix}")

def calculate_deforestation(lat, lon, base_dir, model):
    """
    Calculate the deforestation percentage based on the returned image.
    """
    # Locate the corresponding files
    image_2021_path = find_image_by_coordinates(lat, lon, 2021, base_dir, "Landsat8_SR_RGB_")
    target_2016_path = find_image_by_coordinates(lat, lon, 2016, base_dir, "Deforestation_")

    # Load the images
    with rasterio.open(image_2021_path) as img_2021:
        img_2021_array = img_2021.read().transpose(1, 2, 0) / 255.0  # Normalize

    with rasterio.open(target_2016_path) as target_2016:
        target_2016_array = target_2016.read(1)  # Single band

    # Prediction
    prediction = model.predict(np.expand_dims(img_2021_array, axis=0))
    prediction_binary = (prediction.squeeze() > 0.5).astype(int)

    # Calculate deforestation percentage
    area_2016 = np.sum(target_2016_array == 1)
    area_2021 = np.sum(prediction_binary == 1)
    deforestation_percentage = ((area_2021 - area_2016) / area_2016) * 100

    return deforestation_percentage

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
