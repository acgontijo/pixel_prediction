from fastapi import FastAPI, Query
import random  # Added for generating random percentages

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Pixel Prediction API"}

@app.get("/deforestation/")
def predict(lat: float = Query(..., description="Latitude"), lon: float = Query(..., description="Longitude")):
    """
    Handles deforestation analysis based on coordinates by querying an ML model.
    """
    # Log the received coordinates (useful for debugging)
    print(f"Received coordinates: Latitude={lat}, Longitude={lon}")

    # Simulate interaction with the ML model (placeholder)
    # Remove this block as soon as the real model is connected
    deforestation_percentage = round(random.uniform(10.0, 30.0), 2)

    # Return only the deforestation percentage
    return {"deforestation_percentage": deforestation_percentage}
