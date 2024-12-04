from fastapi.testclient import TestClient
from api.api import app, load_model_on_startup
from unittest import mock
import numpy as np
from api.api import calculate_deforestation

client = TestClient(app)

# Força o carregamento do modelo antes dos testes
load_model_on_startup()

def test_deforestation_endpoint():
    """
    Test the /deforestation endpoint with valid coordinates.
    """
    response = client.post("/deforestation", json={"latitude": -4.39, "longitude": -55.20})
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert "deforestation_percentage" in data, "Response does not contain deforestation percentage"
    print("Deforestation percentage:", data["deforestation_percentage"])

def test_invalid_coordinates():
    """
    Test the /deforestation endpoint with invalid coordinates.
    """
    response = client.post("/deforestation", json={"latitude": 0.0, "longitude": 0.0})
    assert response.status_code == 404, "Expected a 404 for invalid coordinates"
    assert "No valid images found" in response.json()["detail"], "Unexpected error message for invalid coordinates"

def test_zero_area_2016():
    """
    Test the calculate_deforestation function with zero area for 2016.
    """
    from api.api import calculate_deforestation
    from unittest import mock
    import numpy as np

    lat, lon = 0.0, 0.0  # Example coordinates
    base_dir = "api/data/filtered"  # Ajuste conforme necessário
    model = mock.Mock()  # Mocked model for testing
    model.predict.return_value = np.zeros((1, 85, 85, 1))  # Mocked prediction

    deforestation_percentage = calculate_deforestation(lat, lon, base_dir, model)
    assert deforestation_percentage == 0, "Deforestation percentage should be 0 when area_2016 is 0"
