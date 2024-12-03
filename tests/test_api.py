import requests

# Base URL of the FastAPI application
BASE_URL = "http://127.0.0.1:8000"

def test_deforestation_endpoint():
    """
    Tests the /deforestation endpoint by providing sample coordinates.
    """
    url = f"{BASE_URL}/deforestation"
    data = {
        "latitude": -4.39,
        "longitude": -55.20
    }

    response = requests.post(url, json=data)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"

    result = response.json()
    assert "latitude" in result, "Latitude missing in response"
    assert "longitude" in result, "Longitude missing in response"
    assert "deforestation_percentage" in result, "Deforestation percentage missing in response"

    print("API Test Passed: ", result)

if __name__ == "__main__":
    test_deforestation_endpoint()
