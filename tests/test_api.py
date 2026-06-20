import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["docs"] == "/docs"

def test_analyze_no_face():
    import cv2
    import numpy as np
    
    # Create a black dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()
    
    # Send post request with file
    response = client.post(
        "/analyze",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["face_detected"] is False
    assert data["features"] is None
    assert data["stress"]["stress_level"] == "No Face Detected"
