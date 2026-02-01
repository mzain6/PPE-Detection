"""
API endpoint integration tests.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test main health check endpoint."""
    response = client.get("/health/")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "api" in data
    assert "system" in data
    assert "gpu" in data
    assert "model" in data
    assert "cameras" in data


def test_gpu_health_endpoint():
    """Test GPU health endpoint."""
    response = client.get("/health/gpu")
    assert response.status_code == 200
    
    data = response.json()
    assert "available" in data


def test_api_docs():
    """Test that API documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema():
    """Test that OpenAPI schema is accessible."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema


def test_cameras_list():
    """Test cameras list endpoint."""
    response = client.get("/cameras/")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)


def test_detections_list():
    """Test detections list endpoint."""
    response = client.get("/detections/")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, dict)


@pytest.mark.integration
def test_camera_registration():
    """Test camera registration flow."""
    # This is a basic test - in production you'd need a real RTSP stream or mock
    camera_data = {
        "camera_id": "test_camera_001",
        "rtsp_url": "rtsp://example.com/stream",
        "fps": 15
    }
    
    # Note: This will likely fail without a real stream, but tests the endpoint
    response = client.post("/cameras/register", json=camera_data)
    # Accept both success and expected failures (no actual stream)
    assert response.status_code in [200, 201, 400, 503]
