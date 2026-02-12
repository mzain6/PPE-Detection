# PPE Detection System

Enterprise-ready PPE (Personal Protective Equipment) Detection API using YOLOv8 for real-time helmet and safety vest detection.

## Features

- **Real-time Detection**: Person, helmet, and safety vest detection using custom-trained YOLOv8 model
- **GPU Acceleration**: NVIDIA CUDA support with automatic CPU fallback
- **Multi-stream Support**: Handle multiple RTSP streams and webcams simultaneously
- **Tracking**: Persistent person tracking across frames with unique IDs
- **RESTful API**: FastAPI-based backend with comprehensive endpoints
- **WebSocket Support**: Real-time detection streaming
- **Health Monitoring**: System health checks including GPU, memory, and camera status

## Technology Stack

* **Python 3.8+**
* **PyTorch** (with CUDA support for GPU acceleration)
* **Ultralytics YOLOv8**
* **FastAPI** (REST API framework)
* **OpenCV** (Video processing)

---

## System Requirements

### Minimum Requirements
- Python 3.8 or newer
- 4 GB RAM
- CPU: Multi-core processor

### Recommended for GPU Acceleration
- NVIDIA GPU with CUDA support (6+ GB VRAM)
- CUDA Toolkit 11.8+
- 8 GB+ RAM

---

## Getting Started

### 1. Clone or Download the Project

```bash
cd PPE-Detection
```

### 2. Create and Activate Virtual Environment

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For GPU support, ensure you have CUDA-enabled PyTorch installed:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 4. Verify Model File

Ensure the custom PPE model is present:
```bash
# Should see ppe_model.pt in the project root
ls ppe_model.pt
```

### 5. Start the API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health/

---

## Usage

### API Documentation

Access the interactive API documentation at `http://localhost:8000/docs`

### Key Endpoints

- `GET /health/` - System health status
- `GET /health/gpu` - GPU information
- `POST /cameras/register` - Register a camera stream
- `GET /cameras/` - List all cameras
- `GET /detections/` - Get latest detections
- `GET /stream/{camera_id}` - Live video stream with detections

### Example: Register a Camera

```python
import requests

response = requests.post("http://localhost:8000/cameras/register", json={
    "camera_id": "warehouse_cam_01",
    "rtsp_url": "rtsp://192.168.1.100:554/stream",
    "fps": 15
})
```

### Example: Get Detections

```python
response = requests.get("http://localhost:8000/detections/")
detections = response.json()
```

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=app --cov-report=html
```

### Run GPU Tests (requires NVIDIA GPU)

```bash
pytest tests/ -v -m gpu
```

### Run Stability Test

```bash
python tests/stability_check.py
```

---

## Validation & Benchmarking

### Validate Detection with Webcam

```bash
python scripts/validate_detections.py --source 0 --display
```

### Validate with RTSP Stream

```bash
python scripts/validate_detections.py --source "rtsp://example.com/stream" --display
```

### Performance Benchmark

```bash
python scripts/benchmark_performance.py --frames 100 --resolution 1920x1080
```

---

## Model Performance

The custom PPE model was trained on the "PPE Detection v3" dataset from Roboflow.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|:------|:----------|:-------|:------|:---------|
| **Overall** | 0.72 | 0.715 | 0.735 | 0.456 |
| **Helmet** | 0.784 | 0.824 | 0.866 | 0.584 |
| **Vest** | 0.841 | 0.897 | 0.935 | 0.705 |

---

## Configuration

Edit `config.yaml` to customize:

- Model path and parameters
- Confidence thresholds
- Tracking parameters
- Camera settings
- API configuration

---

## Troubleshooting

### GPU Not Detected

1. Verify CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Check PyTorch CUDA support:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Camera Connection Issues

- Verify RTSP URL is accessible
- Check network connectivity
- Ensure ffmpeg is installed for RTSP streams
- Review logs for detailed error messages

### Low FPS

- Enable GPU acceleration
- Reduce input resolution in config
- Decrease FPS setting for cameras
- Close unnecessary applications

---

## Phase 1 Improvements

✅ **Completed Stabilization Work**:
- Removed legacy code and consolidated codebase
- Added GPU utilities with CUDA detection and memory monitoring
- Implemented comprehensive health check endpoints
- Created test infrastructure (pytest, GPU tests, API tests)
- Added validation and benchmarking scripts
- Improved error handling and logging
- Updated configuration for custom PPE model

---

## License

This project is for demonstration and evaluation purposes.

---

## Support

For issues or questions, please check the API documentation at `/docs` or review the logs for detailed error messages.
