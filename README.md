# PPE Detection System - SafeSite AI

A real-time Personal Protective Equipment (PPE) detection system using YOLOv8, capable of detecting safety vests and helmets, performing face recognition for authorized personnel, and tracking violations across multiple cameras.

## 🚀 Features
- **Multi-Camera Support**: Simultaneous processing of webcam and RTSP streams.
- **PPE Detection**: Real-time detection of Hardhats and Safety Vests.
- **Face Recognition**: Identifies authorized personnel at entrance cameras.
- **Violation Logic**: Tracks "No Helmet" and "No Vest" violations with persistence.
- **Alert System**: Sends violation alerts to a central server and dashboard.
- **Evidence Recording**: Automatically records video clips of violations.

---

## 🛠️ System Setup (New Machine)

Follow these steps to set up the system on a fresh machine.

### 1. Prerequisites
- Python 3.8+
- NVIDIA GPU (Recommended) with CUDA installed.
- **Operating System**: Windows / Linux

### 2. Clone & Install Dependencies
Open a terminal in the project folder (e.g., `C:\Users\Username\Downloads\PPE`) and run:

```bash
# Install PyTorch with CUDA (adjust command for your CUDA version if needed)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install ultralytics opencv-python numpy requests fastapi uvicorn python-multipart jinja2
```

### 3. Verify Model Files
Ensure the following model files are present in the project root:
- `helmet.pt`
- `vest.pt`
- `yolov8n.pt`
- `best.pt` (for server-side simple detection)

---

## 🖥️ Running the Server (Backend & Dashboard)

The server handles incoming alerts from the camera script and hosts the dashboard.

**1. Start the Server:**
Run this command from the project root:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**2. Access Dashboards:**
- **Alerts Dashboard**: [http://localhost:8000/alerts](http://localhost:8000/alerts) (Real-time violation log)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

### API Reference (Backend Endpoints)
You can fetch data from these local endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `http://localhost:8000/api/ppe-alerts` | GET | List recent violation alerts (JSON) |
| `http://localhost:8000/api/ppe-alerts/stats` | GET | Violation statistics counts (JSON) |
| `http://localhost:8000/health` | GET | System health status |
| `http://localhost:8000/docs` | - | interactive Swagger UI to test all endpoints |


---

## 📹 Running the Detection System (Cameras)

The main script (`run_cameras_with_face_tracking.py`) connects to cameras, runs AI inference, and sends alerts to the server.

### 1. Configure Cameras
Open `run_cameras_with_face_tracking.py` and submit your camera details in the `CAM_CONFIG` section:

```python
CAM_CONFIG = [
    # Cam 1: Webcam (Index 0) - Entrance
    {"id": "Cam 1", "url": 0, "is_entrance": True},
    
    # Cam 2: RTSP Camera
    # Format: rtsp://username:password@IP:Port/path
    # Note: Encode special chars in password (@ -> %40)
    {"id": "Cam 2", "url": "rtsp://admin:pass%40123@192.168.1.50:554/stream", "is_entrance": False},
]
```

### 2. Run the Script
Execute the script using Python:

```bash
python run_cameras_with_face_tracking.py
```

### Key Controls
- **Q**: Quit the application.
- **Console Output**: Shows detection logs and API connection status.

---

## 📂 Video File Processing
To run detection on a pre-recorded video file instead of live cameras:

1. Place your video file in the project folder (e.g., `video.mp4`).
2. Run the video processing script:
   ```bash
   python videorunning.py
   ```
   *(Note: Edit `videorunning.py` to point to your specific video filename)*

---

## 📁 Directory Structure
- **`app/`**: Backend server code (FastAPI).
- **`evidence/`**: Saved violation video clips (auto-generated).
- **`output_videos/`**: Processed video outputs.
- **`templates/`**: HTML dashboards.

## ❓ Troubleshooting
- **Server Connection Failed**: Ensure the server is running on port 8000 *before* starting the camera script.
- **CUDA/GPU Error**: Verify NVIDIA drivers and PyTorch CUDA installation (`import torch; print(torch.cuda.is_available())`).
- **RTSP Lag**: Switch to TCP transport in OpenCV or check network bandwidth.
