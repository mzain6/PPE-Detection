#tests/test_camera_service.py
import numpy as np
import pytest
from app.services import camera_service

class DummyStream:
    def __init__(self, frame):
        self.frame = frame
    def read(self, timeout=1.0):
        return self.frame

def test_register_and_health(monkeypatch):
    # create dummy frame and monkeypatch VideoStream.from_rtsp
    frame = np.zeros((20,20,3), dtype=np.uint8)
    def fake_from_rtsp(url, fps=5):
        return DummyStream(frame)
    monkeypatch.setattr("app.services.camera_service.VideoStream.from_rtsp", fake_from_rtsp)
    req = type("R", (), {"camera_id":"testcam","rtsp_url":"rtsp://x","fps":5,"roi":None})
    rec = camera_service.register_camera(req)
    healthy, last, msg = camera_service.health_check("testcam", timeout=0.5)
    assert healthy is True
    # cleanup
    # not implemented: in-memory registry persists; tests run in fresh env or need isolation