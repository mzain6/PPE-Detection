#tests/test_video_stream.py
import cv2
import numpy as np
import pytest
from app.utils.video_stream import VideoStream, frame_to_bgr

class DummyCap:
    def __init__(self, frame):
        self._frame = frame
        self._opened = True
    def isOpened(self):
        return self._opened
    def read(self):
        return True, self._frame
    def release(self):
        self._opened = False

def make_jpeg_bytes():
    img = np.zeros((100,100,3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()

def test_webcam_read_monkeypatch(monkeypatch):
    # monkeypatch cv2.VideoCapture to return dummy cap
    import app.utils.video_stream as vsmod
    frame = (np.zeros((10,10,3), dtype=np.uint8))
    def fake_vc(arg):
        return DummyCap(frame)
    monkeypatch.setattr("cv2.VideoCapture", fake_vc)
    vs = VideoStream.from_webcam(0)
    f = vs.read(timeout=1.0)
    assert f is not None
    assert f.shape == frame.shape

def test_ffmpeg_fallback(monkeypatch):
    # simulate ffmpeg stdout producing a single jpeg
    jpeg = make_jpeg_bytes()
    class FakeProc:
        def __init__(self, data):
            from io import BytesIO
            self.stdout = BytesIO(data)
            self.proc = self
        def kill(self): pass
    def fake_popen(cmd, stdout, stderr):
        return FakeProc(jpeg)
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    # ensure cv2.VideoCapture fails
    def fake_vc_fail(arg, backend=None):
        class C: 
            def isOpened(self): return False
        return C()
    monkeypatch.setattr("cv2.VideoCapture", fake_vc_fail)
    vs = VideoStream("rtsp://example", fps=5)
    f = vs.read(timeout=1.0)
    assert f is not None
    assert f.ndim == 3