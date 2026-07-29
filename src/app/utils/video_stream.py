#app/utils/video_stream.py
from typing import Optional, Union
import time
import cv2
import numpy as np
import subprocess
import threading
import os

def frame_to_bgr(frame_bytes: bytes) -> Optional[np.ndarray]:
    if not frame_bytes:
        return None
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

class _FFMPEGProcess:
    """
    Minimal helper that launches ffmpeg and yields concatenated JPEG frames on stdout.
    """
    def __init__(self, url: str):
        self.url = url
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self.proc:
                return
            cmd = [
                "ffmpeg",
                "-rtsp_transport", "tcp",
                "-i", self.url,
                "-f", "image2pipe",
                "-q:v", "5",
                "-vcodec", "mjpeg",
                "-"
            ]
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read_jpeg_frame(self, timeout: float = 5.0) -> Optional[bytes]:
        if self.proc is None or self.proc.stdout is None:
            return None
        start = time.time()
        data = b""
        stdout = self.proc.stdout
        while time.time() - start < timeout:
            chunk = stdout.read(4096)
            if not chunk:
                time.sleep(0.01)
                continue
            data += chunk
            soi = data.find(b"\xff\xd8")
            eoi = data.find(b"\xff\xd9")
            if soi != -1 and eoi != -1 and eoi > soi:
                jpeg = data[soi:eoi + 2]
                return jpeg
        return None

    def stop(self):
        with self._lock:
            if self.proc:
                try:
                    self.proc.kill()
                except Exception:
                    pass
                self.proc = None

class VideoStream:
    """
    Threaded VideoStream supporting webcam (device index) and RTSP URL with zero-buffer latency.
    A background thread continuously reads frames so read() always returns the latest frame instantly.
    """
    def __init__(self, source: Union[int, str], fps: int = 15, backend: int = cv2.CAP_FFMPEG):
        if isinstance(source, str) and source.strip().isdigit():
            source = int(source.strip())
        self.source = source
        self.fps = max(1, int(fps) if fps else 15)
        self.backend = backend
        self.cap: Optional[cv2.VideoCapture] = None
        self.ffmpeg: Optional[_FFMPEGProcess] = None
        self.last_read: Optional[float] = None
        
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def start(self) -> "VideoStream":
        if self._worker_thread and self._worker_thread.is_alive():
            return self
        
        self._stop_event.clear()
        
        # Open OpenCV capture
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(int(self.source))
            if (self.cap is None or not getattr(self.cap, "isOpened", lambda: False)()) and os.name == "nt":
                self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
        else:
            if str(self.source).startswith("rtsp://"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            try:
                self.cap = cv2.VideoCapture(str(self.source), self.backend)
            except Exception:
                self.cap = None

        if self.cap and getattr(self.cap, "isOpened", lambda: False)():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Fall back to FFmpeg if cv2 failed
        if (self.cap is None or not getattr(self.cap, "isOpened", lambda: False)()) and not isinstance(self.source, int):
            self.ffmpeg = _FFMPEGProcess(str(self.source))
            try:
                self.ffmpeg.start()
            except Exception:
                self.ffmpeg = None

        # Start background grabber thread to prevent buffer buildup
        self._worker_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._worker_thread.start()
        return self

    def _capture_loop(self):
        while not self._stop_event.is_set():
            frame = None
            if self.cap and getattr(self.cap, "isOpened", lambda: False)():
                ret, img = self.cap.read()
                if ret and img is not None:
                    frame = img
            elif self.ffmpeg:
                jpeg = self.ffmpeg.read_jpeg_frame(timeout=0.2)
                if jpeg:
                    frame = frame_to_bgr(jpeg)
            
            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self.last_read = time.time()
                time.sleep(0.005)
            else:
                time.sleep(0.03)

    def read(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        """
        Return the latest real-time frame without buffering lag.
        """
        if not self._worker_thread or not self._worker_thread.is_alive():
            self.start()

        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            with self._frame_lock:
                if self._latest_frame is not None:
                    # Return latest frame
                    frame = self._latest_frame
                    return frame.copy()
            time.sleep(0.01)
        return None

    def is_open(self) -> bool:
        if self.cap and getattr(self.cap, "isOpened", lambda: False)():
            return True
        if self.ffmpeg and self.ffmpeg.proc:
            return True
        return False

    def release(self) -> None:
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            self.cap = None
        if self.ffmpeg:
            try:
                self.ffmpeg.stop()
            except Exception:
                pass
            self.ffmpeg = None

    @classmethod
    def from_rtsp(cls, url: str, fps: int = 15) -> "VideoStream":
        return cls(source=url, fps=fps)

    @classmethod
    def from_webcam(cls, index: int = 0, fps: int = 15) -> "VideoStream":
        return cls(source=index, fps=fps)