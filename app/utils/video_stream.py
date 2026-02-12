#app/utils/video_stream.py
from typing import Optional, Union
import time
import cv2
import numpy as np
import subprocess
import threading
import io

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
            # stderr suppressed to avoid noise; ensure stdout is a pipe
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read_jpeg_frame(self, timeout: float = 5.0) -> Optional[bytes]:
        """
        Read one JPEG frame from stdout by searching JPEG SOI/EOI markers.
        """
        if self.proc is None or self.proc.stdout is None:
            return None
        start = time.time()
        data = b""
        stdout = self.proc.stdout
        # Read until we find a full JPEG (0xFFD8 ... 0xFFD9) or timeout
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
                # keep remainder in buffer by seeking back (not possible), so just drop
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
    Unified VideoStream supporting webcam (device index) and RTSP URL with ffmpeg fallback.
    """
    def __init__(self, source: Union[int, str], fps: int = 5, backend: int = cv2.CAP_FFMPEG):
        self.source = source
        self.fps = max(1, int(fps) if fps else 1)
        self.backend = backend
        self.cap: Optional[cv2.VideoCapture] = None
        self.ffmpeg: Optional[_FFMPEGProcess] = None
        self.last_read: Optional[float] = None

    def start(self) -> "VideoStream":
        # If webcam (int), open cv2 capture
        if isinstance(self.source, int):
            if self.cap is None or not getattr(self.cap, "isOpened", lambda: False)():
                self.cap = cv2.VideoCapture(int(self.source))
        else:
            # Try cv2 first
            if self.cap is None or not getattr(self.cap, "isOpened", lambda: False)():
                try:
                    self.cap = cv2.VideoCapture(str(self.source), self.backend)
                except Exception:
                    self.cap = None
            # If cv2 cannot open, prepare ffmpeg process lazily
            if (self.cap is None or not getattr(self.cap, "isOpened", lambda: False)()) and self.ffmpeg is None:
                self.ffmpeg = _FFMPEGProcess(str(self.source))
                try:
                    self.ffmpeg.start()
                except Exception:
                    # ignore; ffmpeg may not be installed
                    self.ffmpeg = None
        return self

    def read(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """
        Attempt to read a single frame within `timeout`. Prefer cv2; fall back to ffmpeg-JPEG.
        """
        if self.cap is None:
            self.start()
        # Try cv2 capture
        if self.cap is not None and getattr(self.cap, "isOpened", lambda: False)():
            deadline = time.time() + float(timeout)
            while time.time() < deadline:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.last_read = time.time()
                    return frame
                time.sleep(max(0.01, 1.0 / (self.fps * 2)))
            # cv2 failed to provide a frame within timeout; try ffmpeg fallback
        # FFmpeg fallback
        if self.ffmpeg is None:
            self.start()  # may start ffmpeg
        if self.ffmpeg:
            jpeg = self.ffmpeg.read_jpeg_frame(timeout=timeout)
            if jpeg:
                frame = frame_to_bgr(jpeg)
                if frame is not None:
                    self.last_read = time.time()
                    return frame
        return None

    def is_open(self) -> bool:
        if self.cap and getattr(self.cap, "isOpened", lambda: False)():
            return True
        if self.ffmpeg and self.ffmpeg.proc:
            return True
        return False

    def release(self) -> None:
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
    def from_rtsp(cls, url: str, fps: int = 5) -> "VideoStream":
        return cls(source=url, fps=fps)

    @classmethod
    def from_webcam(cls, index: int = 0, fps: int = 5) -> "VideoStream":
        return cls(source=index, fps=fps)