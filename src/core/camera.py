"""
camera.py

CameraSource provides a simple interface for USB and RTSP/IP cameras with auto-reconnect.
- source: integer (USB camera index) or string (RTSP/HTTP URL)
- reconnect_delay_sec: time to wait before reconnect attempts
- returns frames via read() or via generator stream_frames()
"""

import cv2
import time
from typing import Optional, Generator

class CameraSource:
    def __init__(self, source=0, reconnect_delay_sec: float = 3.0, read_timeout_sec: float = 5.0):
        """
        source: 0,1,... or rtsp://... url
        """
        self.source = source
        self.reconnect_delay = reconnect_delay_sec
        self.read_timeout = read_timeout_sec
        self.cap = None
        self.open()

    def open(self):
        if isinstance(self.source, str):
            # RTSP streams often need additional flags - leave default for portability
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(int(self.source))
        # small warm-up
        time.sleep(0.2)

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def read(self, reconnect=True):
        """
        Read a single frame. If reconnect is True, attempt to reopen stream on failure.
        Returns (ok: bool, frame)
        """
        if self.cap is None or not self.is_opened():
            if reconnect:
                try:
                    self.open()
                except Exception:
                    time.sleep(self.reconnect_delay)
                    return False, None
            else:
                return False, None

        ok, frame = self.cap.read()
        if not ok or frame is None:
            # try reconnect
            if reconnect:
                try:
                    self.cap.release()
                except Exception:
                    pass
                time.sleep(self.reconnect_delay)
                self.open()
            return False, None
        return True, frame

    def stream_frames(self, reconnect=True) -> Generator:
        """
        Yield frames continuously. This helper is ready for integration into an async endpoint or background task.
        """
        while True:
            ok, frame = self.read(reconnect=reconnect)
            if not ok:
                # small sleep to avoid busy-loop during reconnection
                time.sleep(self.reconnect_delay)
                continue
            yield frame

    def release(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass