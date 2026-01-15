# app/ai/base.py
from __future__ import annotations
from typing import Any, Dict
from abc import ABC, abstractmethod

class Detector(ABC):
    """
    Detector interface - each detector is independent and returns JSON-serializable dicts.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model resources (called during init/start)."""

    @abstractmethod
    def infer(self, frame) -> Dict[str, Any]:
        """
        Run inference on a single BGR numpy frame and return a JSON-serializable dict:
        {
          "camera_id": str,
          "timestamp": float,
          "fps": int,
          "tracks": [...]
        }
        """

    @abstractmethod
    def close(self) -> None:
        """Release model resources."""