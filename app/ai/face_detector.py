"""
Face Detector using YOLOv8m-face model.
Detects faces within person bounding boxes for re-identification.
"""

import logging
import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Import YOLO
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logger.warning("ultralytics not available - face detection disabled")


class FaceDetector:
    """
    YOLOv8m-based face detector for extracting faces from person crops.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, device: Optional[str] = None):
        """
        Args:
            model_path: Path to YOLOv8m-face weights
            confidence_threshold: Minimum confidence for face detection
            device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._loaded = False
        
        # Validate model file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Face detection model not found: {model_path}")
        
        # Load model
        self.load()
    
    def load(self):
        """Load the YOLOv8m-face model."""
        if self._loaded:
            return
        
        if YOLO is None:
            raise RuntimeError("ultralytics not available - cannot load face detector")
        
        try:
            logger.info(f"Loading face detection model: {self.model_path}")
            
            # Load model
            if self.device:
                try:
                    self.model = YOLO(self.model_path, device=self.device)
                except TypeError:
                    # Fallback for older ultralytics versions
                    self.model = YOLO(self.model_path)
                    if self.device:
                        try:
                            self.model.to(self.device)
                        except Exception:
                            logger.debug("model.to(device) not supported")
            else:
                self.model = YOLO(self.model_path)
            
            self._loaded = True
            logger.info(f"Face detector loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load face detection model: {e}")
            raise
    
    def detect_face(self, person_crop: np.ndarray) -> Optional[Dict]:
        """
        Detect face in a person crop image.
        
        Args:
            person_crop: Cropped image of person (numpy array)
        
        Returns:
            Dict with face info or None if no face detected
            {
                'bbox': [x1, y1, x2, y2],  # Relative to crop
                'confidence': float,
                'face_crop': np.ndarray
            }
        """
        if not self._loaded or self.model is None:
            logger.warning("Face detector not loaded")
            return None
        
        if person_crop is None or person_crop.size == 0:
            return None
        
        try:
            # Run face detection on the person crop
            results = self.model(person_crop, verbose=False)
            
            if not results or len(results) == 0:
                return None
            
            result = results[0]
            
            # Check if any faces detected
            if result.boxes is None or len(result.boxes) == 0:
                return None
            
            # Get the most confident face
            boxes = result.boxes
            confidences = boxes.conf.cpu().numpy()
            
            # Filter by confidence threshold
            valid_indices = np.where(confidences >= self.confidence_threshold)[0]
            
            if len(valid_indices) == 0:
                return None
            
            # Get the most confident face
            best_idx = valid_indices[np.argmax(confidences[valid_indices])]
            
            # Extract bbox coordinates
            bbox = boxes.xyxy[best_idx].cpu().numpy()
            confidence = float(confidences[best_idx])
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure bbox is within image bounds
            h, w = person_crop.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Extract face crop
            face_crop = person_crop[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            return {
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'face_crop': face_crop
            }
            
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
            return None
    
    def detect_faces_batch(self, person_crops: List[np.ndarray]) -> List[Optional[Dict]]:
        """
        Detect faces in a batch of person crops.
        
        Args:
            person_crops: List of person crop images
        
        Returns:
            List of face detection results (same length as input)
        """
        if not self._loaded or self.model is None:
            return [None] * len(person_crops)
        
        if not person_crops:
            return []
        
        try:
            # Run batch inference
            results = self.model(person_crops, verbose=False)
            
            face_results = []
            
            for idx, (result, crop) in enumerate(zip(results, person_crops)):
                if result.boxes is None or len(result.boxes) == 0:
                    face_results.append(None)
                    continue
                
                boxes = result.boxes
                confidences = boxes.conf.cpu().numpy()
                
                # Filter by confidence
                valid_indices = np.where(confidences >= self.confidence_threshold)[0]
                
                if len(valid_indices) == 0:
                    face_results.append(None)
                    continue
                
                # Get best face
                best_idx = valid_indices[np.argmax(confidences[valid_indices])]
                bbox = boxes.xyxy[best_idx].cpu().numpy()
                confidence = float(confidences[best_idx])
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Bounds check
                h, w = crop.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                face_crop = crop[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    face_results.append(None)
                    continue
                
                face_results.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'face_crop': face_crop
                })
            
            return face_results
            
        except Exception as e:
            logger.error(f"Batch face detection error: {e}")
            return [None] * len(person_crops)
    
    def convert_to_absolute_bbox(self, person_bbox: List[float], face_bbox: List[int]) -> List[float]:
        """
        Convert face bbox from person-crop coordinates to frame coordinates.
        
        Args:
            person_bbox: [x1, y1, x2, y2] in frame coordinates
            face_bbox: [x1, y1, x2, y2] in crop coordinates
        
        Returns:
            [x1, y1, x2, y2] in frame coordinates
        """
        px1, py1 = person_bbox[0], person_bbox[1]
        fx1, fy1, fx2, fy2 = face_bbox
        
        # Convert to frame coordinates
        abs_x1 = px1 + fx1
        abs_y1 = py1 + fy1
        abs_x2 = px1 + fx2
        abs_y2 = py1 + fy2
        
        return [abs_x1, abs_y1, abs_x2, abs_y2]
    
    def close(self):
        """Release model resources."""
        try:
            self.model = None
            self._loaded = False
        except Exception as e:
            logger.error(f"Error closing face detector: {e}")
