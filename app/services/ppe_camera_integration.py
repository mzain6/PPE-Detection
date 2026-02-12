"""
ppe_camera_integration.py

Integrates PPE detection with multi-camera manager using batch processing.
Provides detection callbacks and manages shared detector with per-camera trackers.
"""

import logging
import time
import cv2
import numpy as np
import threading
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import sys

# Import existing components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.ai.yolov8_detector import YoloV8Detector
from app.config import settings
from app.services.batch_coordinator import BatchCoordinator

# Import tracker
try:
    from app.ai.tracker import IOUTracker
except Exception:
    IOUTracker = None

# Import face detection components
try:
    from app.ai.face_detector import FaceDetector
    from app.ai.face_embedder import FaceEmbedder
    from app.ai.face_tracker import FaceTracker
    FACE_DETECTION_AVAILABLE = True
except Exception as e:
    FaceDetector = None
    FaceEmbedder = None
    FaceTracker = None
    FACE_DETECTION_AVAILABLE = False
    logger.warning(f"Face detection not available: {e}")

# Import GPU utils
try:
    from app.utils.gpu_utils import get_gpu_memory_info, calculate_max_model_instances
    GPU_UTILS_AVAILABLE = True
except Exception:
    GPU_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PPECameraIntegration:
    """
    Manages PPE detection for multiple cameras using shared model and batch processing.
    
    Architecture:
    - Single shared YoloV8Detector for all cameras (saves GPU memory)
    - Per-camera IOUTracker instances for person ID tracking
    - BatchCoordinator for efficient multi-frame GPU processing
    - Falls back to single-frame mode if batch disabled or single camera
    """
    
    def __init__(self, model_path: str = None, device: str = "auto", 
                 batch_enabled: bool = None, batch_size: int = None, batch_timeout_ms: int = None):
        """
        Args:
            model_path: Path to YOLOv8 model weights
            device: 'auto', 'cpu', 'cuda', or 'cuda:0'
            batch_enabled: Enable batch processing (default from config)
            batch_size: Max frames per batch (default from config)
            batch_timeout_ms: Batch timeout in ms (default from config)
        """
        self.model_path = model_path or settings.model_path
        self.device = device
        
        # Batch processing configuration
        if batch_enabled is None:
            # Load from config if available
            batch_enabled = getattr(settings, 'batch_enabled', True)
        if batch_size is None:
            batch_size = getattr(settings, 'batch_size', 8)
        if batch_timeout_ms is None:
            batch_timeout_ms = getattr(settings, 'batch_timeout_ms', 100)
        
        self.batch_enabled = batch_enabled
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
        # Single shared detector for all cameras
        self.shared_detector: Optional[YoloV8Detector] = None
        self.detector_lock = threading.Lock()
        
        # Per-camera trackers (separate tracking for each camera)
        self.trackers: Dict[str, Any] = {}  # camera_id -> IOUTracker or FaceTracker
        
        # Face detection components (shared across cameras)
        self.face_detection_enabled = settings.face_detection_enabled and FACE_DETECTION_AVAILABLE
        self.face_detector: Optional[FaceDetector] = None
        self.face_embedder: Optional[FaceEmbedder] = None
        
        # Batch coordinator (if batch processing enabled)
        self.batch_coordinator: Optional[BatchCoordinator] = None
        self.batch_thread: Optional[threading.Thread] = None
        self.batch_running = False
        
        # Per-camera result storage
        self.latest_results: Dict[str, Tuple[Dict, np.ndarray]] = {}  # camera_id -> (detections, annotated_frame)
        self.results_lock = threading.Lock()
        
        # Statistics
        self.frame_counts = {}
        self.detection_times = {}
        
        # Initialize shared detector
        self._initialize_shared_detector()
        
        # Initialize face detection if enabled
        if self.face_detection_enabled:
            self._initialize_face_detection()
        
        # Initialize batch processing if enabled
        if self.batch_enabled:
            self._initialize_batch_processing()
        
        logger.info(f"PPECameraIntegration initialized (model: {self.model_path}, device: {device}, batch: {self.batch_enabled})")
    
    def _initialize_shared_detector(self):
        """Initialize single shared detector for all cameras."""
        with self.detector_lock:
            if self.shared_detector is None:
                logger.info("Creating shared YOLOv8 detector...")
                self.shared_detector = YoloV8Detector(
                    model_path=self.model_path,
                    device=self.device,
                    camera_id="shared",  # Shared detector
                    person_model_path="yolov8n.pt",
                    ppe_model_path="best.pt"
                )
                logger.info("Shared detector created successfully")
    
    def _initialize_face_detection(self):
        """Initialize face detection and embedding components."""
        try:
            logger.info("Initializing face detection...")
            
            # Initialize face detector
            self.face_detector = FaceDetector(
                model_path=settings.face_model_path,
                confidence_threshold=settings.face_confidence_threshold,
                device=self.device
            )
            
            # Initialize face embedder
            self.face_embedder = FaceEmbedder(model="large")
            
            logger.info("Face detection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face detection: {e}")
            self.face_detection_enabled = False
            self.face_detector = None
            self.face_embedder = None
    
    def _initialize_batch_processing(self):
        """Initialize batch coordinator and processing thread."""
        self.batch_coordinator = BatchCoordinator(
            batch_size=self.batch_size,
            timeout_ms=self.batch_timeout_ms
        )
        
        # Start batch processing thread
        self.batch_running = True
        self.batch_thread = threading.Thread(target=self._batch_processing_loop, daemon=True)
        self.batch_thread.start()
        logger.info("Batch processing thread started")
    
    def _batch_processing_loop(self):
        """Background thread that processes frame batches."""
        logger.info("Batch processing loop started")
        
        while self.batch_running:
            try:
                # Get a batch of frames
                camera_ids, frames, timestamps = self.batch_coordinator.get_batch()
                
                if not frames:
                    time.sleep(0.01)  # Brief sleep if no frames
                    continue
                
                # Run batch inference
                start_time = time.time()
                with self.detector_lock:
                    if self.shared_detector:
                        batch_results = self.shared_detector.infer_batch(frames)
                    else:
                        batch_results = []
                
                detection_time = time.time() - start_time
                
                # Process results for each camera
                for camera_id, result, frame in zip(camera_ids, batch_results, frames):
                    try:
                        # Apply per-camera tracking (pass frame for face detection)
                        tracked_result = self._apply_tracking(camera_id, result, frame)
                        
                        # Annotate frame
                        annotated_frame = self._annotate_frame(frame.copy(), tracked_result.get('tracks', []), camera_id)
                        
                        # Store result
                        with self.results_lock:
                            self.latest_results[camera_id] = (tracked_result, annotated_frame)
                            
                            # Update statistics
                            if camera_id not in self.frame_counts:
                                self.frame_counts[camera_id] = 0
                                self.detection_times[camera_id] = []
                            
                            self.frame_counts[camera_id] += 1
                            self.detection_times[camera_id].append(detection_time / len(frames))  # Per-frame time
                            
                            # Keep only last 100 timing samples
                            if len(self.detection_times[camera_id]) > 100:
                                self.detection_times[camera_id].pop(0)
                    
                    except Exception as e:
                        logger.error(f"Error processing batch result for {camera_id}: {e}")
            
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                time.sleep(0.1)
        
        logger.info("Batch processing loop stopped")
    
    def _apply_tracking(self, camera_id: str, detection_result: Dict, frame: Optional[np.ndarray] = None) -> Dict:
        """
        Apply per-camera tracking to detection results with face-based re-identification.
        
        Args:
            camera_id: Camera identifier
            detection_result: Raw detection result from detector
            frame: Optional frame for face detection
        
        Returns:
            Detection result with tracking applied
        """
        # Get or create tracker for this camera
        if camera_id not in self.trackers:
            if self.face_detection_enabled and settings.face_reidentification_enabled and FaceTracker is not None:
                # Use face-based tracker
                self.trackers[camera_id] = FaceTracker(
                    similarity_threshold=settings.face_similarity_threshold,
                    max_face_age_seconds=settings.face_max_age_seconds,
                    min_stable_frames=settings.face_min_stable_frames
                )
                logger.info(f"Created FaceTracker for camera {camera_id}")
            elif IOUTracker is not None:
                # Fall back to IOU tracker
                self.trackers[camera_id] = IOUTracker(
                    iou_thresh=float(settings.iou_threshold),
                    max_missing=int(settings.max_missing_frames),
                    stable_frames=int(settings.stable_frames)
                )
                logger.info(f"Created IOUTracker for camera {camera_id}")
            else:
                self.trackers[camera_id] = None
        
        tracker = self.trackers[camera_id]
        
        # Get tracks from detection result (detector already outputs tracks with PPE data)
        # The key is 'tracks' NOT 'persons' - the detector does initial tracking
        input_tracks = detection_result.get('tracks', [])
        timestamp = detection_result.get('timestamp', time.time())
        
        # If no tracks, return empty
        if not input_tracks:
            return {"camera_id": camera_id, "timestamp": timestamp, "tracks": []}
        
        # If using face tracker for re-identification
        if isinstance(tracker, FaceTracker) and self.face_detection_enabled and frame is not None:
            # Extract face embeddings from each track (person)
            face_embeddings = self._extract_face_embeddings(input_tracks, frame)
            
            # Update face tracker - this reassigns IDs based on face recognition
            face_tracked = tracker.update(input_tracks, face_embeddings, timestamp)
            
            # Merge face tracking results with original track data (preserve PPE info)
            tracks_out = []
            for face_track, orig_track in zip(face_tracked, input_tracks):
                # Use face-based ID but keep all original data (bbox, ppe, etc.)
                merged = {
                    "track_id": face_track.get("track_id", orig_track.get("track_id", -1)),
                    "bbox": orig_track.get("bbox", face_track.get("bbox", {})),
                    "person_confidence": orig_track.get("person_confidence", 0.0),
                    "ppe": orig_track.get("ppe", []),  # Keep PPE data from detector
                    "stable": face_track.get("stable", True),
                    "has_face": face_track.get("has_face", False)
                }
                tracks_out.append(merged)
            
            return {"camera_id": camera_id, "timestamp": timestamp, "tracks": tracks_out}
        
        # No face tracking - just pass through the tracks from detector
        # Add has_face: False to each track
        tracks_out = []
        for track in input_tracks:
            track_copy = dict(track)
            track_copy["has_face"] = False
            tracks_out.append(track_copy)
        
        return {"camera_id": camera_id, "timestamp": timestamp, "fps": detection_result.get("fps", 0), "tracks": tracks_out}
    
    def _extract_face_embeddings(self, persons: List[Dict], frame: np.ndarray) -> List[Optional[np.ndarray]]:
        """
        Extract face embeddings from person detections.
        
        Args:
            persons: List of person detection dictionaries
            frame: Full frame image
        
        Returns:
            List of face embeddings (128-dim arrays or None)
        """
        if not self.face_detection_enabled or self.face_detector is None or self.face_embedder is None:
            return [None] * len(persons)
        
        face_embeddings = []
        
        for person in persons:
            bbox = person.get('bbox', {})
            
            try:
                # Extract person crop from frame
                x1 = int(bbox.get('x1', 0))
                y1 = int(bbox.get('y1', 0))
                x2 = int(bbox.get('x2', 0))
                y2 = int(bbox.get('y2', 0))
                
                # Bounds check
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                person_crop = frame[y1:y2, x1:x2]
                
                if person_crop.size == 0:
                    face_embeddings.append(None)
                    continue
                
                # Detect face in person crop
                face_result = self.face_detector.detect_face(person_crop)
                
                if face_result is None:
                    face_embeddings.append(None)
                    continue
                
                # Extract embedding from face
                face_crop = face_result['face_crop']
                embedding = self.face_embedder.extract_embedding(face_crop, num_jitters=1)
                
                face_embeddings.append(embedding)
                
            except Exception as e:
                logger.debug(f"Face embedding extraction error: {e}")
                face_embeddings.append(None)
        
        return face_embeddings
    
    def create_detection_callback(self, camera_id: str):
        """
        Create a detection callback function for a specific camera.
        This function is passed to CameraProcessor.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Callback function (camera_id, frame) -> (detections, annotated_frame)
        """
        def detection_callback(cam_id: str, frame: np.ndarray) -> Tuple[Dict, np.ndarray]:
            """
            Process frame and return detections.
            
            Args:
                cam_id: Camera ID (provided by CameraProcessor)
                frame: Frame to process
                
            Returns:
                (detections_dict, annotated_frame)
            """
            try:
                if self.batch_enabled and self.batch_coordinator:
                    # Submit frame to batch coordinator
                    self.batch_coordinator.submit_frame(camera_id, frame)
                    
                    # Return latest available result (may be from previous frame)
                    with self.results_lock:
                        if camera_id in self.latest_results:
                            return self.latest_results[camera_id]
                        else:
                            # No result yet - annotate frame with empty tracks as placeholder
                            empty_result = {'camera_id': camera_id, 'tracks': []}
                            placeholder_frame = self._annotate_frame(frame.copy(), [], camera_id)
                            return empty_result, placeholder_frame
                else:
                    # Fallback to single-frame processing
                    start_time = time.time()
                    
                    with self.detector_lock:
                        if self.shared_detector:
                            result = self.shared_detector.infer(frame)
                        else:
                            result = {'camera_id': camera_id, 'tracks': []}
                    
                    detection_time = time.time() - start_time
                    
                    # Annotate frame
                    annotated_frame = self._annotate_frame(frame.copy(), result.get('tracks', []), cam_id)
                    
                    # Update statistics
                    if camera_id not in self.frame_counts:
                        self.frame_counts[camera_id] = 0
                        self.detection_times[camera_id] = []
                    
                    self.frame_counts[camera_id] += 1
                    self.detection_times[camera_id].append(detection_time)
                    
                    if len(self.detection_times[camera_id]) > 100:
                        self.detection_times[camera_id].pop(0)
                    
                    return result, annotated_frame
                    
            except Exception as e:
                logger.error(f"Detection error for camera {camera_id}: {e}")
                return {'tracks': [], 'camera_id': camera_id}, frame
        
        return detection_callback
    
    def _annotate_frame(self, frame: np.ndarray, tracks: List[Dict], camera_id: str) -> np.ndarray:
        """
        Annotate frame with detection results.
        
        Args:
            frame: Frame to annotate
            tracks: List of track dictionaries
            camera_id: Camera identifier
            
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        
        # Debug logging
        logger.debug(f"Annotating frame for {camera_id}: {len(tracks)} tracks")
        if len(tracks) > 0:
            logger.debug(f"First track: {tracks[0]}")
        
        # Add camera ID
        cv2.putText(frame, f"Camera: {camera_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Process each track - ALWAYS draw boxes
        for idx, track in enumerate(tracks):
            logger.debug(f"Processing track {idx}: {track.keys()}")
            person_id = track.get('track_id', -1)
            bbox = track.get('bbox', {})
            
            # Extract bbox coordinates
            if isinstance(bbox, dict):
                x1 = int(bbox.get('x1', 0))
                y1 = int(bbox.get('y1', 0))
                x2 = int(bbox.get('x2', 0))
                y2 = int(bbox.get('y2', 0))
            else:
                continue
            
            # Get PPE info
            ppe_list = track.get('ppe', [])
            helmet_ok = any(p.get('label') in ['head_helmet', 'helmet'] for p in ppe_list)
            vest_ok = any(p.get('label') == 'vest' for p in ppe_list)
            
            # Determine overall status
            if helmet_ok and vest_ok:
                color = (0, 255, 0)  # Green - compliant
                status = "COMPLIANT"
            elif not helmet_ok and not vest_ok:
                color = (0, 0, 255)  # Red - both missing
                status = "NO PPE"
            else:
                color = (0, 165, 255)  # Orange - partial
                status = "PARTIAL PPE"
            
            # Draw person bounding box (always show)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw person label with ID and status
            label = f"Person ID:{person_id} - {status}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Background for text
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 12), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            # Text
            cv2.putText(frame, label, (x1 + 5, y1 - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw individual PPE items (helmet, vest) with their own boxes
            helmet_detected = False
            vest_detected = False
            
            for ppe_item in ppe_list:
                ppe_label = ppe_item.get('label', '')
                ppe_bbox = ppe_item.get('bbox', None)
                ppe_conf = ppe_item.get('conf', ppe_item.get('confidence', 0.0))
                
                # Handle bbox as tuple or dict
                if ppe_bbox is None:
                    continue
                
                if isinstance(ppe_bbox, (tuple, list)) and len(ppe_bbox) >= 4:
                    # Tuple format: (x1, y1, x2, y2)
                    px1 = int(ppe_bbox[0])
                    py1 = int(ppe_bbox[1])
                    px2 = int(ppe_bbox[2])
                    py2 = int(ppe_bbox[3])
                elif isinstance(ppe_bbox, dict):
                    # Dict format: {x1, y1, x2, y2}
                    px1 = int(ppe_bbox.get('x1', 0))
                    py1 = int(ppe_bbox.get('y1', 0))
                    px2 = int(ppe_bbox.get('x2', 0))
                    py2 = int(ppe_bbox.get('y2', 0))
                else:
                    continue
                
                # Determine color and display name based on PPE type
                if ppe_label in ['head_helmet', 'helmet']:
                    helmet_detected = True
                    ppe_color = (0, 255, 0)  # Green for detected helmet
                    display_name = f"Person Helmet ({ppe_conf:.2f})"
                elif ppe_label == 'vest':
                    vest_detected = True
                    ppe_color = (0, 255, 0)  # Green for detected vest  
                    display_name = f"Safety Vest ({ppe_conf:.2f})"
                else:
                    ppe_color = (255, 255, 0)  # Cyan for other PPE
                    display_name = f"{ppe_label} ({ppe_conf:.2f})"
                
                # Draw PPE item box
                cv2.rectangle(frame, (px1, py1), (px2, py2), ppe_color, 2)
                
                # Draw PPE label
                ppe_label_size = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (px1, py1 - ppe_label_size[1] - 10),
                             (px1 + ppe_label_size[0] + 5, py1), ppe_color, -1)
                cv2.putText(frame, display_name, (px1 + 2, py1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw missing PPE indicators (red boxes)
            # Estimate helmet position (head area - top 25% of person bbox)
            if not helmet_detected:
                helmet_height = int((y2 - y1) * 0.25)
                helmet_y1 = y1
                helmet_y2 = y1 + helmet_height
                helmet_x1 = x1 + int((x2 - x1) * 0.2)
                helmet_x2 = x2 - int((x2 - x1) * 0.2)
                
                # Draw red box for missing helmet
                cv2.rectangle(frame, (helmet_x1, helmet_y1), (helmet_x2, helmet_y2), (0, 0, 255), 2)
                missing_helmet_label = "NO HELMET"
                cv2.rectangle(frame, (helmet_x1, helmet_y1 - 30), 
                             (helmet_x1 + 140, helmet_y1), (0, 0, 255), -1)
                cv2.putText(frame, missing_helmet_label, (helmet_x1 + 2, helmet_y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Estimate vest position (torso area - middle 50% of person bbox)
            if not vest_detected:
                vest_height = int((y2 - y1) * 0.4)
                vest_y1 = y1 + int((y2 - y1) * 0.25)
                vest_y2 = vest_y1 + vest_height
                vest_x1 = x1 + int((x2 - x1) * 0.15)
                vest_x2 = x2 - int((x2 - x1) * 0.15)
                
                # Draw red box for missing vest
                cv2.rectangle(frame, (vest_x1, vest_y1), (vest_x2, vest_y2), (0, 0, 255), 2)
                missing_vest_label = "NO VEST"
                cv2.rectangle(frame, (vest_x1, vest_y1 - 30),
                             (vest_x1 + 115, vest_y1), (0, 0, 255), -1)
                cv2.putText(frame, missing_vest_label, (vest_x1 + 2, vest_y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add FPS indicator in top right
        if hasattr(self, 'latest_fps') and self.latest_fps > 0:
            fps_text = f"FPS: {self.latest_fps:.1f}"
            fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(frame, fps_text, (w - fps_size[0] - 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add track ID indicator in top right
        tracked_count = len([t for t in tracks if t.get('stable', False)])
        track_text = f"Tracks: {tracked_count}/{len(tracks)}"
        cv2.putText(frame, track_text, (w - 200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add detection count
        count_text = f"Persons: {len(tracks)}"
        cv2.putText(frame, count_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add batch mode indicator
        if self.batch_enabled:
            cv2.putText(frame, "BATCH MODE", (10, h - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame
    
    def check_gpu_memory_available(self, camera_id: str) -> bool:
        """
        Check if sufficient GPU memory is available before adding camera.
        
        Args:
            camera_id: Camera to add
        
        Returns:
            True if memory available or GPU not used
        """
        if not GPU_UTILS_AVAILABLE:
            return True  # Can't check, assume OK
        
        try:
            mem_info = get_gpu_memory_info(0)
            free_gb = mem_info.get('free_gb', 0)
            
            if free_gb < 0.5:  # Less than 500MB free
                logger.warning(f"Insufficient GPU memory for camera {camera_id}: {free_gb:.2f}GB free")
                return False
            
            return True
        except Exception as e:
            logger.debug(f"Error checking GPU memory: {e}")
            return True  # Assume OK if can't check
    
    def get_stats(self, camera_id: str = None) -> Dict:
        """
        Get detection statistics.
        
        Args:
            camera_id: Specific camera or None for all cameras
            
        Returns:
            Statistics dictionary
        """
        if camera_id and camera_id in self.frame_counts:
            avg_time = np.mean(self.detection_times[camera_id]) if self.detection_times[camera_id] else 0
            stats = {
                'camera_id': camera_id,
                'total_frames': self.frame_counts[camera_id],
                'avg_detection_time_ms': round(avg_time * 1000, 2),
                'inference_fps': round(1 / avg_time, 2) if avg_time > 0 else 0
            }
            
            # Add batch stats if batch processing enabled
            if self.batch_enabled and self.batch_coordinator:
                batch_stats = self.batch_coordinator.get_stats()
                stats['batch_stats'] = batch_stats
            
            return stats
        else:
            # All cameras
            all_stats = {}
            for cam_id in self.frame_counts:
                all_stats[cam_id] = self.get_stats(cam_id)
            
            # Add shared batch stats
            if self.batch_enabled and self.batch_coordinator:
                all_stats['_batch_coordinator'] = self.batch_coordinator.get_stats()
            
            return all_stats
    
    def get_memory_stats(self) -> Dict:
        """Get GPU memory statistics."""
        stats = {}
        
        if GPU_UTILS_AVAILABLE:
            try:
                mem_info = get_gpu_memory_info(0)
                stats['gpu_memory'] = mem_info
            except Exception as e:
                logger.debug(f"Error getting GPU memory stats: {e}")
        
        if self.shared_detector:
            try:
                stats['model_memory_mb'] = self.shared_detector.get_model_memory_usage()
            except Exception as e:
                logger.debug(f"Error getting model memory: {e}")
        
        return stats
    
    def remove_camera(self, camera_id: str):
        """
        Remove tracker for camera and cleanup.
        
        Args:
            camera_id: Camera to remove
        """
        if camera_id in self.trackers:
            try:
                del self.trackers[camera_id]
                del self.frame_counts[camera_id]
                del self.detection_times[camera_id]
                
                with self.results_lock:
                    if camera_id in self.latest_results:
                        del self.latest_results[camera_id]
                
                logger.info(f"Removed tracker and stats for camera {camera_id}")
            except Exception as e:
                logger.error(f"Error removing camera {camera_id}: {e}")
    
    def shutdown(self):
        """Shutdown all detectors and batch processing."""
        logger.info("Shutting down PPECameraIntegration")
        
        # Stop batch processing
        if self.batch_enabled and self.batch_thread:
            self.batch_running = False
            if self.batch_thread:
                self.batch_thread.join(timeout=2.0)
            logger.info("Batch processing stopped")
        
        # Clean up trackers
        camera_ids = list(self.trackers.keys())
        for camera_id in camera_ids:
            self.remove_camera(camera_id)
        
        # Clean up shared detector
        with self.detector_lock:
            if self.shared_detector:
                try:
                    self.shared_detector.close()
                except Exception as e:
                    logger.error(f"Error closing shared detector: {e}")
                self.shared_detector = None
        
        logger.info("PPECameraIntegration shutdown complete")
