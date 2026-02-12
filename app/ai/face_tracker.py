"""
Face-based tracker for persistent person re-identification.
Maintains person IDs based on face recognition across frames.
"""

import logging
import time
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict

logger = logging.getLogger(__name__)


class FaceTracker:
    """
    Track persons using face embeddings for re-identification.
    Maintains persistent IDs even when persons leave and return.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.6,
                 max_face_age_seconds: float = 300.0,
                 min_stable_frames: int = 3):
        """
        Args:
            similarity_threshold: Max distance for face matching (lower = stricter)
            max_face_age_seconds: How long to remember faces after last seen
            min_stable_frames: Frames needed before ID considered stable
        """
        self.similarity_threshold = similarity_threshold
        self.max_face_age_seconds = max_face_age_seconds
        self.min_stable_frames = min_stable_frames
        
        # Known faces database: person_id -> face_data
        self.known_faces: OrderedDict[int, Dict] = OrderedDict()
        
        # Next person ID to assign
        self.next_id = 1
        
        # Temporary tracking for persons without faces
        self.temp_tracks: Dict[str, Dict] = {}  # bbox_key -> temp_data
        
        logger.info(f"FaceTracker initialized (threshold={similarity_threshold}, max_age={max_face_age_seconds}s)")
    
    def update(self, 
               persons: List[Dict],
               face_embeddings: List[Optional[np.ndarray]],
               timestamp: Optional[float] = None) -> List[Dict]:
        """
        Update tracker with new detections and face embeddings.
        
        Args:
            persons: List of person dictionaries with 'bbox', 'conf', 'ppe'
            face_embeddings: List of face embeddings (128-dim arrays or None)
            timestamp: Current timestamp (uses time.time() if None)
        
        Returns:
            List of tracked persons with persistent IDs
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Clean up old faces
        self._cleanup_old_faces(timestamp)
        
        tracked_persons = []
        
        for person,face_embedding in zip(persons, face_embeddings):
            bbox = person.get('bbox', {})
            conf = person.get('conf', person.get('person_confidence', 0.0))
            ppe = person.get('ppe', [])
            
            # Try to match with known faces
            if face_embedding is not None:
                person_id, stable = self._match_or_create_face(face_embedding, bbox, timestamp)
            else:
                # No face detected - assign temporary ID based on position
                person_id, stable = self._handle_no_face(bbox, timestamp)
            
            # Create tracked person
            tracked_person = {
                'track_id': person_id,
                'bbox': bbox,
                'person_confidence': float(conf),
                'ppe': ppe,
                'stable': stable,
                'has_face': face_embedding is not None
            }
            
            tracked_persons.append(tracked_person)
        
        return tracked_persons
    
    def _match_or_create_face(self, 
                              face_embedding: np.ndarray,
                              bbox: Dict,
                              timestamp: float) -> Tuple[int, bool]:
        """
        Match face embedding with known faces or create new entry.
        
        Returns:
            (person_id, is_stable)
        """
        best_match_id = None
        best_distance = float('inf')
        
        # Compare with all known faces
        for person_id, face_data in self.known_faces.items():
            known_embedding = face_data['embedding']
            
            # Calculate distance
            distance = np.linalg.norm(face_embedding - known_embedding)
            
            if distance < self.similarity_threshold and distance < best_distance:
                best_distance = distance
                best_match_id = person_id
        
        if best_match_id is not None:
            # Matched with known person
            self.known_faces[best_match_id]['last_seen'] = timestamp
            self.known_faces[best_match_id]['bbox'] = bbox
            self.known_faces[best_match_id]['match_count'] += 1
            
            # Update embedding (running average for robustness)
            alpha = 0.2  # Weight for new embedding
            old_emb = self.known_faces[best_match_id]['embedding']
            self.known_faces[best_match_id]['embedding'] = (
                alpha * face_embedding + (1 - alpha) * old_emb
            )
            
            is_stable = self.known_faces[best_match_id]['match_count'] >= self.min_stable_frames
            
            return (best_match_id, is_stable)
        
        else:
            # New person - create entry
            person_id = self.next_id
            self.next_id += 1
            
            self.known_faces[person_id] = {
                'embedding': face_embedding.copy(),
                'last_seen': timestamp,
                'bbox': bbox,
                'match_count': 1,
                'created_at': timestamp
            }
            
            logger.info(f"New person detected with ID {person_id}")
            
            return (person_id, False)
    
    def _handle_no_face(self, bbox: Dict, timestamp: float) -> Tuple[int, bool]:
        """
        Handle person without detected face (fallback tracking).
        
        Returns:
            (temp_id, is_stable=False)
        """
        # Create a temporary ID based on position
        # These are negative IDs to distinguish from face-based IDs
        temp_id = -(len(self.temp_tracks) + 1)
        
        # Simple bbox-based tracking (very basic)
        bbox_key = self._bbox_to_key(bbox)
        
        if bbox_key not in self.temp_tracks:
            self.temp_tracks[bbox_key] = {
                'id': temp_id,
                'last_seen': timestamp,
                'bbox': bbox
            }
        
        return (temp_id, False)
    
    def _bbox_to_key(self, bbox: Dict) -> str:
        """Convert bbox to string key for temporary tracking."""
        try:
            x1 = int(bbox.get('x1', 0))
            y1 = int(bbox.get('y1', 0))
            # Quantize to grid for robustness
            grid_x = x1 // 50
            grid_y = y1 // 50
            return f"{grid_x}_{grid_y}"
        except:
            return "0_0"
    
    def _cleanup_old_faces(self, current_time: float):
        """Remove faces not seen recently."""
        to_remove = []
        
        for person_id, face_data in self.known_faces.items():
            age = current_time - face_data['last_seen']
            if age > self.max_face_age_seconds:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.known_faces[person_id]
            logger.debug(f"Removed person ID {person_id} (not seen for {self.max_face_age_seconds}s)")
        
        # Also cleanup temp tracks
        temp_to_remove = []
        for bbox_key, temp_data in self.temp_tracks.items():
            age = current_time - temp_data['last_seen']
            if age > 30.0:  # 30 second timeout for temp tracks
                temp_to_remove.append(bbox_key)
        
        for bbox_key in temp_to_remove:
            del self.temp_tracks[bbox_key]
    
    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return {
            'known_faces': len(self.known_faces),
            'temp_tracks': len(self.temp_tracks),
            'next_id': self.next_id,
            'similarity_threshold': self.similarity_threshold,
            'max_age_seconds': self.max_face_age_seconds
        }
    
    def get_known_person_ids(self) -> List[int]:
        """Get list of all known person IDs."""
        return list(self.known_faces.keys())
    
    def reset(self):
        """Reset tracker (clear all known faces)."""
        self.known_faces.clear()
        self.temp_tracks.clear()
        self.next_id = 1
        logger.info("FaceTracker reset")
