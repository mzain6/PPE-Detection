"""
Face Embedder for extracting facial features for re-identification.
Uses face_recognition library (dlib-based) to extract 128-dimensional embeddings.
"""

import logging
import numpy as np
import cv2
from typing import Optional, List, Tuple
import sys

logger = logging.getLogger(__name__)

# Try to import face_recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition library not available - install with: pip install face-recognition")


class FaceEmbedder:
    """
    Extract face embeddings for person re-identification.
    Uses face_recognition (dlib) to create 128-dim feature vectors.
    """
    
    def __init__(self, model: str = "large"):
        """
        Args:
            model: 'small' (faster) or 'large' (more accurate)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            raise RuntimeError("face_recognition library not installed")
        
        self.model = model
        logger.info(f"FaceEmbedder initialized with {model} model")
    
    def extract_embedding(self, face_crop: np.ndarray, num_jitters: int = 1) -> Optional[np.ndarray]:
        """
        Extract 128-dimensional face embedding from face crop.
        
        Args:
            face_crop: Face image (numpy array, RGB or BGR)
            num_jitters: Number of re-samplings (higher = more accurate but slower)
        
        Returns:
            128-dim numpy array or None if face not found
        """
        if face_crop is None or face_crop.size == 0:
            return None
        
        try:
            # Convert BGR to RGB if needed (OpenCV uses BGR)
            if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                # Assume BGR from OpenCV, convert to RGB
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_crop
            
            # Get face encodings
            encodings = face_recognition.face_encodings(
                face_rgb,
                num_jitters=num_jitters,
                model=self.model
            )
            
            if len(encodings) == 0:
                logger.debug("No face encoding found in crop")
                return None
            
            # Return first (most prominent) face embedding
            return encodings[0]
            
        except Exception as e:
            logger.debug(f"Face embedding extraction error: {e}")
            return None
    
    def extract_embeddings_batch(self, face_crops: List[np.ndarray], num_jitters: int = 1) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings from multiple face crops.
        
        Args:
            face_crops: List of face images
            num_jitters: Number of re-samplings per face
        
        Returns:
            List of 128-dim arrays (or None for failed extractions)
        """
        embeddings = []
        
        for face_crop in face_crops:
            embedding = self.extract_embedding(face_crop, num_jitters=num_jitters)
            embeddings.append(embedding)
        
        return embeddings
    
    @staticmethod
    def compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.6) -> Tuple[float, bool]:
        """
        Compare two face embeddings using Euclidean distance.
        
        Args:
            embedding1: First embedding (128-dim)
            embedding2: Second embedding (128-dim)
            threshold: Distance threshold (lower = stricter matching)
        
        Returns:
            (distance, is_match)
            - distance: Euclidean distance between embeddings
            - is_match: True if distance < threshold (same person)
        """
        if embedding1 is None or embedding2 is None:
            return (float('inf'), False)
        
        try:
            # Calculate Euclidean distance
            distance = np.linalg.norm(embedding1 - embedding2)
            
            # Match if distance is below threshold
            is_match = distance < threshold
            
            return (float(distance), is_match)
            
        except Exception as e:
            logger.error(f"Embedding comparison error: {e}")
            return (float('inf'), False)
    
    @staticmethod
    def find_best_match(query_embedding: np.ndarray, 
                       known_embeddings: List[np.ndarray],
                       threshold: float = 0.6) -> Tuple[Optional[int], float]:
        """
        Find the best matching embedding from a list of known embeddings.
        
        Args:
            query_embedding: Embedding to match
            known_embeddings: List of known embeddings
            threshold: Distance threshold for matching
        
        Returns:
            (best_match_index, distance) or (None, inf) if no match
        """
        if query_embedding is None or not known_embeddings:
            return (None, float('inf'))
        
        best_idx = None
        best_distance = float('inf')
        
        for idx, known_emb in enumerate(known_embeddings):
            if known_emb is None:
                continue
            
            distance, is_match = FaceEmbedder.compare_embeddings(
                query_embedding, known_emb, threshold
            )
            
            if is_match and distance < best_distance:
                best_distance = distance
                best_idx = idx
        
        return (best_idx, best_distance)
    
    @staticmethod
    def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        try:
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Clamp to [0, 1]
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0
