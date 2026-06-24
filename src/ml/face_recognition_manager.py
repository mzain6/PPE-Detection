"""
Face Recognition Manager
Handles face detection, embedding extraction, and face matching against authorized database
"""

import cv2
import numpy as np
import pickle
import os
from insightface.app import FaceAnalysis

class FaceRecognitionManager:
    def __init__(self, face_db_path='face_database.pkl', similarity_threshold=0.6):
        """
        Initialize Face Recognition Manager
        
        Args:
            face_db_path: Path to the face database file
            similarity_threshold: Threshold for face matching (0.6 = 60% similarity)
        """
        self.similarity_threshold = similarity_threshold
        self.face_db_path = face_db_path
        
        # Initialize InsightFace model with GPU support
        print("[FaceRecognition] Initializing InsightFace model with GPU...")
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load authorized face database
        self.authorized_faces = {}
        self.load_face_database()
        
    def load_face_database(self):
        """Load authorized faces from pickle file"""
        if not os.path.exists(self.face_db_path):
            print(f"[FaceRecognition] WARNING: Face database not found at {self.face_db_path}")
            print("[FaceRecognition] Please run setup_face_database.py first!")
            self.authorized_faces = {}
            return
            
        try:
            with open(self.face_db_path, 'rb') as f:
                self.authorized_faces = pickle.load(f)
            print(f"[FaceRecognition] Loaded {len(self.authorized_faces)} authorized persons")
            for person_id, data in self.authorized_faces.items():
                print(f"  - {person_id}: {data['name']} ({len(data['embeddings'])} face samples)")
        except Exception as e:
            print(f"[FaceRecognition] ERROR loading face database: {e}")
            self.authorized_faces = {}
    
    def detect_faces(self, frame):
        """
        Detect faces in frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of face objects with embeddings and bounding boxes
        """
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.app.get(img_rgb)
            
            return faces
        except Exception as e:
            print(f"[FaceRecognition] Error detecting faces: {e}")
            return []
    
    def recognize_face(self, face_embedding):
        """
        Match face embedding against authorized database
        
        Args:
            face_embedding: Face embedding vector from InsightFace
            
        Returns:
            tuple: (person_id, similarity_score) or (None, 0) if no match
        """
        if not self.authorized_faces:
            return None, 0
        
        best_match_id = None
        best_similarity = 0
        
        # Compare with all authorized faces
        for person_id, person_data in self.authorized_faces.items():
            for stored_embedding in person_data['embeddings']:
                # Calculate cosine similarity
                similarity = self.cosine_similarity(face_embedding, stored_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = person_id
        
        # Check if best match exceeds threshold
        if best_similarity >= self.similarity_threshold:
            return best_match_id, best_similarity
        else:
            return None, best_similarity
    
    def cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        # Normalize embeddings
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        return similarity
    
    def process_frame_for_faces(self, frame):
        """
        Process frame to detect and recognize faces
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of dicts with face information:
            [{
                'person_id': '01' or None,
                'bbox': (x1, y1, x2, y2),
                'similarity': 0.85,
                'authorized': True/False
            }, ...]
        """
        results = []
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        for face in faces:
            # Get bounding box
            bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
            
            # Get embedding
            embedding = face.embedding
            
            # Recognize face
            person_id, similarity = self.recognize_face(embedding)
            
            result = {
                'person_id': person_id,
                'bbox': tuple(bbox),
                'similarity': similarity,
                'authorized': person_id is not None
            }
            
            results.append(result)
        
        return results
    
    def draw_face_results(self, frame, face_results):
        """
        Draw face detection results on frame
        
        Args:
            frame: Input image to draw on
            face_results: List of face result dicts from process_frame_for_faces
            
        Returns:
            Modified frame with drawings
        """
        for result in face_results:
            x1, y1, x2, y2 = result['bbox']
            
            # Color: Green for authorized, Red for unauthorized
            color = (0, 255, 0) if result['authorized'] else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            if result['authorized']:
                label = f"Person {result['person_id']} ({result['similarity']:.2f})"
            else:
                label = f"Unknown ({result['similarity']:.2f})"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame


if __name__ == "__main__":
    # Test the face recognition manager
    print("Testing Face Recognition Manager...")
    
    mgr = FaceRecognitionManager()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open webcam for testing")
        exit()
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        face_results = mgr.process_frame_for_faces(frame)
        
        # Draw results
        frame = mgr.draw_face_results(frame, face_results)
        
        # Show
        cv2.imshow("Face Recognition Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
