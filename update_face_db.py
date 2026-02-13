
import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# Configuration
PICTURES_DIR = "pictures"
FACE_DB_PATH = "face_database.pkl"
# Map filename prefixes to Person IDs and Names
PERSON_MAPPING = {
    "person1": {"id": "01", "name": "Person 1"},
    "person2": {"id": "02", "name": "Person 2"},
    "person3": {"id": "03", "name": "Person 3"}, 
}

def create_face_database():
    # Initialize InsightFace
    print("Initializing InsightFace...")
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    face_db = {}
    
    # Process each image in pictures folder
    if not os.path.exists(PICTURES_DIR):
        print(f"Error: {PICTURES_DIR} not found")
        return

    print(f"Processing images in {PICTURES_DIR}...")
    
    for filename in os.listdir(PICTURES_DIR):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        # Determine person ID from filename
        person_key = None
        for key in PERSON_MAPPING:
            if filename.startswith(key):
                person_key = key
                break
        
        if not person_key:
            print(f"Skipping {filename} - unknown person prefix")
            continue
            
        person_info = PERSON_MAPPING[person_key]
        person_id = person_info['id']
        person_name = person_info['name']
        
        # Read image
        img_path = os.path.join(PICTURES_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {filename}")
            continue
            
        # Detect faces
        faces = app.get(img)
        
        if len(faces) == 0:
            print(f"No face detected in {filename}")
            continue
            
        # Use the largest face if multiple found
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        embedding = face.embedding
        
        # Add to database
        if person_id not in face_db:
            face_db[person_id] = {
                'name': person_name,
                'embeddings': []
            }
        
        face_db[person_id]['embeddings'].append(embedding)
        print(f"Processed {filename} -> {person_name} (ID: {person_id})")
        
    # Save database
    print(f"Saving database with {len(face_db)} persons...")
    with open(FACE_DB_PATH, 'wb') as f:
        pickle.dump(face_db, f)
    print(f"Database saved to {FACE_DB_PATH}")

if __name__ == "__main__":
    create_face_database()
