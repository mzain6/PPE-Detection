"""
Central Tracking Manager
Manages person IDs and tracker handoff across multiple cameras
"""

import threading
import time
import numpy as np
from collections import defaultdict

class CentralTrackingManager:
    def __init__(self):
        """Initialize Central Tracking Manager"""
        self.lock = threading.Lock()
        
        # Person registry: {person_id: person_data}
        self.active_persons = {}
        
        # Tracker to Person mapping: {camera_id: {tracker_id: person_id}}
        self.tracker_to_person = defaultdict(dict)
        
        # Next person ID counter
        self.next_person_id = 1
        
        print("[CentralManager] Initialized")
    
    def register_person(self, person_id, tracker_id, camera_id, face_embedding=None, bbox=None):
        """
        Register a new person or update existing person
        
        Args:
            person_id: Person ID (e.g., "01")
            tracker_id: Object tracker ID
            camera_id: Camera identifier
            face_embedding: Face embedding vector (optional)
            bbox: Bounding box (x1, y1, x2, y2) (optional)
        """
        with self.lock:
            # Create or update person record
            if person_id not in self.active_persons:
                self.active_persons[person_id] = {
                    'person_id': person_id,
                    'authorized': True,
                    'face_embedding': face_embedding,
                    'cameras': {},
                    'last_seen': time.time(),
                    'first_seen': time.time()
                }
                print(f"[CentralManager] Registered new person: {person_id}")
            
            # Update person's current camera and tracker
            self.active_persons[person_id]['cameras'][camera_id] = {
                'tracker_id': tracker_id,
                'bbox': bbox,
                'last_seen': time.time()
            }
            self.active_persons[person_id]['last_seen'] = time.time()
            
            # Map tracker to person
            self.tracker_to_person[camera_id][tracker_id] = person_id
            
            return person_id
    
    def register_unauthorized_person(self, tracker_id, camera_id, bbox=None):
        """
        Register unauthorized person (no face match)
        
        Args:
            tracker_id: Object tracker ID
            camera_id: Camera identifier
            bbox: Bounding box (optional)
            
        Returns:
            Assigned person ID (e.g., "U01" for unauthorized)
        """
        with self.lock:
            # Generate unauthorized person ID
            person_id = f"U{self.next_person_id:02d}"
            self.next_person_id += 1
            
            # Create person record
            self.active_persons[person_id] = {
                'person_id': person_id,
                'authorized': False,
                'face_embedding': None,
                'cameras': {
                    camera_id: {
                        'tracker_id': tracker_id,
                        'bbox': bbox,
                        'last_seen': time.time()
                    }
                },
                'last_seen': time.time(),
                'first_seen': time.time()
            }
            
            # Map tracker to person
            self.tracker_to_person[camera_id][tracker_id] = person_id
            
            print(f"[CentralManager] Registered unauthorized person: {person_id}")
            
            return person_id
    
    def get_person_by_tracker(self, tracker_id, camera_id):
        """
        Get person ID from tracker ID
        
        Args:
            tracker_id: Object tracker ID
            camera_id: Camera identifier
            
        Returns:
            person_id or None if not found
        """
        with self.lock:
            return self.tracker_to_person.get(camera_id, {}).get(tracker_id, None)
    
    def identify_new_tracker(self, tracker_id, camera_id, bbox, appearance_features=None):
        """
        Identify person ID for a new tracker using spatial-temporal reasoning
        
        Args:
            tracker_id: New tracker ID
            camera_id: Camera where tracker appeared
            bbox: Bounding box (x1, y1, x2, y2)
            appearance_features: Visual features (optional)
            
        Returns:
            person_id if match found, None otherwise
        """
        with self.lock:
            # Check if person from another camera could have moved here
            current_time = time.time()
            
            best_match_id = None
            best_match_score = 0
            
            for person_id, person_data in self.active_persons.items():
                # Skip if already in this camera
                if camera_id in person_data['cameras']:
                    continue
                
                # Check temporal continuity (person seen recently in other camera)
                time_since_last_seen = current_time - person_data['last_seen']
                
                # Must be seen within last 60 seconds (increased for back-and-forth movement)
                if time_since_last_seen > 60.0:
                    continue
                
                # Calculate spatial plausibility
                # (In a real system, you'd use camera topology and distance)
                # For now, we assume cameras are adjacent
                
                # Simple heuristic: recent person likely moved to new camera
                # We give a boost to re-identification to ensure IDs stick
                base_score = 0.8  # Start with high confidence for any plausible match
                time_penalty = (time_since_last_seen / 60.0) * 0.5  # Max penalty 0.5
                match_score = base_score - time_penalty
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_id = person_id
            
            # Require at least 20% confidence (very loose to catch everything)
            if best_match_score >= 0.2:
                print(f"[CentralManager] Camera {camera_id}: Matched tracker {tracker_id} to {best_match_id} (score: {best_match_score:.2f})")
                
                # Update person's location
                self.active_persons[best_match_id]['cameras'][camera_id] = {
                    'tracker_id': tracker_id,
                    'bbox': bbox,
                    'last_seen': current_time
                }
                self.active_persons[best_match_id]['last_seen'] = current_time
                
                # Map tracker to person
                self.tracker_to_person[camera_id][tracker_id] = best_match_id
                
                return best_match_id
            
            return None
    
    def update_tracker_bbox(self, tracker_id, camera_id, bbox):
        """
        Update bounding box for existing tracker
        
        Args:
            tracker_id: Tracker ID
            camera_id: Camera identifier
            bbox: New bounding box
        """
        with self.lock:
            person_id = self.tracker_to_person.get(camera_id, {}).get(tracker_id)
            
            if person_id and person_id in self.active_persons:
                if camera_id in self.active_persons[person_id]['cameras']:
                    self.active_persons[person_id]['cameras'][camera_id]['bbox'] = bbox
                    self.active_persons[person_id]['cameras'][camera_id]['last_seen'] = time.time()
                    self.active_persons[person_id]['last_seen'] = time.time()

    def force_get_recent_id(self, exclude_camera_id=None, timeout=60.0):
        """
        Aggressively get the most recently active authorized person ID.
        Used to force-match people on side cameras to the entrance ID.
        """
        with self.lock:
            best_person_id = None
            min_time_diff = float('inf')
            current_time = time.time()
            
            for person_id, data in self.active_persons.items():
                # Skip unauthorized temporary IDs if we want strictly the "Entrance ID"
                if person_id.startswith('U'):
                    continue
                    
                # Check if this person was seen recently anywhere
                time_diff = current_time - data['last_seen']
                
                if time_diff < timeout and time_diff < min_time_diff:
                    # Also, prefer people seen on Entrance (Cam 1)
                    # We can check data['cameras'] keys.
                    is_entrance_person = any(c for c in data['cameras'] if "Cam 1" in c or "Entrance" in c)
                    
                    # If we really want to enforce "from Entrance", we could require is_entrance_person
                    # But for now, just taking the most recent authorized person is a good heuristic
                    min_time_diff = time_diff
                    best_person_id = person_id
            
            if best_person_id:
                print(f"[CentralManager] Force-Found recent person: {best_person_id} (seen {min_time_diff:.1f}s ago)")
                return best_person_id
            return None
    
    def remove_tracker(self, tracker_id, camera_id):
        """
        Remove tracker when person leaves camera view
        
        Args:
            tracker_id: Tracker ID
            camera_id: Camera identifier
        """
        with self.lock:
            person_id = self.tracker_to_person.get(camera_id, {}).get(tracker_id)
            
            if person_id and person_id in self.active_persons:
                # Remove camera from person's record
                if camera_id in self.active_persons[person_id]['cameras']:
                    del self.active_persons[person_id]['cameras'][camera_id]
                    print(f"[CentralManager] Person {person_id} left camera {camera_id}")
                
                # If person not in any camera, mark as left
                if not self.active_persons[person_id]['cameras']:
                    print(f"[CentralManager] Person {person_id} left all cameras")
                    # Could delete or mark as inactive
                    # For now, keep for potential re-entry
            
            # Remove tracker mapping
            if camera_id in self.tracker_to_person:
                if tracker_id in self.tracker_to_person[camera_id]:
                    del self.tracker_to_person[camera_id][tracker_id]
    
    def cleanup_old_persons(self, timeout=30):
        """
        Remove persons not seen in any camera for timeout seconds
        
        Args:
            timeout: Seconds after which to remove inactive persons
        """
        with self.lock:
            current_time = time.time()
            to_remove = []
            
            for person_id, person_data in self.active_persons.items():
                if current_time - person_data['last_seen'] > timeout:
                    to_remove.append(person_id)
            
            for person_id in to_remove:
                print(f"[CentralManager] Removing inactive person: {person_id}")
                del self.active_persons[person_id]
    
    def get_active_persons(self):
        """Get list of all active persons"""
        with self.lock:
            return list(self.active_persons.keys())
    
    def get_person_info(self, person_id):
        """Get full information about a person"""
        with self.lock:
            return self.active_persons.get(person_id, None)
    
    def get_stats(self):
        """Get tracking statistics"""
        with self.lock:
            total_persons = len(self.active_persons)
            authorized = sum(1 for p in self.active_persons.values() if p['authorized'])
            unauthorized = total_persons - authorized
            
            cameras_active = {}
            for person_data in self.active_persons.values():
                for cam_id in person_data['cameras'].keys():
                    cameras_active[cam_id] = cameras_active.get(cam_id, 0) + 1
            
            return {
                'total_persons': total_persons,
                'authorized': authorized,
                'unauthorized': unauthorized,
                'cameras': cameras_active
            }


if __name__ == "__main__":
    # Test the central tracking manager
    print("Testing Central Tracking Manager...")
    
    mgr = CentralTrackingManager()
    
    # Simulate person entering at camera 1
    print("\n1. Person enters at Camera 1")
    mgr.register_person("01", tracker_id=5, camera_id="cam1", bbox=(100, 100, 200, 300))
    
    print(f"Active persons: {mgr.get_active_persons()}")
    print(f"Person by tracker: {mgr.get_person_by_tracker(5, 'cam1')}")
    
    # Simulate person moving to camera 2
    print("\n2. New tracker appears at Camera 2")
    time.sleep(1)  # 1 second delay
    person_id = mgr.identify_new_tracker(tracker_id=12, camera_id="cam2", bbox=(150, 150, 250, 350))
    print(f"Identified as: {person_id}")
    
    # Show stats
    print(f"\nStats: {mgr.get_stats()}")
