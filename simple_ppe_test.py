"""
PPE Detection with IMPROVED Face ID Stability
- Lower threshold for better matching
- Temporal smoothing - requires multiple frames to change ID
- Embedding averaging for stability
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
from collections import defaultdict

print("Loading YOLO models...")
person_model = YOLO("yolov8n.pt")
face_model = YOLO("yolov8m-face-lindevs.pt")
ppe_model = YOLO("best.pt")
print("✓ YOLO models loaded!")

# Load FaceNet
try:
    from facenet_pytorch import InceptionResnetV1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_encoder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    FACENET_OK = True
    print(f"✓ FaceNet loaded on {device}")
except Exception as e:
    FACENET_OK = False
    face_encoder = None
    print(f"⚠ FaceNet not available: {e}")

class StableFaceTracker:
    def __init__(self):
        # Face database
        self.known_faces = {}  # id -> {"emb": averaged embedding, "count": times seen}
        self.next_id = 1
        
        # TUNING PARAMETERS
        self.match_threshold = 0.55  # LOWER = more lenient matching (was 0.7)
        self.new_face_threshold = 0.45  # Must be very different to be new face
        
        # Temporal smoothing
        self.person_current_id = {}  # track_key -> current stable ID
        self.person_candidates = defaultdict(list)  # track_key -> list of recent candidate IDs
        self.stability_frames = 5  # Need 5 consistent frames to change ID
        
        # Position tracking
        self.positions = {}  # (cx,cy) -> (id, timestamp)
        
    def get_id(self, face_img, bbox, timestamp):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)//2, (y1+y2)//2
        track_key = f"{cx//50}_{cy//50}"  # Grid-based tracking key
        
        candidate_id = None
        
        # Try face matching first
        if FACENET_OK and face_img is not None and face_img.size > 500:
            emb = self._get_embedding(face_img)
            if emb is not None:
                candidate_id = self._match_face(emb)
        
        # Fallback to position if no face match
        if candidate_id is None:
            candidate_id = self._from_position(cx, cy, timestamp)
        
        # If still no match, assign new ID
        if candidate_id is None:
            candidate_id = self.next_id
            self.next_id += 1
            if FACENET_OK and face_img is not None and face_img.size > 500:
                emb = self._get_embedding(face_img)
                if emb is not None:
                    self.known_faces[candidate_id] = {"emb": emb, "count": 1}
                    print(f"[NEW FACE] ID {candidate_id}")
        
        # TEMPORAL SMOOTHING - don't change ID immediately
        stable_id = self._apply_temporal_smoothing(track_key, candidate_id)
        
        # Update position
        self._update_position(cx, cy, stable_id, timestamp)
        
        return stable_id
    
    def _get_embedding(self, face_img):
        try:
            if face_img.shape[0] < 30 or face_img.shape[1] < 30:
                return None
            
            face_resized = cv2.resize(face_img, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
            face_tensor = (face_tensor - 127.5) / 128.0
            face_tensor = face_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = face_encoder(face_tensor)
            
            return emb.cpu().numpy().flatten()
        except:
            return None
    
    def _match_face(self, emb):
        best_id = None
        best_sim = self.match_threshold
        
        for fid, data in self.known_faces.items():
            stored_emb = data["emb"]
            # Cosine similarity
            sim = np.dot(emb, stored_emb) / (np.linalg.norm(emb) * np.linalg.norm(stored_emb) + 1e-8)
            
            if sim > best_sim:
                best_sim = sim
                best_id = fid
        
        if best_id is not None:
            # Update embedding with running average for stability
            old_emb = self.known_faces[best_id]["emb"]
            count = self.known_faces[best_id]["count"]
            # Weighted average - gives more weight to established embedding
            alpha = min(0.1, 1.0 / (count + 1))
            new_emb = (1 - alpha) * old_emb + alpha * emb
            self.known_faces[best_id]["emb"] = new_emb / (np.linalg.norm(new_emb) + 1e-8)  # Normalize
            self.known_faces[best_id]["count"] = count + 1
        
        return best_id
    
    def _apply_temporal_smoothing(self, track_key, candidate_id):
        """Require consistent ID for several frames before changing"""
        
        # Get current stable ID
        current_id = self.person_current_id.get(track_key)
        
        if current_id is None:
            # First time seeing this position - accept candidate
            self.person_current_id[track_key] = candidate_id
            self.person_candidates[track_key] = [candidate_id]
            return candidate_id
        
        if candidate_id == current_id:
            # Same as current - reset candidates, stay stable
            self.person_candidates[track_key] = [candidate_id]
            return current_id
        
        # Different ID - add to candidates
        self.person_candidates[track_key].append(candidate_id)
        
        # Keep only recent candidates
        if len(self.person_candidates[track_key]) > self.stability_frames * 2:
            self.person_candidates[track_key] = self.person_candidates[track_key][-self.stability_frames * 2:]
        
        # Count how many recent frames have this candidate
        recent = self.person_candidates[track_key][-self.stability_frames:]
        if len(recent) >= self.stability_frames and all(c == candidate_id for c in recent):
            # Consistent for N frames - switch ID
            self.person_current_id[track_key] = candidate_id
            self.person_candidates[track_key] = [candidate_id]
            print(f"[ID SWITCH] {current_id} -> {candidate_id}")
            return candidate_id
        
        # Not stable enough - keep current ID
        return current_id
    
    def _from_position(self, cx, cy, timestamp):
        best_id = None
        best_dist = 100
        
        to_del = []
        for (px, py), (pid, ts) in self.positions.items():
            if timestamp - ts > 3:  # 3 sec timeout
                to_del.append((px, py))
                continue
            dist = ((cx-px)**2 + (cy-py)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_id = pid
        
        for k in to_del:
            del self.positions[k]
        
        return best_id
    
    def _update_position(self, cx, cy, pid, ts):
        # Remove old positions for this ID
        to_del = [k for k, (fid, _) in self.positions.items() if fid == pid]
        for k in to_del:
            del self.positions[k]
        self.positions[(cx, cy)] = (pid, ts)

tracker = StableFaceTracker()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    exit(1)

print("\n✓ Ready! Press 'q' to quit")
print("ID changes require 5 consistent frames for stability\n")

frame_n = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    ts = time.time()
    frame_n += 1
    
    # Face detection every 3 frames for better tracking
    do_face = (frame_n % 3 == 0) or (frame_n < 20)
    
    # 1. Detect persons
    res = person_model(frame, verbose=False, conf=0.3)
    
    persons = []
    for box in res[0].boxes:
        name = res[0].names.get(int(box.cls.cpu().numpy()[0]), "")
        if name == "person":
            x1, y1, x2, y2 = [int(v) for v in box.xyxy.cpu().numpy()[0]]
            persons.append({
                "bbox": (x1, y1, x2, y2),
                "helmet": False, "vest": False,
                "hbox": None, "vbox": None, "id": None
            })
    
    # 2. Face detection + Stable ID
    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        
        face_img = None
        if do_face and crop.size > 0:
            fres = face_model(crop, verbose=False, conf=0.35)
            for fb in fres[0].boxes:
                fx1, fy1, fx2, fy2 = [int(v) for v in fb.xyxy.cpu().numpy()[0]]
                fx1, fy1 = max(0, fx1), max(0, fy1)
                fx2, fy2 = min(crop.shape[1], fx2), min(crop.shape[0], fy2)
                face_img = crop[fy1:fy2, fx1:fx2].copy()
                break
        
        p["id"] = tracker.get_id(face_img, p["bbox"], ts)
    
    # 3. PPE detection
    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size > 0:
            pres = ppe_model(crop, verbose=False, conf=0.3)
            for pb in pres[0].boxes:
                name = pres[0].names.get(int(pb.cls.cpu().numpy()[0]), "").lower()
                px1, py1, px2, py2 = [int(v) for v in pb.xyxy.cpu().numpy()[0]]
                fbox = (x1+px1, y1+py1, x1+px2, y1+py2)
                if "helmet" in name or "head" in name:
                    p["helmet"] = True
                    p["hbox"] = fbox
                elif "vest" in name:
                    p["vest"] = True
                    p["vbox"] = fbox
    
    # 4. DRAW
    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        pid = p["id"]
        ph = y2 - y1
        pw = x2 - x1
        
        # Person box is always GREEN when person detected with ID
        col = (0, 255, 0)  # Green for detected person
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
        
        id_txt = f"ID {pid}" if pid else "ID ?"
        (tw, th), _ = cv2.getTextSize(id_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (x1, y1), (x1 + tw + 20, y1 + th + 20), col, -1)
        cv2.putText(frame, id_txt, (x1 + 10, y1 + th + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if p["helmet"] and p["hbox"]:
            hx1, hy1, hx2, hy2 = p["hbox"]
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
        else:
            hx1, hy1 = x1 + pw//5, y1 + int(ph*0.05)
            hx2, hy2 = x2 - pw//5, y1 + int(ph*0.25)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
            cv2.putText(frame, "NO HELMET", (hx1, hy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if p["vest"] and p["vbox"]:
            vx1, vy1, vx2, vy2 = p["vbox"]
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
        else:
            vx1, vy1 = x1 + pw//8, y1 + int(ph*0.55)
            vx2, vy2 = x2 - pw//8, y1 + int(ph*0.90)
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
            cv2.putText(frame, "NO VEST", (vx1, vy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    info = f"Persons: {len(persons)} | Known: {len(tracker.known_faces)} | 'q' to quit"
    cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow("PPE Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")
