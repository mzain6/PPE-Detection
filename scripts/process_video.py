"""
PPE Detection on Video File - Construction Safety Video
Outputs annotated video with person IDs, helmet, and vest detection
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
from collections import defaultdict
import os

# Video paths
INPUT_VIDEO = r"C:\Users\ST\Desktop\PPE_Phas2_05Feb\Phase_Detection_05Feb\PPE-Detection-2\Construction Safety at Mortenson_ Hard Hats to Helmets on the Jobsite.mp4"
OUTPUT_VIDEO = r"C:\Users\ST\Desktop\PPE_Phas2_05Feb\Phase_Detection_05Feb\PPE-Detection-2\ppe_detection_output.mp4"

print("=" * 60)
print("PPE Detection on Video")
print("=" * 60)

# Check input exists
if not os.path.exists(INPUT_VIDEO):
    print(f"ERROR: Input video not found: {INPUT_VIDEO}")
    exit(1)

print(f"\nInput: {INPUT_VIDEO}")
print(f"Output: {OUTPUT_VIDEO}")

print("\nLoading models...")
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
        self.known_faces = {}
        self.next_id = 1
        self.match_threshold = 0.55
        self.person_current_id = {}
        self.person_candidates = defaultdict(list)
        self.stability_frames = 5
        self.positions = {}
        
    def get_id(self, face_img, bbox, timestamp):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)//2, (y1+y2)//2
        track_key = f"{cx//50}_{cy//50}"
        
        candidate_id = None
        
        if FACENET_OK and face_img is not None and face_img.size > 500:
            emb = self._get_embedding(face_img)
            if emb is not None:
                candidate_id = self._match_face(emb)
        
        if candidate_id is None:
            candidate_id = self._from_position(cx, cy, timestamp)
        
        if candidate_id is None:
            candidate_id = self.next_id
            self.next_id += 1
            if FACENET_OK and face_img is not None and face_img.size > 500:
                emb = self._get_embedding(face_img)
                if emb is not None:
                    self.known_faces[candidate_id] = {"emb": emb, "count": 1}
        
        stable_id = self._apply_temporal_smoothing(track_key, candidate_id)
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
            sim = np.dot(emb, stored_emb) / (np.linalg.norm(emb) * np.linalg.norm(stored_emb) + 1e-8)
            if sim > best_sim:
                best_sim = sim
                best_id = fid
        if best_id is not None:
            old_emb = self.known_faces[best_id]["emb"]
            count = self.known_faces[best_id]["count"]
            alpha = min(0.1, 1.0 / (count + 1))
            new_emb = (1 - alpha) * old_emb + alpha * emb
            self.known_faces[best_id]["emb"] = new_emb / (np.linalg.norm(new_emb) + 1e-8)
            self.known_faces[best_id]["count"] = count + 1
        return best_id
    
    def _apply_temporal_smoothing(self, track_key, candidate_id):
        current_id = self.person_current_id.get(track_key)
        if current_id is None:
            self.person_current_id[track_key] = candidate_id
            self.person_candidates[track_key] = [candidate_id]
            return candidate_id
        if candidate_id == current_id:
            self.person_candidates[track_key] = [candidate_id]
            return current_id
        self.person_candidates[track_key].append(candidate_id)
        if len(self.person_candidates[track_key]) > self.stability_frames * 2:
            self.person_candidates[track_key] = self.person_candidates[track_key][-self.stability_frames * 2:]
        recent = self.person_candidates[track_key][-self.stability_frames:]
        if len(recent) >= self.stability_frames and all(c == candidate_id for c in recent):
            self.person_current_id[track_key] = candidate_id
            self.person_candidates[track_key] = [candidate_id]
            return candidate_id
        return current_id
    
    def _from_position(self, cx, cy, timestamp):
        best_id = None
        best_dist = 100
        to_del = []
        for (px, py), (pid, ts) in self.positions.items():
            if timestamp - ts > 3:
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
        to_del = [k for k, (fid, _) in self.positions.items() if fid == pid]
        for k in to_del:
            del self.positions[k]
        self.positions[(cx, cy)] = (pid, ts)

tracker = StableFaceTracker()

# Open video
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print("ERROR: Cannot open video!")
    exit(1)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo: {width}x{height} @ {fps}fps, {total_frames} frames")

# Setup output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"\nProcessing video...")

frame_n = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_n += 1
    h, w = frame.shape[:2]
    ts = time.time()
    
    # Progress
    if frame_n % 30 == 0:
        elapsed = time.time() - start_time
        fps_actual = frame_n / elapsed
        eta = (total_frames - frame_n) / fps_actual if fps_actual > 0 else 0
        print(f"  Frame {frame_n}/{total_frames} ({100*frame_n/total_frames:.1f}%) - {fps_actual:.1f} fps - ETA: {eta:.0f}s")
    
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
    
    # 2. Face detection + ID
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
    
    # 4. Draw
    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        pid = p["id"]
        ph = y2 - y1
        pw = x2 - x1
        
        # Person box always green
        col = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
        
        # ID label
        id_txt = f"ID {pid}" if pid else "ID ?"
        (tw, th), _ = cv2.getTextSize(id_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (x1, y1), (x1 + tw + 20, y1 + th + 20), col, -1)
        cv2.putText(frame, id_txt, (x1 + 10, y1 + th + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Helmet
        if p["helmet"] and p["hbox"]:
            hx1, hy1, hx2, hy2 = p["hbox"]
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
        else:
            hx1, hy1 = x1 + pw//5, y1 + int(ph*0.05)
            hx2, hy2 = x2 - pw//5, y1 + int(ph*0.25)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
            cv2.putText(frame, "NO HELMET", (hx1, hy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Vest
        if p["vest"] and p["vbox"]:
            vx1, vy1, vx2, vy2 = p["vbox"]
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
        else:
            vx1, vy1 = x1 + pw//8, y1 + int(ph*0.55)
            vx2, vy2 = x2 - pw//8, y1 + int(ph*0.90)
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
            cv2.putText(frame, "NO VEST", (vx1, vy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Info overlay
    info = f"Persons: {len(persons)} | Known Faces: {len(tracker.known_faces)}"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Write frame
    out.write(frame)

cap.release()
out.release()

elapsed = time.time() - start_time
print(f"\n✓ Processing complete!")
print(f"  Processed {frame_n} frames in {elapsed:.1f}s ({frame_n/elapsed:.1f} fps)")
print(f"  Output saved to: {OUTPUT_VIDEO}")
print(f"  Total unique persons: {len(tracker.known_faces)}")
