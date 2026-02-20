"""
PPE Detection Manager
─────────────────────
• GREEN box    = Person (tracked, stable ID)
• BLUE box     = Safety Vest detected on this person
• RED box      = No Vest / missing vest on this person
• YELLOW box   = Vest status still being checked (first few frames)

Two boxes are ALWAYS drawn separately — person box never merges with vest box.
Vest status persists once determined and does NOT reset until person leaves frame.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch

# ─── Config ───────────────────────────────────────────────────────────────────
PERSON_MODEL_PATH = "yolov8m.pt"
VEST_MODEL_PATH   = r"C:\Users\Mzain\OneDrive\Desktop\PPE-Detection\yolov8_vest_small.pt"
VIDEO_PATH        = r"C:\Users\Mzain\OneDrive\Desktop\PPE-Detection\test_final.mp4"

PERSON_CONF    = 0.25
VEST_CONF      = 0.1   # Lowered — catches vest even when partially visible
INFER_SIZE     = 1280
MAX_LOST       = 30       # frames to keep track alive after losing detection
IOU_MATCH      = 0.30     # IOU threshold for matching detections to tracks
NMS_THRESH     = 0.45     # NMS to remove duplicate person boxes

# Vest model class IDs
VEST_POSITIVE_IDS = {2}            # class 2 = 'vest'  → wearing it
VEST_NEGATIVE_IDS = {}             # no dedicated 'no_vest' class in this model
# Skip: helmet(0), gloves(1), boots(3), goggles(4), none(5), Person(6),
#        no_helmet(7), no_goggle(8), no_gloves(9), no_boots(10)
# Only vest (2) passes through
VEST_SKIP_IDS  = {0, 1, 3, 4, 5, 6, 7, 8, 9, 10}

# Colors  (BGR)
COLOR_PERSON    = (0, 220, 0)      # Green  — person box
COLOR_VEST_OK   = (0, 200, 0)      # Green  — vest confirmed
COLOR_VEST_NO   = (0, 0, 220)      # Red    — no vest
COLOR_VEST_UNK  = (0, 200, 255)    # Yellow — still checking
# ──────────────────────────────────────────────────────────────────────────────


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua


class PersonTrack:
    """Tracks a single person + their vest status."""
    _nxt = 1

    def __init__(self, box):
        self.id          = PersonTrack._nxt; PersonTrack._nxt += 1
        self.box         = box          # person bounding box [x1,y1,x2,y2]
        self.vest_box    = None         # last known vest box [x1,y1,x2,y2]
        self.vest_status = "Checking"   # "Checking" | "Vest" | "No Vest"
        self.lost        = 0

    def update_person(self, box):
        self.box  = box
        self.lost = 0

    def update_vest(self, vest_box, is_positive: bool):
        self.vest_box = vest_box
        # Once confirmed VEST — never downgrade back.
        # Once confirmed NO VEST — update only if vest is later found.
        if is_positive:
            self.vest_status = "Vest"
        elif self.vest_status != "Vest":
            self.vest_status = "No Vest"


class PersonTracker:
    def __init__(self):
        self.tracks: list[PersonTrack] = []

    def step(self, new_boxes):
        """Match new_boxes (list of [x1,y1,x2,y2]) to existing tracks."""
        used = set()
        unmatched = list(range(len(new_boxes)))

        for det_i in list(unmatched):
            best_s, best_t = 0.0, None
            for t in self.tracks:
                if t.id in used: continue
                s = iou(new_boxes[det_i], t.box)
                if s > best_s: best_s, best_t = s, t
            if best_s >= IOU_MATCH and best_t:
                best_t.update_person(new_boxes[det_i])
                used.add(best_t.id)
                unmatched.remove(det_i)

        for t in self.tracks:
            if t.id not in used:
                t.lost += 1

        self.tracks = [t for t in self.tracks if t.lost <= MAX_LOST]

        for det_i in unmatched:
            self.tracks.append(PersonTrack(new_boxes[det_i]))

        return self.tracks   # all tracks (including coasting ones)


def assign_vest(tracks: list[PersonTrack], vest_results, vest_model):
    """Match vest detections to person tracks and update vest status."""
    for r in vest_results:
        if r.boxes is None: continue
        for b in r.boxes:
            cls_id = int(b.cls[0])
            if cls_id in VEST_SKIP_IDS: continue
            vx1,vy1,vx2,vy2 = b.xyxy[0].cpu().numpy().astype(int)
            vbox       = [vx1, vy1, vx2, vy2]
            is_pos     = cls_id in VEST_POSITIVE_IDS

            # Find person track with highest overlap
            best_s, best_t = 0.0, None
            for t in tracks:
                s = iou(vbox, t.box)
                if s > best_s: best_s, best_t = s, t

            if best_t and best_s > 0.05:
                best_t.update_vest(vbox, is_pos)


def draw_label(img, text, x1, y1, color, font_scale=0.55):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(img, (x1, y1-20), (x1+tw+6, y1), color, -1)
    cv2.putText(img, text, (x1+3, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1)


def render(frame, tracks):
    for t in tracks:
        px1, py1, px2, py2 = t.box

        # ── Person Box (always green) ─────────────────────────────────────────
        cv2.rectangle(frame, (px1,py1), (px2,py2), COLOR_PERSON, 2)
        draw_label(frame, f"Person ID:{t.id}", px1, py1, COLOR_PERSON)

        # ── Vest Status Box (separate, always shown) ──────────────────────────
        if t.vest_status == "Vest":
            vest_color = COLOR_VEST_OK
        elif t.vest_status == "No Vest":
            vest_color = COLOR_VEST_NO
        else:
            vest_color = COLOR_VEST_UNK

        if t.vest_box:
            vx1, vy1, vx2, vy2 = t.vest_box
        else:
            # No vest detection yet — draw a small box at chest level of person
            mid_x = (px1+px2)//2
            h     = py2-py1
            vx1, vy1 = mid_x-30, py1 + h//3
            vx2, vy2 = mid_x+30, py1 + 2*h//3

        cv2.rectangle(frame, (vx1,vy1), (vx2,vy2), vest_color, 2)
        draw_label(frame, t.vest_status, vx1, vy2, vest_color, font_scale=0.48)

    return frame


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    person_model = YOLO(PERSON_MODEL_PATH); person_model.to(device)
    vest_model   = YOLO(VEST_MODEL_PATH);   vest_model.to(device)
    print(f"Vest classes: {vest_model.names}")

    tracker = PersonTracker()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open: {VIDEO_PATH}"); return

    print("Running PPE Detection Manager... Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: print("End of video."); break

        # ── Person Detection ──────────────────────────────────────────────────
        p_res = person_model.predict(frame, classes=[0], conf=PERSON_CONF,
                                     imgsz=INFER_SIZE, verbose=False)
        raw_boxes, raw_scores = [], []
        for r in p_res:
            if r.boxes is None: continue
            for b in r.boxes:
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
                raw_boxes.append([x1,y1,x2-x1,y2-y1])
                raw_scores.append(float(b.conf[0]))

        # NMS to eliminate duplicates
        person_boxes = []
        if raw_boxes:
            keep = cv2.dnn.NMSBoxes(raw_boxes, raw_scores, PERSON_CONF, NMS_THRESH)
            for i in (keep.flatten() if len(keep) else []):
                x,y,w,h = raw_boxes[i]
                person_boxes.append([x, y, x+w, y+h])

        tracks = tracker.step(person_boxes)

        # ── Vest Detection ─────────────────────────────────────────────────────
        v_res = vest_model.predict(frame, conf=VEST_CONF,
                                   imgsz=INFER_SIZE, verbose=False)
        assign_vest(tracks, v_res, vest_model)

        # ── Draw ───────────────────────────────────────────────────────────────
        out = render(frame.copy(), tracks)
        out = cv2.resize(out, (1280, 720))
        cv2.imshow("PPE Detection Manager", out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
