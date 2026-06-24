import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import time

# Load Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize Model
print("Loading YOLOv8 model for Vehicle Detection...")
model = YOLO(config['model']['path'])  # standard yolov8n.pt
model.to('cuda' if config['model']['device'] == 'auto' else 'cpu')

# Vehicle Classes (COCO)
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle", 
    5: "Bus",
    7: "Truck"
}

# Colors for visualization
COLORS = {
    2: (0, 255, 255),  # Yellow for Cars
    3: (255, 100, 0),  # Blue-ish for Bikes
    5: (255, 0, 255),  # Purple for Bus
    7: (0, 165, 255)   # Orange for Truck
}

def process_video(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error opening source: {source}")
        return

    print(f"Starting detection on source: {source}")
    print("Press 'Q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for performance if needed
        # frame = cv2.resize(frame, (1280, 720))

        # Inference
        results = model.predict(frame, conf=0.4, classes=list(VEHICLE_CLASSES.keys()), verbose=False)

        # Draw Detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                label = VEHICLE_CLASSES.get(cls_id, "Unknown")
                color = COLORS.get(cls_id, (0, 255, 0))

                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label
                text = f"{label} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Vehicle Count Display
        count_text = f"Vehicles: {len(boxes)}"
        cv2.putText(frame, count_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Vehicle Detection Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use webcam by default, or change to video file path
    # process_video("Test Video.mp4") 
    process_video(0) 
