import cv2
import torch
from ultralytics import YOLO

def initialize_model(model_name: str = 'yolov8x.pt'):
    """
    Load the YOLOv8 model and assign it to the optimal device (CUDA if available).
    """
    model = YOLO(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"🤖 Model loaded successfully. Using compute device: {device.upper()}")
    return model

def process_video(video_path: str, model: YOLO, output_path: str = "output_detection.mp4"):
    """
    Read the video frame by frame, run inference, filter target classes, visualize, and save the output.
    """
    # Open the video source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at {video_path}")
        return

    # Get video properties for saving
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize VideoWriter
    # mp4v codec is commonly used for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # COCO class mapping for specific vehicles
    vehicle_classes = {
        2: "Car",
        3: "Bike",
        5: "Bus",
        7: "Truck"
    }
    
    # We can pass the target class indices directly to the YOLO tracking/inference
    target_class_ids = list(vehicle_classes.keys())
    
    print(f"🎬 Starting video processing. Saving to '{output_path}'. Press 'q' to stop.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ End of video stream or cannot fetch the next frame.")
            break

        # Run inference
        # Filter for selected classes [2, 3, 5, 7] and confidence >= 0.5
        results = model(frame, classes=target_class_ids, conf=0.5, verbose=False)

        # Process the detections for the current frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extraction of bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Extraction of confidence score and class ID
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if cls_id in vehicle_classes:
                    class_name = vehicle_classes[cls_id]
                    
                    # 1. Draw the bounding box (Green)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 2. Prepare text label
                    label = f"{class_name}: {conf:.2f}"
                    
                    # 3. Draw background rectangle for the text to improve legibility
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Background rectangle coordinates
                    bg_x2 = x1 + text_width
                    bg_y1 = max(0, y1 - text_height - 10)
                    cv2.rectangle(frame, (x1, bg_y1), (bg_x2, y1), (0, 255, 0), -1)
                    
                    # 4. Put text label (Black text over Green background)
                    text_y = max(text_height, y1 - 5)
                    cv2.putText(frame, label, (x1, text_y), font, font_scale, (0, 0, 0), thickness)

        # Display the output frame
        cv2.imshow("Vehicle Detection Test", frame)
        
        # Write the processed frame to the output video
        out.write(frame)

        # Terminate loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exit signal received. Terminating...")
            break

    # Gracefully release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"💾 Video successfully saved at: {output_path}")

def main():
    # 1. Model Initialization
    model = initialize_model('yolov8x.pt')
    
    # 2. Video Path Selection
    video_source = r"C:\Users\hashi\Downloads\PPE PHASE 3\PPE-Detection\Car Detection 3.mp4"
    output_video = r"C:\Users\hashi\Downloads\PPE PHASE 3\PPE-Detection\Detection_Output.mp4"
    
    # 3. Start Processing
    process_video(video_source, model, output_video)

if __name__ == "__main__":
    main()
