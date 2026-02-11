from ultralytics import YOLO

try:
    model = YOLO(r"C:\Users\Mzain\OneDrive\Desktop\Syed-PPE\helmet.pt")
    print("--- Helmet Model Classes ---")
    for k, v in model.names.items():
        print(f"{k}: {v}")
    print("----------------------------")
except Exception as e:
    print(f"Error loading model: {e}")
