from ultralytics import YOLO

with open("model_init_check.txt", "w") as f:
    try:
        f.write("--- Helmet Model (helmet.pt) ---\n")
        model_h = YOLO(r"C:\Users\Mzain\OneDrive\Desktop\Syed-PPE\helmet.pt")
        for k, v in model_h.names.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        
        f.write("--- Vest Model (vest.pt) ---\n")
        model_v = YOLO(r"C:\Users\Mzain\OneDrive\Desktop\Syed-PPE\vest.pt")
        for k, v in model_v.names.items():
            f.write(f"{k}: {v}\n")
    except Exception as e:
        f.write(f"\nError: {e}\n")
