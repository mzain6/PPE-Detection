"""
export_to_openvino.py
=====================
Run this script ONCE to convert all active ONNX models to OpenVINO IR format.
OpenVINO is optimised specifically for Intel CPUs (your i5-1135G7) and gives
2-4x faster inference than ONNX Runtime, potentially reaching 20-30 FPS.

Prerequisites:
    pip install openvino

Usage:
    python scripts/export_to_openvino.py

After running, the system will automatically prefer OpenVINO > ONNX > .pt.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)

# Only the models actively used by the detection pipeline
MODELS_TO_EXPORT = [
    os.path.join(BASE_DIR, "models", "custom", "helmet.pt"),
    os.path.join(BASE_DIR, "models", "custom", "yolov8_vest_small.pt"),
    os.path.join(BASE_DIR, "models", "custom", "best1.pt"),
    os.path.join(BASE_DIR, "models", "custom", "yolov8n.pt"),
    os.path.join(BASE_DIR, "models", "custom", "yolov8n-pose.pt"),
    os.path.join(BASE_DIR, "helmet.pt"),
    os.path.join(BASE_DIR, "yolov8_vest_small.pt"),
    os.path.join(BASE_DIR, "yolov8n-pose.pt"),
    os.path.join(BASE_DIR, "yolov8n.pt"),
]

# imgsz=320 halves inference time with minimal accuracy drop
EXPORT_KWARGS = dict(
    format="openvino",
    imgsz=320,
    half=False,   # FP32 for stability; set True if you want FP16 (faster, less accurate)
)


def openvino_dir_for(pt_path: str) -> str:
    """Return the expected OpenVINO output directory path for a given .pt file."""
    stem = os.path.splitext(os.path.basename(pt_path))[0]
    return os.path.join(os.path.dirname(pt_path), f"{stem}_openvino_model")


def export_model(pt_path: str) -> bool:
    if not os.path.isfile(pt_path):
        logger.debug("Skipping (not found): %s", pt_path)
        return False

    ov_dir = openvino_dir_for(pt_path)
    if os.path.isdir(ov_dir):
        logger.info("Already exported, skipping: %s", ov_dir)
        return True

    try:
        from ultralytics import YOLO
        logger.info("Exporting to OpenVINO (imgsz=320): %s", pt_path)
        model = YOLO(pt_path)
        model.export(**EXPORT_KWARGS)
        if os.path.isdir(ov_dir):
            logger.info("  [OK] Done -> %s", ov_dir)
            return True
        else:
            logger.error("  [FAIL] OpenVINO dir not created for: %s", pt_path)
            return False
    except Exception as e:
        logger.error("  [FAIL] Export failed for %s: %s", pt_path, e)
        return False


def main():
    logger.info("=" * 60)
    logger.info("  SafeSite AI - OpenVINO Export (Intel CPU Optimized)")
    logger.info("=" * 60)

    try:
        import openvino
        logger.info("openvino version: %s", openvino.__version__)
    except ImportError:
        logger.error("openvino is not installed.")
        logger.error("Run: pip install openvino")
        sys.exit(1)

    try:
        import ultralytics
        logger.info("ultralytics version: %s", ultralytics.__version__)
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    success = skipped = failed = 0
    for pt_path in MODELS_TO_EXPORT:
        if not os.path.isfile(pt_path):
            skipped += 1
            continue
        if export_model(pt_path):
            success += 1
        else:
            failed += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Export complete: %d done, %d not found, %d failed",
                success, skipped, failed)
    logger.info("=" * 60)

    if failed > 0:
        logger.warning("Some models failed. System will fall back to ONNX or .pt for those.")
    else:
        logger.info("All models ready. Restart the backend - it will now use OpenVINO.")


if __name__ == "__main__":
    main()
