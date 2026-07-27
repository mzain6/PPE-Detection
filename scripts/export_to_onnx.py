"""
export_to_onnx.py
=================
Run this script ONCE to convert all active YOLOv8 .pt models to ONNX format.
ONNX models run significantly faster (1.5–3× on CPU, ~2× on GPU) with zero
changes to the detection API or dashboard.

Usage:
    python scripts/export_to_onnx.py

After running, the system will automatically prefer .onnx files over .pt.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Repo root is one level above this script ──────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)

# ── Models to convert (path relative to BASE_DIR) ─────────────────────────────
# Only the models actively used by the detection pipeline are listed here.
# Large unused models (yolov8l.pt, yolov8m.pt, yolov8m-face-lindevs.pt)
# are intentionally skipped to save conversion time.
MODELS_TO_EXPORT = [
    os.path.join(BASE_DIR, "models", "custom", "helmet.pt"),
    os.path.join(BASE_DIR, "models", "custom", "yolov8_vest_small.pt"),
    os.path.join(BASE_DIR, "models", "custom", "best1.pt"),
    os.path.join(BASE_DIR, "models", "custom", "yolov8n.pt"),
    os.path.join(BASE_DIR, "models", "custom", "yolov8n-pose.pt"),
    # Also convert pretrained if they exist at project root (used by ppe_stream_worker)
    os.path.join(BASE_DIR, "helmet.pt"),
    os.path.join(BASE_DIR, "yolov8_vest_small.pt"),
    os.path.join(BASE_DIR, "yolov8n-pose.pt"),
    os.path.join(BASE_DIR, "yolov8n.pt"),
]

EXPORT_KWARGS = dict(
    format="onnx",
    imgsz=640,
    optimize=True,
    simplify=True,
    dynamic=False,   # static batch=1, faster for real-time single-frame inference
)


def export_model(pt_path: str) -> bool:
    """
    Export a single .pt model to ONNX.
    Returns True on success, False if skipped or failed.
    """
    if not os.path.isfile(pt_path):
        logger.debug("Skipping (not found): %s", pt_path)
        return False

    onnx_path = pt_path.replace(".pt", ".onnx")

    if os.path.isfile(onnx_path):
        logger.info("Already exported, skipping: %s", onnx_path)
        return True

    try:
        from ultralytics import YOLO  # imported here so script fails fast if not installed
        logger.info("Exporting: %s", pt_path)
        model = YOLO(pt_path)
        model.export(**EXPORT_KWARGS)
        if os.path.isfile(onnx_path):
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            logger.info("  ✅ Done → %s (%.1f MB)", onnx_path, size_mb)
            return True
        else:
            logger.error("  ❌ Export finished but .onnx file not found for: %s", pt_path)
            return False
    except Exception as e:
        logger.error("  ❌ Failed to export %s: %s", pt_path, e)
        return False


def main():
    logger.info("=" * 60)
    logger.info("  Safety Equipment Detection — ONNX Export Script")
    logger.info("=" * 60)

    try:
        import ultralytics
        logger.info("ultralytics version: %s", ultralytics.__version__)
    except ImportError:
        logger.error("ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    success = 0
    skipped = 0
    failed  = 0

    for pt_path in MODELS_TO_EXPORT:
        result = export_model(pt_path)
        if result is True:
            # Check if it was a skip (already existed) or a real export
            onnx_path = pt_path.replace(".pt", ".onnx")
            success += 1
        elif result is False and not os.path.isfile(pt_path):
            skipped += 1
        else:
            failed += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Export complete: %d converted/ready, %d not found, %d failed",
                success, skipped, failed)
    logger.info("=" * 60)

    if failed > 0:
        logger.warning("Some models failed to export. The system will fall back to .pt for those.")
    else:
        logger.info("All models ready. Start the backend normally — it will now use ONNX.")


if __name__ == "__main__":
    main()
