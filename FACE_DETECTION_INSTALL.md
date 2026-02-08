# Face Detection Installation Guide

## Prerequisites

Face detection requires the `face_recognition` library which depends on `dlib`.

## Windows Installation

### Option 1: Using Pre-built Wheels (Recommended)

```bash
# Install CMake (required for dlib)
pip install cmake

# Install dlib (prebuilt wheel)
pip install dlib

# Install face_recognition
pip install face-recognition
```

### Option 2: If Option 1 Fails

You may need Visual C++ Build Tools:

1. **Download Visual Studio Build Tools**:
   - Visit: https://visualstudio.microsoft.com/downloads/
   - Download "Build Tools for Visual Studio 2022"
   - Install with "Desktop development with C++" workload

2. **Then install packages**:
```bash
pip install cmake
pip install dlib
pip install face-recognition
```

### Option 3: Using Conda (Alternative)

```bash
conda install -c conda-forge dlib
pip install face-recognition
```

## Verify Installation

```bash
python -c "import face_recognition; print('✅ face_recognition installed successfully')"
```

## If Installation Fails

Face detection will be disabled automatically. The system will fall back to IOU-based tracking (still works, just without face re-identification).

To disable face detection manually, edit `config.yaml`:

```yaml
face_detection:
  enabled: false  # Set to false
```

## Testing Face Detection

```bash
# Run the system
python multi_camera_ppe_demo.py

# Check logs for:
# "Face detection initialized successfully" ✅

# Or if failed:
# "face_recognition library not available" ⚠️
```

## Alternative: Install All Dependencies

```bash
# Install all PPE detection dependencies including face detection
pip install -r requirements.txt
```

## Troubleshooting

**Error: "No module named 'dlib'"**
- Solution: Install dlib first: `pip install dlib`

**Error: "Microsoft Visual C++ 14.0 is required"**
- Solution: Install Visual Studio Build Tools (see Option 2 above)

**Error: "Could not find a version that satisfies the requirement dlib"**
- Solution: Try conda installation (Option 3)

**Face detection slow**
- This is normal - face embedding extraction takes ~100-200ms per person
- Batch processing helps mitigate this

## Performance Impact

With face detection enabled:
- **Processing Time**: +50-150ms per frame (depending on # of people)
- **GPU Memory**: +100-200MB (face detection model)
- **Accuracy**: Persistent IDs across disappearance/reappearance

Without face detection:
- Uses IOU-based tracking (bbox overlap)
- Faster but IDs reset when person leaves/returns
