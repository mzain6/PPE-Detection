@router.post("/detect/webcam", response_model=DetectionsResponse)
def detect_webcam(device_index: Optional[int] = 0):
    """
    Open local webcam, capture one frame and run detection. Useful for testing.
    This variant performs explicit start/is_open checks and returns clear errors.
    """
    cam_id = f"webcam-{device_index}"
    try:
        register_webcam(cam_id, device_index=int(device_index), fps=settings.default_fps)
    except ValueError:
        pass  # already registered

    # Ensure stream exists and start it explicitly
    from ..services.camera_service import ensure_stream
    try:
        stream = ensure_stream(cam_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="webcam camera not registered")

    # Start stream and check availability
    try:
        stream.start()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to start webcam stream: {e}")

    # Quick health checks
    is_open = getattr(stream, "is_open", None)
    if callable(is_open):
        opened = stream.is_open()
    else:
        opened = True  # assume open if no method

    if not opened:
        raise HTTPException(status_code=503, detail="webcam not available (cap not opened)")

    # Try several reads with small backoff (some webcams need a few frames)
    frame = None
    for _ in range(3):
        frame = stream.read(timeout=1.0)
        if frame is not None:
            break
    if frame is None:
        # Provide guidance in the error message
        raise HTTPException(
            status_code=503,
            detail="no frame from webcam after multiple attempts. "
                   "Verify device index, permissions, and that no other process uses the camera."
        )

    # Run AI pipeline
    try:
        res = pipeline.process_frame(frame, camera_id=cam_id, roi=None)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    _last_detections[cam_id] = res
    return {
        "camera_id": res["camera_id"],
        "timestamp": datetime.utcfromtimestamp(res["timestamp"]),
        "fps": res.get("fps", settings.default_fps),
        "tracks": res["tracks"]
    }