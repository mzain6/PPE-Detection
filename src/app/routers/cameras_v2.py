"""
Cameras router — full CRUD + test stream + active camera list.
"""
import cv2
import asyncio
from uuid import UUID
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.database import get_db
from app.models import Camera
from app.api_schemas import CameraCreate, CameraUpdate, CameraOut, CameraTestRequest, CameraTestResponse

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


@router.get("", response_model=List[CameraOut])
async def list_cameras(
    site_id: Optional[UUID] = None,
    db: AsyncSession = Depends(get_db)
):
    query = select(Camera)
    if site_id:
        query = query.where(Camera.site_id == site_id)
    result = await db.execute(query.order_by(Camera.created_at))
    return result.scalars().all()


@router.post("", response_model=CameraOut, status_code=status.HTTP_201_CREATED)
async def create_camera(req: CameraCreate, db: AsyncSession = Depends(get_db)):
    camera = Camera(**req.model_dump())
    db.add(camera)
    await db.flush()
    await db.refresh(camera)
    return camera


@router.put("/{camera_id}", response_model=CameraOut)
async def update_camera(
    camera_id: UUID,
    req: CameraUpdate,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    for field, value in req.model_dump(exclude_none=True).items():
        setattr(camera, field, value)

    await db.flush()
    await db.refresh(camera)
    return camera


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(camera_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    from app.models import Violation, AccessLog
    from sqlalchemy import delete
    # Manually delete dependent records to avoid DB-level FK constraint errors
    await db.execute(delete(Violation).where(Violation.camera_id == camera_id))
    await db.execute(delete(AccessLog).where(AccessLog.camera_id == camera_id))
    
    await db.delete(camera)


@router.post("/test", response_model=CameraTestResponse)
async def test_camera_stream(req: CameraTestRequest):
    """Test if an RTSP/webcam URL is reachable."""
    try:
        source = int(req.stream_url) if req.stream_url.isdigit() else req.stream_url
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            cap.release()
            return CameraTestResponse(success=True, message="Stream connected successfully")
        cap.release()
        return CameraTestResponse(success=False, message="Could not open stream — check URL/network")
    except Exception as e:
        return CameraTestResponse(success=False, message=str(e))


from app.services.frame_store import get_frame
import asyncio

async def _mjpeg_generator(camera_id: str):
    """Yield MJPEG frames from the central frame store (annotated by the background worker)."""
    seq = 0
    while True:
        try:
            # Run the synchronous blocking call in a thread pool so it doesn't block the event loop
            frame, seq = await asyncio.to_thread(get_frame, camera_id, seq, 2.0)
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            
            import cv2
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        except asyncio.CancelledError:
            break
        except Exception as e:
            await asyncio.sleep(1.0)


@router.get("/{camera_id}/stream")
async def stream_camera(camera_id: UUID):
    """Stream MJPEG frames from a camera. Uses a manual DB session to avoid exhausting the pool."""
    from app.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Camera).where(Camera.id == camera_id))
        camera = result.scalar_one_or_none()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        # Update last_seen
        await db.execute(
            update(Camera).where(Camera.id == camera_id).values(last_seen=datetime.utcnow())
        )
        await db.commit()

    return StreamingResponse(
        _mjpeg_generator(str(camera_id)),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )
