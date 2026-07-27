"""
Personnel router — employee management + face photo upload.
"""
import os
import uuid as _uuid
from uuid import UUID
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import Personnel, AccessLog
from app.api_schemas import PersonnelOut, PersonnelUpdate, AccessLogOut

router = APIRouter(prefix="/api/personnel", tags=["Personnel"])

PHOTO_DIR = "data/personnel_photos"
os.makedirs(PHOTO_DIR, exist_ok=True)


@router.get("", response_model=List[PersonnelOut])
async def list_personnel(
    site_id: Optional[UUID] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(Personnel)
    if site_id:
        query = query.where(Personnel.site_id == site_id)
    result = await db.execute(query.order_by(Personnel.name))
    return result.scalars().all()


@router.post("", response_model=PersonnelOut, status_code=201)
async def register_personnel(
    name: str       = Form(...),
    employee_id: str = Form(...),
    department: str  = Form(""),
    site_id: str     = Form(...),
    photo: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
):
    photo_path = None
    if photo and photo.filename:
        ext = os.path.splitext(photo.filename)[1]
        filename = f"{_uuid.uuid4()}{ext}"
        path = os.path.join(PHOTO_DIR, filename)
        with open(path, "wb") as f:
            f.write(await photo.read())
        photo_path = path

    person = Personnel(
        name=name,
        employee_id=employee_id,
        department=department or None,
        site_id=UUID(site_id),
        photo_path=photo_path,
    )
    db.add(person)
    await db.flush()
    await db.refresh(person)
    return person


@router.put("/{person_id}", response_model=PersonnelOut)
async def update_personnel(
    person_id: UUID,
    req: PersonnelUpdate,
    db: AsyncSession = Depends(get_db),
):
    person = await db.get(Personnel, person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    for field, value in req.model_dump(exclude_none=True).items():
        setattr(person, field, value)
    await db.flush()
    await db.refresh(person)
    return person


@router.post("/{person_id}/face")
async def upload_face(
    person_id: UUID,
    photo: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Re-upload face photo and trigger embedding regeneration."""
    person = await db.get(Personnel, person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    ext = os.path.splitext(photo.filename)[1]
    filename = f"{person_id}{ext}"
    path = os.path.join(PHOTO_DIR, filename)
    with open(path, "wb") as f:
        f.write(await photo.read())

    person.photo_path = path
    # TODO: trigger face embedding generation here
    await db.flush()
    return {"success": True, "photo_path": path}


# ─── Access Log ──────────────────────────────────────

access_router = APIRouter(prefix="/api/access-log", tags=["Access Log"])


@access_router.get("", response_model=dict)
async def list_access_log(
    page: int = 1,
    page_size: int = 50,
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import func
    from app.models import Camera

    total = (await db.execute(
        select(func.count()).select_from(AccessLog)
    )).scalar()

    rows = (await db.execute(
        select(AccessLog)
        .order_by(AccessLog.timestamp.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )).scalars().all()

    items = []
    for log in rows:
        out = AccessLogOut.model_validate(log)
        if log.person_id:
            p = await db.get(Personnel, log.person_id)
            out.person_name = p.name if p else None
        cam = await db.get(Camera, log.camera_id)
        out.camera_name = cam.name if cam else None
        items.append(out)

    return {"items": items, "total": total, "page": page, "page_size": page_size}
