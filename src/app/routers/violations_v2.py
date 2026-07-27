"""
Violations router — paginated list, filtering, review, export.
"""
import csv
import io
from uuid import UUID
from datetime import datetime, date
from typing import Optional, List

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.database import get_db
from app.models import Violation, Camera, Personnel, ViolationType
from app.api_schemas import ViolationOut, ViolationListResponse, ViolationCreate

router = APIRouter(prefix="/api/violations", tags=["Violations"])


def _build_filter(
    camera_id: Optional[UUID],
    site_id: Optional[UUID],
    violation_type: Optional[str],
    date_from: Optional[date],
    date_to: Optional[date],
    since: Optional[datetime],
    is_reviewed: Optional[bool],
):
    filters = []
    if camera_id:
        filters.append(Violation.camera_id == camera_id)
    if site_id:
        filters.append(Violation.site_id == site_id)
    if violation_type and violation_type != "all":
        filters.append(Violation.violation_type == violation_type)
    if date_from:
        filters.append(Violation.timestamp >= datetime.combine(date_from, datetime.min.time()))
    if date_to:
        filters.append(Violation.timestamp <= datetime.combine(date_to, datetime.max.time()))
    if since:
        filters.append(Violation.timestamp > since.replace(tzinfo=None))
    if is_reviewed is not None:
        filters.append(Violation.is_reviewed == is_reviewed)
    return filters


@router.get("", response_model=ViolationListResponse)
async def list_violations(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    camera_id: Optional[UUID] = None,
    site_id: Optional[UUID] = None,
    violation_type: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    since: Optional[datetime] = None,
    is_reviewed: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
):
    filters = _build_filter(camera_id, site_id, violation_type, date_from, date_to, since, is_reviewed)

    count_q = select(func.count()).select_from(Violation)
    if filters:
        count_q = count_q.where(and_(*filters))
    total = (await db.execute(count_q)).scalar()

    query = (
        select(Violation)
        .where(and_(*filters) if filters else True)
        .order_by(Violation.timestamp.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = (await db.execute(query)).scalars().all()

    items = []
    for v in rows:
        out = ViolationOut.model_validate(v)
        if v.person_id:
            person = await db.get(Personnel, v.person_id)
            out.person_name = person.name if person else None
        camera = await db.get(Camera, v.camera_id)
        out.camera_name = camera.name if camera else None
        items.append(out)

    return ViolationListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=max(1, -(-total // page_size)),
    )


@router.post("", response_model=ViolationOut, status_code=201)
async def create_violation(req: ViolationCreate, db: AsyncSession = Depends(get_db)):
    """Called by the AI detection engine to log a new violation."""
    v = Violation(**req.model_dump())
    db.add(v)
    await db.flush()
    await db.refresh(v)
    return ViolationOut.model_validate(v)


@router.patch("/{violation_id}/review")
async def mark_reviewed(violation_id: UUID, db: AsyncSession = Depends(get_db)):
    v = await db.get(Violation, violation_id)
    if not v:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Violation not found")
    v.is_reviewed = True
    await db.flush()
    return {"success": True}


@router.get("/export/csv")
async def export_csv(
    camera_id: Optional[UUID] = None,
    site_id: Optional[UUID] = None,
    violation_type: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    db: AsyncSession = Depends(get_db),
):
    filters = _build_filter(camera_id, site_id, violation_type, date_from, date_to, None, None)
    query = (
        select(Violation)
        .where(and_(*filters) if filters else True)
        .order_by(Violation.timestamp.desc())
    )
    rows = (await db.execute(query)).scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Camera ID", "Site ID", "Violation Type", "Confidence", "Reviewed"])
    for v in rows:
        writer.writerow([
            str(v.id), v.timestamp.isoformat(), str(v.camera_id),
            str(v.site_id), v.violation_type.value, f"{v.confidence:.3f}", v.is_reviewed
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=violations.csv"},
    )
