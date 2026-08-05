"""
Analytics router — KPI summary + chart data for all dashboard charts.
"""
from datetime import datetime, date, timedelta
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, cast, Date

from app.database import get_db
from app.models import Violation, Camera, ViolationType
from app.api_schemas import (
    SummaryResponse, ViolationsByTypeItem, ViolationsByCameraItem,
    TrendItem, PeakHourCell
)

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/summary", response_model=SummaryResponse)
async def get_summary(db: AsyncSession = Depends(get_db)):
    today = date.today()
    today_start = datetime.combine(today, datetime.min.time())

    # Total violations today
    total_today = (await db.execute(
        select(func.count()).select_from(Violation).where(Violation.timestamp >= today_start)
    )).scalar()

    # Camera stats
    all_cams = (await db.execute(select(func.count()).select_from(Camera))).scalar()
    online_threshold = datetime.utcnow() - timedelta(seconds=30)
    active_cams = (await db.execute(
        select(func.count()).select_from(Camera).where(
            and_(Camera.is_active == True, Camera.last_seen >= online_threshold)
        )
    )).scalar()
    offline_cams = all_cams - active_cams

    # Unreviewed
    unreviewed = (await db.execute(
        select(func.count()).select_from(Violation).where(Violation.is_reviewed == False)
    )).scalar()

    return SummaryResponse(
        total_violations_today=total_today,
        active_cameras=active_cams,
        total_cameras=all_cams,
        unreviewed_incidents=unreviewed,
        offline_cameras=offline_cams,
        system_healthy=(offline_cams == 0),
    )


@router.get("/violations-by-type", response_model=List[ViolationsByTypeItem])
async def violations_by_type(
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    db: AsyncSession = Depends(get_db),
):
    filters = []
    if date_from:
        filters.append(Violation.timestamp >= datetime.combine(date_from, datetime.min.time()))
    if date_to:
        filters.append(Violation.timestamp <= datetime.combine(date_to, datetime.max.time()))

    query = (
        select(Violation.violation_type, func.count().label("count"))
        .where(and_(*filters) if filters else True)
        .group_by(Violation.violation_type)
    )
    rows = (await db.execute(query)).all()
    return [ViolationsByTypeItem(type=r[0].value, count=r[1]) for r in rows]


@router.get("/violations-by-camera", response_model=List[ViolationsByCameraItem])
async def violations_by_camera(
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    db: AsyncSession = Depends(get_db),
):
    filters = []
    if date_from:
        filters.append(Violation.timestamp >= datetime.combine(date_from, datetime.min.time()))
    if date_to:
        filters.append(Violation.timestamp <= datetime.combine(date_to, datetime.max.time()))

    query = (
        select(Camera.name, func.count(Violation.id).label("count"))
        .join(Camera, Violation.camera_id == Camera.id)
        .where(and_(*filters) if filters else True)
        .group_by(Camera.name)
        .order_by(func.count(Violation.id).desc())
        .limit(10)
    )
    rows = (await db.execute(query)).all()
    return [ViolationsByCameraItem(camera_name=r[0], count=r[1]) for r in rows]


@router.get("/trend", response_model=List[TrendItem])
async def violation_trend(
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    start = datetime.utcnow() - timedelta(days=days)
    query = (
        select(Violation.timestamp)
        .where(Violation.timestamp >= start)
        .order_by(Violation.timestamp.asc())
    )
    rows = (await db.execute(query)).scalars().all()
    
    counts_by_date = {}
    for ts in rows:
        if ts:
            d_str = ts.strftime("%Y-%m-%d")
            counts_by_date[d_str] = counts_by_date.get(d_str, 0) + 1

    return [TrendItem(date=d, count=c) for d, c in counts_by_date.items()]


@router.get("/peak-hours", response_model=List[PeakHourCell])
async def peak_hours(
    days: int = Query(30, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """Returns violation counts per (day_of_week, hour) for heatmap."""
    start = datetime.utcnow() - timedelta(days=days)
    query = (
        select(
            func.extract("dow", Violation.timestamp).label("day"),
            func.extract("hour", Violation.timestamp).label("hour"),
            func.count().label("count")
        )
        .where(Violation.timestamp >= start)
        .group_by("day", "hour")
    )
    rows = (await db.execute(query)).all()
    return [PeakHourCell(day=int(r[0]), hour=int(r[1]), count=r[2]) for r in rows]
