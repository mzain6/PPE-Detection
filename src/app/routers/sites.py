"""
Sites router — list sites (used by frontend site selector).
"""
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import Site
from app.api_schemas import SiteOut

router = APIRouter(prefix="/api/sites", tags=["Sites"])


@router.get("", response_model=List[SiteOut])
async def list_sites(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Site).order_by(Site.name))
    return result.scalars().all()


@router.post("", response_model=SiteOut, status_code=201)
async def create_site(name: str, location: str = "", db: AsyncSession = Depends(get_db)):
    site = Site(name=name, location=location or None)
    db.add(site)
    await db.flush()
    await db.refresh(site)
    return site
