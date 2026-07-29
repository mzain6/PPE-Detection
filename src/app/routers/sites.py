"""
Sites router — list, create, and delete sites.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid as _uuid

from app.database import get_db
from app.models import Site
from app.api_schemas import SiteOut

router = APIRouter(prefix="/api/sites", tags=["Sites"])


class SiteCreate(BaseModel):
    name: str
    location: Optional[str] = ""


@router.get("", response_model=List[SiteOut])
async def list_sites(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Site).order_by(Site.name))
    return result.scalars().all()


@router.post("", response_model=SiteOut, status_code=201)
async def create_site(body: SiteCreate, db: AsyncSession = Depends(get_db)):
    """Create a new site. Accepts JSON body: {name, location}."""
    site = Site(name=body.name, location=body.location or None)
    db.add(site)
    await db.flush()
    await db.refresh(site)
    return site


@router.delete("/{site_id}", status_code=204)
async def delete_site(site_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a site by ID."""
    result = await db.execute(select(Site).where(Site.id == _uuid.UUID(site_id)))
    site = result.scalar_one_or_none()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    await db.delete(site)
    return None
