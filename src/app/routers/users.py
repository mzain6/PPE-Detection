"""
Users router — admin-only user management + invite.
"""
from uuid import UUID
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User
from app.api_schemas import UserCreate, UserUpdate, UserOut
from app.services.auth_service import hash_password

router = APIRouter(prefix="/api/users", tags=["Users"])


@router.get("", response_model=List[UserOut])
async def list_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).order_by(User.created_at))
    return result.scalars().all()


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def invite_user(req: UserCreate, db: AsyncSession = Depends(get_db)):
    # Check email uniqueness
    existing = (await db.execute(
        select(User).where(User.email == req.email)
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    # Create user with a temporary password (they'll reset via email in production)
    import secrets
    temp_password = secrets.token_urlsafe(12)
    user = User(
        name=req.name,
        email=req.email,
        role=req.role,
        site_ids=req.site_ids or [],
        hashed_password=hash_password(temp_password),
        is_active=True,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    # TODO: send invite email with temp_password
    return user


@router.put("/{user_id}", response_model=UserOut)
async def update_user(
    user_id: UUID,
    req: UserUpdate,
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    for field, value in req.model_dump(exclude_none=True).items():
        setattr(user, field, value)
    await db.flush()
    await db.refresh(user)
    return user
