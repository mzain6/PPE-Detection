"""
Auth router — /api/auth/login
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import User, UserRole
from app.api_schemas import LoginRequest, LoginResponse, UserOut, UserRegister
from app.services.auth_service import verify_password, create_access_token, hash_password

router = APIRouter(prefix="/api/auth", tags=["Auth"])


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )

    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "name": user.name,
        "role": user.role.value,
        "site_ids": [str(s) for s in (user.site_ids or [])],
    }
    access_token = create_access_token(token_data)

    return LoginResponse(
        access_token=access_token,
        user=UserOut.model_validate(user),
    )


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(req: UserRegister, db: AsyncSession = Depends(get_db)):
    # Check email uniqueness
    existing = (await db.execute(
        select(User).where(User.email == req.email)
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Create new user
    user = User(
        name=req.name,
        email=req.email,
        hashed_password=hash_password(req.password),
        role=UserRole.viewer,  # Default role for public signups
        site_ids=[],           # Empty list by default
        is_active=True,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    return user
