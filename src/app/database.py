import os
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# ── Database URL Resolution ───────────────────────────────────────────────────
# Use PostgreSQL if explicitly configured via environment variable.
# Fall back to local SQLite so the app works without a running PostgreSQL server.

_configured_url = os.getenv("DATABASE_URL", "")

if _configured_url and "postgresql" in _configured_url:
    DATABASE_URL = _configured_url
    _is_sqlite = False
else:
    # SQLite fallback — file stored next to the src/ directory
    _db_path = Path(__file__).resolve().parent.parent.parent / "safesite.db"
    DATABASE_URL = f"sqlite+aiosqlite:///{_db_path}"
    _is_sqlite = True

# ── Engine ────────────────────────────────────────────────────────────────────
if _is_sqlite:
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    """Dependency: yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
