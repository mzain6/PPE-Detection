"""
reset_admin_password.py
Resets the admin@safesite.ai password to Password123 with a fresh valid bcrypt hash.
Run once: python scripts/reset_admin_password.py
"""
import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from sqlalchemy import select
from app.database import AsyncSessionLocal
from app.models import User
from app.services.auth_service import hash_password

async def reset():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.email == "admin@safesite.ai"))
        user = result.scalar_one_or_none()

        if not user:
            print("[ERROR] Admin user not found. Run seed_db.py first.")
            return

        new_hash = hash_password("Password123")
        user.hashed_password = new_hash
        user.is_active = True
        await session.commit()
        print(f"[OK] Password reset for: {user.email}")
        print(f"     Login with: admin@safesite.ai / Password123")

if __name__ == "__main__":
    asyncio.run(reset())
