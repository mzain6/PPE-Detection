import asyncio
import os
import sys

# Add src to pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from app.database import AsyncSessionLocal
from app.models import User, Site, UserRole
from app.services.auth_service import hash_password

async def seed():
    async with AsyncSessionLocal() as session:
        # Create a site
        site = Site(name="Main Facility", location="Zone A")
        session.add(site)
        await session.flush()
        
        # Create an admin user
        admin = User(
            name="Admin User",
            email="admin@safesite.ai",
            hashed_password=hash_password("Password123"),
            role=UserRole.super_admin,
            site_ids=[site.id],
            is_active=True
        )
        session.add(admin)
        await session.commit()
        print(f"Seeded Site: {site.name} ({site.id})")
        print(f"Seeded Admin User: {admin.email} / Password: Password123")

if __name__ == "__main__":
    asyncio.run(seed())
