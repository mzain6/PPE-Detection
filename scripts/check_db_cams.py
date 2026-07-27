import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from sqlalchemy import select
from app.database import AsyncSessionLocal
from app.models import Camera

async def check_cams():
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(Camera))
        cams = res.scalars().all()
        print(f"Found {len(cams)} cameras in DB:")
        for c in cams:
            print(f"  ID: {c.id} | Name: {c.name} | URL/Source: {c.stream_url} | Type: {c.type} | Active: {c.is_active}")

if __name__ == "__main__":
    asyncio.run(check_cams())
