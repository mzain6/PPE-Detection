import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from sqlalchemy import select
from app.database import AsyncSessionLocal
from app.models import Camera, CameraType

async def fix_jhb():
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(Camera))
        cams = res.scalars().all()
        for c in cams:
            print(f"Updating camera {c.name} ({c.id}) -> Type: webcam, stream_url: '0'")
            c.type = CameraType.webcam
            c.stream_url = "0"
            c.is_active = True
        await session.commit()
        print("Done updating DB cameras.")

if __name__ == "__main__":
    asyncio.run(fix_jhb())
