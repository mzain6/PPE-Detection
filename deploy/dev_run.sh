#!/usr/bin/env bash
# run from repo root
python -m venv .venv || true
source .venv/bin/activate || .\.venv\Scripts\Activate
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn[standard] pydantic PyYAML ultralytics opencv-python-headless numpy aiohttp python-multipart
ren app.py legacy_app.py
echo # package init for PPE-Detection > app\__init__.py
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload