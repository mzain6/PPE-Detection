# SafeSite AI — Backend Startup Script
# Forces NVIDIA T500 GPU (CUDA device 0) for all inference

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   SafeSite AI Backend — Starting..." -ForegroundColor Cyan
Write-Host "   GPU: NVIDIA T500 (CUDA:0)" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan

# Set environment variables for GPU
$env:CUDA_VISIBLE_DEVICES = "0"
$env:TORCH_DEVICE = "cuda:0"
$env:SECRET_KEY = "safesite-jwt-secret-key-min-32-chars-secure"
$env:ACCESS_TOKEN_EXPIRE_MINUTES = "1440"
$env:NEXT_PUBLIC_API_URL = "http://localhost:8000"
$env:DATABASE_URL = "postgresql+asyncpg://safesite_user:safesite_pass@localhost:5432/safesite"

Write-Host "`n[INFO] CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES" -ForegroundColor Yellow
Write-Host "[INFO] Working directory: src\" -ForegroundColor Yellow

# Run from the src directory so imports resolve correctly
Set-Location src
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
