# SafeSite AI — Frontend Startup Script
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   SafeSite AI Frontend — Starting..." -ForegroundColor Cyan
Write-Host "   Next.js 16 on http://localhost:3000" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan

$env:NEXT_PUBLIC_API_URL = "http://localhost:8000"
$env:NEXTAUTH_URL = "http://localhost:3000"
$env:NEXTAUTH_SECRET = "safesite-nextauth-secret-key-min-32-chars"

Set-Location frontend
npm run dev
