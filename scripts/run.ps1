# ============================================================================
# run.ps1 — start the RGB-D-T annotation web app. Windows (PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 -Port 8080
#
# Bootstraps automatically: if .venv is missing it runs setup first, then
# migrates annotations (idempotent) and launches the server.
# ============================================================================
param([int]$Port = 8000)
$ErrorActionPreference = "Stop"
# On PS 7.4+ this stops native (python) non-zero exits from throwing; we check
# exit codes ourselves. Harmless no-op on Windows PowerShell 5.1.
$PSNativeCommandUseErrorActionPreference = $false
Set-Location (Split-Path -Parent $PSScriptRoot)

$vpy = ".venv\Scripts\python.exe"

# 1. Ensure the environment exists and has the deps.
$needSetup = $false
if (-not (Test-Path $vpy)) {
  $needSetup = $true
} else {
  & $vpy -c "import fastapi, uvicorn, cv2, PIL, numpy" 2>$null
  if ($LASTEXITCODE -ne 0) { $needSetup = $true }
}
if ($needSetup) {
  Write-Host "> Environment not ready - running setup ..."
  & "$PSScriptRoot\setup.ps1"
}

# 2. Dataset present? (Dataset\ is gitignored — must be synced separately.)
if (-not (Test-Path "Dataset\*")) {
  Write-Host ""
  Write-Host "WARNING: No scenes found under .\Dataset\"
  Write-Host "  The dataset is not in git — copy the shared Dataset\ folder to the"
  Write-Host "  project root, then re-run .\scripts\run.ps1"
  Write-Host ""
}

# 3. One-time annotation migration (idempotent: skips existing per-scene files).
Write-Host "> Migrating annotations to per-scene files ..."
try { & $vpy tools\split_dataset.py } catch { Write-Host "  (migration skipped: $_)" }

# 4. Launch (python -m uvicorn — no need to locate the uvicorn exe).
$url = "http://localhost:$Port"
Write-Host ""
Write-Host "OK - Starting annotator at $url   (Ctrl-C to stop)"
Start-Process $url -ErrorAction SilentlyContinue
& $vpy -m uvicorn app.server:app --port $Port --host 0.0.0.0
