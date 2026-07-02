# ============================================================================
# setup.ps1 — create the virtualenv and install dependencies (run once). Windows.
#
#   git clone <repo>; cd RGB-DT
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
#
# Safe to re-run: reuses an existing .venv.
# ============================================================================
$ErrorActionPreference = "Stop"
# On PS 7.4+ this stops native (python/pip) non-zero exits from throwing; we
# handle exit codes ourselves. Harmless no-op on Windows PowerShell 5.1.
$PSNativeCommandUseErrorActionPreference = $false
Set-Location (Split-Path -Parent $PSScriptRoot)

# Find a Python launcher: prefer "py", fall back to "python".
$py = $null
foreach ($c in @("py", "python")) {
  if (Get-Command $c -ErrorAction SilentlyContinue) { $py = $c; break }
}
if (-not $py) {
  Write-Error "Python 3 not found. Install Python 3.10+ from https://python.org (tick 'Add to PATH') and re-run."
  exit 1
}
Write-Host "> Using $(& $py --version)"

if (-not (Test-Path ".venv")) {
  Write-Host "> Creating virtualenv in .venv ..."
  & $py -m venv .venv
}
$vpy = ".venv\Scripts\python.exe"

Write-Host "> Installing dependencies ..."
& $vpy -m pip install --quiet --upgrade pip
& $vpy -m pip install --quiet -r requirements.txt

Write-Host "OK - Setup complete. Start the app with:  .\scripts\run.ps1"
