#!/usr/bin/env bash
# ============================================================================
# setup.sh — create the virtualenv and install dependencies (run once).
#
#   git clone <repo> && cd RGB-DT
#   ./scripts/setup.sh
#
# Safe to re-run: it reuses an existing .venv and only reinstalls if needed.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "❌ Python 3 not found. Install Python 3.10+ and re-run." >&2
  exit 1
fi
echo "▶ Using $("$PY" --version 2>&1)"

if [ ! -d .venv ]; then
  echo "▶ Creating virtualenv in .venv …"
  "$PY" -m venv .venv
fi

echo "▶ Installing dependencies …"
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

echo "✅ Setup complete. Start the app with:  ./scripts/run.sh"
