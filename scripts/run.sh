#!/usr/bin/env bash
# ============================================================================
# run.sh — start the RGB-D-T annotation web app.
#
#   ./scripts/run.sh            # http://localhost:8000
#   ./scripts/run.sh 8080       # custom port
#   PORT=8080 ./scripts/run.sh  # same via env
#
# Bootstraps automatically: if .venv is missing it runs setup first, then
# migrates annotations (idempotent) and launches the server.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${1:-${PORT:-8000}}"

# 1. Ensure the environment exists.
if [ ! -d .venv ] || ! .venv/bin/python -c "import fastapi, uvicorn, cv2, PIL, numpy" >/dev/null 2>&1; then
  echo "▶ Environment not ready — running setup …"
  bash scripts/setup.sh
fi

# 2. Dataset present? (Dataset/ is gitignored — must be synced separately.)
if ! ls -d Dataset/*/ >/dev/null 2>&1; then
  echo ""
  echo "⚠️  No scenes found under ./Dataset/"
  echo "   The dataset is not in git — copy the shared Dataset/ folder to the"
  echo "   project root, then re-run ./scripts/run.sh"
  echo ""
fi

# 3. One-time annotation migration (idempotent: skips existing per-scene files).
echo "▶ Migrating annotations to per-scene files …"
.venv/bin/python tools/split_dataset.py || true

# 4. Launch.
URL="http://localhost:${PORT}"
echo ""
echo "✅ Starting annotator at ${URL}   (Ctrl-C to stop)"
( sleep 2; (command -v open >/dev/null && open "$URL") \
        || (command -v xdg-open >/dev/null && xdg-open "$URL") ) >/dev/null 2>&1 &
exec .venv/bin/uvicorn app.server:app --port "$PORT" --host 0.0.0.0
