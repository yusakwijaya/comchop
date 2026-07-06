#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
FRONTEND_URL="http://localhost:5173"

cleanup() {
  echo "Stopping servers..."
  [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting backend (uvicorn) on :8000..."
(
  cd "$BACKEND_DIR"
  ./.venv/bin/uvicorn app.main:app --reload --port 8000
) &
BACKEND_PID=$!

echo "Starting frontend (vite) on :5173..."
(
  cd "$FRONTEND_DIR"
  npm run dev
) &
FRONTEND_PID=$!

# Give vite a moment to boot, then open browser
sleep 2
open "$FRONTEND_URL" 2>/dev/null || true

wait
