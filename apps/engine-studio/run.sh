#!/usr/bin/env bash
# Engine Studio launcher: opens the browser once the control server is listening.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${ENGINE_STUDIO_PORT:-7860}"
( sleep 1; open "http://127.0.0.1:${PORT}/" 2>/dev/null || true ) &
exec python3 server.py
