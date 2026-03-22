#!/usr/bin/env bash
# Serve the outputs directory on localhost:8080 with Range request support.
# PMTiles requires HTTP Byte Serving which Python's http.server lacks.
# Open http://localhost:8080/interactive_map.html in your browser.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${1:-8080}"

echo "🌐 Serving outputs at http://localhost:$PORT"
echo "   Open: http://localhost:$PORT/interactive_map.html"
echo "   Press Ctrl+C to stop"
python "$SCRIPT_DIR/serve_range.py" "$PORT" "$PROJECT_DIR/outputs"
