#!/usr/bin/env bash
# Build PMTiles from exported GeoJSON lines using tippecanoe.
#
# Usage:
#     bash scripts/build_tiles.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INPUT="$PROJECT_DIR/outputs/records.geojsonl"
OUTPUT="$PROJECT_DIR/outputs/obis.pmtiles"

if [ ! -f "$INPUT" ]; then
    echo "❌ $INPUT not found. Run export_geojson.py first."
    exit 1
fi

echo "🔨 Building PMTiles from $(wc -l < "$INPUT") features..."
echo "   Input:  $INPUT"
echo "   Output: $OUTPUT"

TIPPECANOE="${HOME}/bin/tippecanoe"
if [ ! -x "$TIPPECANOE" ]; then
    TIPPECANOE="tippecanoe"
fi

"$TIPPECANOE" \
    --output="$OUTPUT" \
    --force \
    --layer=obis \
    --name="OBIS Occurrences" \
    --description="OBIS ocean biodiversity occurrence records" \
    --attribution="© OBIS" \
    --minimum-zoom=0 \
    --maximum-zoom=12 \
    --drop-densest-as-needed \
    --no-tile-compression \
    "$INPUT"

SIZE_MB=$(du -m "$OUTPUT" | cut -f1)
echo ""
echo "✅ PMTiles created: $OUTPUT ($SIZE_MB MB)"
echo "   Serve with: python -m http.server 8080 -d outputs"
