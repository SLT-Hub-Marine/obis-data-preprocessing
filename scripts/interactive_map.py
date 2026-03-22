#!/usr/bin/env python3
"""
OBIS Interactive Map — PMTiles Viewer Generator
Generates a self-contained HTML viewer that loads vector tiles from a local
PMTiles file via MapLibre GL JS.

Usage:
    python scripts/interactive_map.py
    bash scripts/serve.sh
    # Open http://localhost:8080/interactive_map.html
"""

import os
from config import OUTPUTS_DIR

OUTPUT_FILE = OUTPUTS_DIR / 'interactive_map.html'
PMTILES_FILENAME = 'obis.pmtiles'

# Phylum colour palette — 10 most common marine phyla
PHYLUM_COLORS = {
    'Chordata':       '#e74c3c',
    'Arthropoda':     '#e67e22',
    'Mollusca':       '#f1c40f',
    'Cnidaria':       '#9b59b6',
    'Echinodermata':  '#1abc9c',
    'Annelida':       '#2ecc71',
    'Bryozoa':        '#3498db',
    'Porifera':       '#e84393',
    'Rhodophyta':     '#d63031',
    'Ochrophyta':     '#6c5ce7',
}
DEFAULT_COLOR = '#7f8c8d'


def build_match_expr():
    """Build a MapLibre match expression for phylum → colour."""
    parts = []
    for ph, col in PHYLUM_COLORS.items():
        parts.append(f'"{ph}", "{col}"')
    return ', '.join(parts)


def build_legend_rows():
    rows = ''
    for ph, col in PHYLUM_COLORS.items():
        rows += (
            f'<div class="lr" data-ph="{ph}" onclick="togglePhylum(\'{ph}\')">'
            f'<span class="dot" style="background:{col}"></span>{ph}</div>\n'
        )
    return rows


def main():
    match_expr = build_match_expr()
    legend_rows = build_legend_rows()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OBIS Occurrence Map — All Records</title>
<meta name="description" content="Interactive map of OBIS ocean occurrence records with vector tiles">
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/pmtiles@4.0.0/dist/pmtiles.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{height:100%;overflow:hidden}}
body{{font-family:'Inter',system-ui,-apple-system,sans-serif;background:#0a0e1a;color:#e0e6ed}}
#map{{position:absolute;inset:0}}

/* Loading */
.loader{{position:fixed;inset:0;z-index:9999;background:#0a0e1a;display:flex;flex-direction:column;align-items:center;justify-content:center;transition:opacity 0.6s,visibility 0.6s}}
.loader.hide{{opacity:0;visibility:hidden;pointer-events:none}}
.loader h1{{font-size:26px;font-weight:700;background:linear-gradient(135deg,#56ccf2,#2f80ed);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:14px}}
.loader p{{color:#556;font-size:13px;margin-top:8px}}
.spinner{{width:36px;height:36px;border:3px solid rgba(86,204,242,0.15);border-top-color:#56ccf2;border-radius:50%;animation:spin 0.7s linear infinite}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}

/* Header */
.hdr{{position:fixed;top:0;left:0;right:0;z-index:1100;background:linear-gradient(180deg,rgba(10,14,26,0.96) 0%,rgba(10,14,26,0.7) 70%,transparent 100%);padding:10px 16px 20px;pointer-events:none}}
.hdr h1{{font-size:16px;font-weight:700;background:linear-gradient(135deg,#56ccf2,#2f80ed);-webkit-background-clip:text;-webkit-text-fill-color:transparent;pointer-events:auto}}
.hdr .sub{{font-size:11px;color:#556;margin-top:2px}}

/* Glass panels */
.gp{{position:absolute;z-index:1100;background:rgba(12,16,30,0.88);border:1px solid rgba(86,204,242,0.12);border-radius:10px;padding:12px 14px;backdrop-filter:blur(16px);font-size:12px;box-shadow:0 4px 24px rgba(0,0,0,0.35)}}
.gp h3{{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#56ccf2;margin-bottom:6px;font-weight:600}}
.dot{{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:3px;vertical-align:middle}}

/* Legend */
.bl{{bottom:10px;left:10px}}
.lr{{display:flex;align-items:center;gap:5px;margin:3px 0;cursor:pointer;padding:2px 3px;border-radius:3px;transition:opacity 0.15s;font-size:11px}}
.lr:hover{{background:rgba(86,204,242,0.05)}}
.lr.dim{{opacity:0.2}}

/* Map style switcher */
.ms{{top:52px;left:10px}}
.ms-btn{{padding:4px 8px;border:1px solid rgba(86,204,242,0.12);border-radius:5px;background:transparent;color:#556;cursor:pointer;font-size:10px;font-family:inherit;margin:1px;transition:all 0.12s}}
.ms-btn:hover,.ms-btn.on{{background:rgba(47,128,237,0.12);color:#56ccf2;border-color:#56ccf240}}

/* Search */
.rp{{top:52px;right:10px;width:220px}}
.rp input[type=text]{{width:100%;padding:6px 9px;border:1px solid rgba(86,204,242,0.18);border-radius:7px;background:rgba(255,255,255,0.03);color:#e0e6ed;font-size:12px;font-family:inherit;outline:none;transition:border-color 0.2s}}
.rp input:focus{{border-color:#56ccf2}}
.rp input::placeholder{{color:#334}}
.qf{{display:flex;flex-wrap:wrap;gap:3px;margin:5px 0}}
.qb{{padding:3px 7px;border:1px solid rgba(86,204,242,0.15);border-radius:5px;background:transparent;color:#556;cursor:pointer;font-size:10px;font-family:inherit;transition:all 0.15s}}
.qb:hover,.qb.on{{background:rgba(47,128,237,0.15);color:#56ccf2;border-color:rgba(86,204,242,0.3)}}
.sep{{border:none;border-top:1px solid rgba(86,204,242,0.06);margin:8px 0}}

/* Popup */
.maplibregl-popup-content{{background:rgba(12,16,30,0.96)!important;border:1px solid rgba(86,204,242,0.2)!important;border-radius:10px!important;box-shadow:0 6px 28px rgba(0,0,0,0.5)!important;color:#e0e6ed!important;padding:10px 12px!important;font-family:'Inter',sans-serif!important}}
.maplibregl-popup-tip{{border-top-color:rgba(12,16,30,0.96)!important}}
.maplibregl-popup-close-button{{color:#556!important;font-size:16px!important}}
.pt{{font-size:13px;font-weight:700;color:#56ccf2;margin-bottom:4px}}
.pr{{font-size:11px;margin:2px 0;color:#7b8d9e}}
.pr b{{color:#b8c8d4}}
.pb{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:9px;font-weight:700;letter-spacing:0.4px}}
.tax{{font-size:10px;color:#445;margin-top:3px}}
.tax span{{color:#5b6d7e}}

/* Info panel */
.ip{{bottom:10px;right:10px;width:200px}}
.ip .stat{{display:flex;justify-content:space-between;margin:2px 0;font-size:11px;color:#6b7d8e}}
.ip .stat b{{color:#b8c8d4}}

/* Zoom control override */
.maplibregl-ctrl-group button{{background:rgba(12,16,30,0.9)!important;color:#56ccf2!important;border-color:rgba(86,204,242,0.15)!important}}
.maplibregl-ctrl-group button:hover{{background:rgba(47,128,237,0.2)!important}}
.maplibregl-ctrl-attrib{{background:rgba(12,16,30,0.7)!important;color:#334!important;font-size:9px!important}}
.maplibregl-ctrl-attrib a{{color:#445!important}}
</style>
</head>
<body>

<div class="loader" id="loader">
    <h1>🌊 OBIS Occurrence Map</h1>
    <div class="spinner"></div>
    <p>Loading vector tiles...</p>
</div>

<div class="hdr">
    <h1>🌊 OBIS Occurrence Map — All Records</h1>
    <div class="sub">Vector tiles · Colored by phylum</div>
</div>

<div id="map"></div>

<!-- Search & Filter -->
<div class="gp rp" id="filterPanel">
    <h3>🔍 Filter by Phylum</h3>
    <div class="qf">
        <button class="qb" onclick="filterPhylum('Chordata')">🐟 Chordata</button>
        <button class="qb" onclick="filterPhylum('Arthropoda')">🦀 Arthropoda</button>
        <button class="qb" onclick="filterPhylum('Mollusca')">🐚 Mollusca</button>
        <button class="qb" onclick="filterPhylum('Cnidaria')">🪼 Cnidaria</button>
        <button class="qb" onclick="filterPhylum('Echinodermata')">⭐ Echino</button>
        <button class="qb" onclick="filterPhylum('Annelida')">🪱 Annelida</button>
        <button class="qb on" onclick="filterPhylum('')">All</button>
    </div>
</div>

<!-- Legend -->
<div class="gp bl" id="legend">
    <h3>Phylum</h3>
    {legend_rows}
</div>

<!-- Map style -->
<div class="gp ms">
    <h3>🗺 Map</h3>
    <button class="ms-btn on" onclick="setStyle(0)">Dark</button>
    <button class="ms-btn" onclick="setStyle(1)">Satellite</button>
    <button class="ms-btn" onclick="setStyle(2)">Light</button>
</div>

<!-- Info -->
<div class="gp ip" id="info">
    <h3>📊 Viewport</h3>
    <div class="stat"><span>Features visible:</span><b id="featCount">—</b></div>
    <div class="stat"><span>Zoom:</span><b id="zoomLvl">3</b></div>
</div>

<script>
"use strict";

// Register PMTiles protocol
const protocol = new pmtiles.Protocol();
maplibregl.addProtocol("pmtiles", protocol.tile);

const TILE_URL = "pmtiles://" + window.location.origin + "/{PMTILES_FILENAME}";

const STYLES = [
    {{ name: "Dark",      url: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json" }},
    {{ name: "Satellite",  url: "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json" }},
    {{ name: "Light",     url: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json" }}
];

let currentFilter = null;
const hiddenPhyla = new Set();

const map = new maplibregl.Map({{
    container: 'map',
    style: STYLES[0].url,
    center: [0, 30],
    zoom: 3,
    attributionControl: true
}});

map.addControl(new maplibregl.NavigationControl(), 'top-right');

function addOBISLayer() {{
    if (map.getSource('obis')) return;

    map.addSource('obis', {{
        type: 'vector',
        url: TILE_URL
    }});

    map.addLayer({{
        id: 'obis-points',
        type: 'circle',
        source: 'obis',
        'source-layer': 'obis',
        paint: {{
            'circle-radius': [
                'interpolate', ['linear'], ['zoom'],
                0, 1.5,
                5, 3,
                10, 5,
                14, 8
            ],
            'circle-color': [
                'match', ['get', 'phylum'],
                {match_expr},
                '{DEFAULT_COLOR}'
            ],
            'circle-opacity': 0.85,
            'circle-stroke-width': 0.3,
            'circle-stroke-color': 'rgba(255,255,255,0.2)'
        }}
    }});
}}

map.on('load', function() {{
    addOBISLayer();

    // Hide loader
    setTimeout(() => {{
        const l = document.getElementById('loader');
        l.classList.add('hide');
        setTimeout(() => l.remove(), 700);
    }}, 600);
}});

// Popup on click
map.on('click', 'obis-points', function(e) {{
    const f = e.features[0];
    if (!f) return;
    const p = f.properties;
    const dp = p.depth != null ? p.depth + 'm' : '—';
    const bt = p.bathymetry != null ? p.bathymetry + 'm' : '—';
    const tax = [p.phylum, p['class'], p.order, p.family].filter(Boolean);
    const txh = tax.length ? '<div class="tax">' + tax.map(t => '<span>' + t + '</span>').join(' › ') + '</div>' : '';

    const phCol = ({{ {', '.join(f'"{ph}": "{col}"' for ph, col in PHYLUM_COLORS.items())} }})[p.phylum] || '{DEFAULT_COLOR}';
    const phBadge = p.phylum ? '<span class="pb" style="background:' + phCol + '20;color:' + phCol + ';border:1px solid ' + phCol + '40">' + p.phylum + '</span>' : '';

    const html = '<div class="pt">' + (p.species || 'Unknown') + '</div>' +
        '<div class="pr">' + phBadge + '</div>' +
        '<div class="pr"><b>Depth:</b> ' + dp + ' · <b>Bathy:</b> ' + bt + '</div>' +
        txh +
        (p.habitat ? '<div class="pr"><b>Habitat:</b> ' + p.habitat + '</div>' : '') +
        (p.waterBody ? '<div class="pr"><b>Water Body:</b> ' + p.waterBody + '</div>' : '');

    new maplibregl.Popup({{ maxWidth: '280px' }})
        .setLngLat(e.lngLat)
        .setHTML(html)
        .addTo(map);
}});

// Cursor
map.on('mouseenter', 'obis-points', () => {{ map.getCanvas().style.cursor = 'pointer'; }});
map.on('mouseleave', 'obis-points', () => {{ map.getCanvas().style.cursor = ''; }});

// Feature count + zoom
function updateInfo() {{
    const z = map.getZoom().toFixed(1);
    document.getElementById('zoomLvl').textContent = z;
    if (map.getLayer('obis-points')) {{
        const features = map.queryRenderedFeatures({{ layers: ['obis-points'] }});
        document.getElementById('featCount').textContent = features.length.toLocaleString();
    }}
}}
map.on('moveend', updateInfo);
map.on('zoomend', updateInfo);

// Filter by phylum
function filterPhylum(ph) {{
    currentFilter = ph || null;
    document.querySelectorAll('.qb').forEach(b => {{
        if (!ph && b.textContent.includes('All')) b.classList.add('on');
        else if (ph && b.textContent.toLowerCase().includes(ph.toLowerCase().substring(0, 4))) b.classList.add('on');
        else b.classList.remove('on');
    }});
    applyFilter();
}}

function togglePhylum(ph) {{
    if (hiddenPhyla.has(ph)) hiddenPhyla.delete(ph);
    else hiddenPhyla.add(ph);
    document.querySelectorAll('.lr').forEach(r => r.classList.toggle('dim', hiddenPhyla.has(r.dataset.ph)));
    applyFilter();
}}

function applyFilter() {{
    if (!map.getLayer('obis-points')) return;
    const filters = ['all'];
    if (currentFilter) {{
        filters.push(['==', ['get', 'phylum'], currentFilter]);
    }}
    if (hiddenPhyla.size > 0) {{
        hiddenPhyla.forEach(ph => {{
            filters.push(['!=', ['get', 'phylum'], ph]);
        }});
    }}
    map.setFilter('obis-points', filters.length > 1 ? filters : null);
    setTimeout(updateInfo, 100);
}}

// Style switcher
function setStyle(idx) {{
    document.querySelectorAll('.ms-btn').forEach((b, j) => b.classList.toggle('on', j === idx));
    map.setStyle(STYLES[idx].url);
    map.once('style.load', () => {{
        addOBISLayer();
        applyFilter();
    }});
}}
</script>
</body>
</html>"""

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"✅ Saved: {OUTPUT_FILE} ({size_kb:.1f} KB)")
    print(f"   Requires: {PMTILES_FILENAME} in the same directory")
    print(f"   Serve with: bash scripts/serve.sh")
    print(f"   Open: http://localhost:8080/interactive_map.html")


if __name__ == '__main__':
    main()
