#!/usr/bin/env python3
"""
Export all OBIS parquet records to newline-delimited GeoJSON (.geojsonl).
Uses polars for fast vectorised processing of 200M+ records.

Usage:
    python scripts/export_geojson.py
"""

import glob
import json
import os
import sys
import time

import polars as pl

from config import DATA_DIR, OUTPUTS_DIR

OUTPUT_FILE = OUTPUTS_DIR / "records.geojsonl"

# Fields to extract from the interpreted struct
FIELDS = [
    "decimalLatitude",
    "decimalLongitude",
    "scientificName",
    "depth",
    "bathymetry",
    "phylum",
    "class",
    "order",
    "family",
    "habitat",
    "waterBody",
]

# Property name mapping (interpreted field -> geojson property)
PROP_MAP = {
    "scientificName": "species",
    "depth": "depth",
    "bathymetry": "bathymetry",
    "phylum": "phylum",
    "class": "class",
    "order": "order",
    "family": "family",
    "habitat": "habitat",
    "waterBody": "waterBody",
}

BATCH_SIZE = 200  # number of parquet files to process at once


def process_batch(files: list[str], out_fh) -> int:
    """Process a batch of parquet files and write GeoJSON lines. Returns row count."""
    try:
        df = pl.scan_parquet(files)
    except Exception as e:
        print(f"  ⚠ Skipping batch ({len(files)} files): {e}")
        return 0

    # Extract struct fields, filter for valid coordinates
    try:
        df = df.select(
            [pl.col("interpreted").struct.field(f).alias(f) for f in FIELDS]
        ).filter(
            pl.col("decimalLatitude").is_not_null()
            & pl.col("decimalLongitude").is_not_null()
        )
        result = df.collect()
    except Exception as e:
        print(f"  ⚠ Collect error: {e}")
        return 0

    if result.is_empty():
        return 0

    # Round coordinates
    lats = result["decimalLatitude"].cast(pl.Float64).round(5).to_list()
    lons = result["decimalLongitude"].cast(pl.Float64).round(5).to_list()

    # Extract property columns as lists
    prop_cols = {}
    for field, prop_name in PROP_MAP.items():
        try:
            col = result[field]
            if col.dtype in (pl.Float64, pl.Float32):
                prop_cols[prop_name] = col.round(1).to_list()
            else:
                prop_cols[prop_name] = col.to_list()
        except Exception:
            prop_cols[prop_name] = [None] * len(result)

    n = len(result)
    lines = []
    for i in range(n):
        lat = lats[i]
        lon = lons[i]
        if lat is None or lon is None:
            continue

        props = {}
        for prop_name, values in prop_cols.items():
            v = values[i]
            if v is not None:
                props[prop_name] = v

        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props,
        }
        lines.append(json.dumps(feature, separators=(",", ":"), allow_nan=False))

    if lines:
        out_fh.write("\n".join(lines))
        out_fh.write("\n")

    return len(lines)


def main():
    print("=" * 60)
    print("OBIS Parquet → GeoJSON Lines Exporter (polars)")
    print("=" * 60)

    parquet_files = sorted(glob.glob(str(DATA_DIR / "**/*.parquet"), recursive=True))
    print(f"Found {len(parquet_files)} parquet files")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    t0 = time.time()
    total = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for batch_start in range(0, len(parquet_files), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(parquet_files))
            batch_files = parquet_files[batch_start:batch_end]

            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            pct = batch_start / len(parquet_files) * 100
            print(
                f"  [{batch_start}/{len(parquet_files)}] "
                f"{pct:.1f}% | {total:,} records | "
                f"{elapsed:.0f}s | {rate:,.0f} rec/s"
            )

            count = process_batch(batch_files, out)
            total += count

    elapsed = time.time() - t0
    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    rate = total / elapsed if elapsed > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"✅ Exported {total:,} records in {elapsed:.0f}s ({rate:,.0f} rec/s)")
    print(f"   Output: {OUTPUT_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
