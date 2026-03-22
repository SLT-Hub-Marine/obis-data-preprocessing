#!/usr/bin/env python3
"""
Inspect a single record's 'interpreted' column using Polars.

Reads the first record from a sample parquet file and pretty-prints
the full interpreted struct as JSON.

Usage:
    python scripts/inspect_interpreted.py
"""

import json
import glob

import polars as pl

from config import DATA_DIR

TARGET_COLUMN = "interpreted"


def main():
    # Pick the first available parquet file
    all_files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    if not all_files:
        print(f"No parquet files found in {DATA_DIR}")
        return

    sample_file = all_files[0]
    print(f"Inspecting: {sample_file}")

    # Read just the first row's interpreted column and unnest it
    row = (
        pl.scan_parquet(sample_file)
        .select(pl.col(TARGET_COLUMN))
        .head(1)
        .unnest(TARGET_COLUMN)
        .collect()
        .to_dicts()[0]
    )

    print(json.dumps(row, indent=2, default=str))


if __name__ == "__main__":
    main()
