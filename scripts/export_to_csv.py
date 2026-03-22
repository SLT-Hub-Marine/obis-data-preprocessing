#!/usr/bin/env python3
"""
Export interested fields from the 'interpreted' column to CSV using Polars.

Reads all parquet files in the data directory, extracts the fields
defined in INTERESTED_FIELDS from the interpreted struct, and writes
them to a CSV file.

Usage:
    python scripts/export_to_csv.py
"""

import polars as pl

from config import DATA_DIR, OUTPUTS_DIR, INTERESTED_FIELDS

TARGET_COLUMN = "interpreted"
OUTPUT_FILE = OUTPUTS_DIR / "extracted.csv"


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning parquet files in {DATA_DIR}...")

    # Build expressions to extract each interested field from the interpreted struct
    field_exprs = [
        pl.col(TARGET_COLUMN).struct.field(f).alias(f)
        for f in INTERESTED_FIELDS
    ]

    # Lazy scan all parquet files, extract fields, and sink to CSV
    df = (
        pl.scan_parquet(str(DATA_DIR / "*.parquet"))
        .select(field_exprs)
        .collect(engine="streaming")
    )

    df.write_csv(OUTPUT_FILE)
    print(f"Wrote {len(df):,} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
