#!/usr/bin/env python3
"""
Count total and valid samples for a specific field using Polars.

Scans all parquet files and reports how many records have a non-null
value for the target field within the 'interpreted' struct.

Usage:
    python scripts/scan_all_fields_polars.py
"""

import time
import polars as pl

from config import DATA_DIR

# The target column we want to analyze for unique fields
TARGET_COLUMN = "interpreted"
TARGET_FIELD = "samplingProtocol"


def main():

    time_start = time.time()

    # count total samples
    total_count = pl.scan_parquet(str(DATA_DIR / "*.parquet")).select(pl.len()).collect().item()

    # count valid samples
    valid_count = (
        pl.scan_parquet(str(DATA_DIR / "*.parquet"))
        .select(
            pl.col(TARGET_COLUMN).struct.field(TARGET_FIELD).count().alias("cnt")
        )
        .collect(engine="streaming")
        .item()
    )

    time_end = time.time()
    print(f"Time taken: {time_end - time_start:.2f} seconds")

    print(f"Total samples: {total_count}")
    print(f"Valid samples: {valid_count}")

if __name__ == "__main__":
    main()
