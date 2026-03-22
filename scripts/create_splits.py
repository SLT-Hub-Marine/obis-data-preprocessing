#!/usr/bin/env python3
"""
Create reproducible train/dev/test splits using Polars.

Each parquet file in data/ corresponds to one dataset_id (the filename
*is* the dataset_id).  We hash the filename to deterministically assign
each file to a split, then stream only the relevant files into each
output — no full-dataset scan required.

Split ratios (by dataset_id hash bucket 0-99):
  - Train: 80%  (buckets 0-79)
  - Dev:   10%  (buckets 80-89)
  - Test:  10%  (buckets 90-99)

By default, only rows where ALL interested fields are non-null are
included. Use --keep-missing to include all rows regardless.

Outputs:
  - splits/train.parquet
  - splits/dev.parquet
  - splits/test.parquet

Usage:
    python scripts/create_splits.py
    python scripts/create_splits.py --keep-missing
"""

import argparse
import hashlib
import sys
import time
import logging
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from config import DATA_DIR, SPLITS_DIR, INTERESTED_FIELDS

# The column and fields we care about
TARGET_COLUMN = "interpreted"
TARGET_FIELDS = INTERESTED_FIELDS

# Columns to keep in the output splits.  We exclude 'extensions' and 'source'
# because their nested struct schemas vary across dataset files, causing
# ParquetWriter schema mismatches.
KEEP_COLUMNS = [
    "_id", "_event_id", "_occurrence_id", "dataset_id", "node_ids",
    "interpreted", "missing", "invalid", "flags",
    "dropped", "absence", "tags", "geometry",
]


def bucket_for_file(path: Path) -> int:
    """Return a deterministic bucket 0-99 from a filename (dataset_id)."""
    dataset_id = path.stem  # e.g. "00017595-e015-4ec6-bf8a-b013e0dca521"
    h = int(hashlib.sha256(dataset_id.encode()).hexdigest(), 16)
    return h % 100


def main():
    parser = argparse.ArgumentParser(
        description="Create reproducible train/dev/test splits"
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Keep all rows, even those with missing interested fields "
             "(default: drop rows where any interested field is null)",
    )
    args = parser.parse_args()

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    time_start = time.time()

    if args.keep_missing:
        logger.info("--keep-missing enabled: all rows will be included")
    else:
        logger.info("Filtering to rows where all interested fields are non-null")

    # --- Clean up leftover temp batch files from any previous crashed run ---
    stale = sorted(SPLITS_DIR.glob("_*_batch_*.parquet"))
    if stale:
        logger.info(f"Cleaning up {len(stale)} leftover temp batch files")
        for bf in stale:
            bf.unlink()

    # --- Step 1: Assign each parquet file to a split by hashing its name ---
    all_files = sorted(DATA_DIR.glob("*.parquet"))
    logger.info(f"Found {len(all_files):,} parquet files in {DATA_DIR}")

    split_files: dict[str, list[Path]] = {"train": [], "dev": [], "test": []}
    for f in all_files:
        b = bucket_for_file(f)
        if b < 80:
            split_files["train"].append(f)
        elif b < 90:
            split_files["dev"].append(f)
        else:
            split_files["test"].append(f)

    for name in ("train", "dev", "test"):
        logger.info(f"  {name}: {len(split_files[name]):,} files")

    # --- Step 2: Write each split by streaming row groups via PyArrow ---
    #
    # Read each source file one row group at a time, filter in Arrow,
    # and immediately write to the output.  Peak memory ≈ 1 row group.
    #
    import pyarrow.compute as pc

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("train", "dev", "test"):
        files = split_files[name]
        out_path = SPLITS_DIR / f"{name}.parquet"
        logger.info(f"Writing {name} split ({len(files):,} files) → {out_path}")

        if not files:
            logger.warning(f"  No files for {name} split, skipping")
            continue

        total_rows = 0
        writer: pq.ParquetWriter | None = None

        try:
            for fi, fpath in enumerate(files, 1):
                pf = pq.ParquetFile(str(fpath))
                file_rows = 0

                for rg_idx in range(pf.metadata.num_row_groups):
                    table = pf.read_row_group(rg_idx, columns=KEEP_COLUMNS)

                    # Apply non-null filter unless --keep-missing
                    if not args.keep_missing:
                        interp = table.column(TARGET_COLUMN)
                        mask = None
                        for field_name in TARGET_FIELDS:
                            # struct_field returns the nested column
                            col = pc.struct_field(interp, field_name)
                            not_null = pc.is_valid(col)
                            mask = not_null if mask is None else pc.and_(mask, not_null)
                        if mask is not None:
                            table = pc.filter(table, mask)

                    n = table.num_rows
                    if n > 0:
                        if writer is None:
                            writer = pq.ParquetWriter(str(out_path), table.schema)
                        writer.write_table(table)
                    file_rows += n
                    del table  # free immediately

                total_rows += file_rows

                if fi % 100 == 0 or fi == len(files):
                    logger.info(
                        f"  {name}: {fi:,}/{len(files):,} files "
                        f"({total_rows:,} rows so far)"
                    )
        finally:
            if writer is not None:
                writer.close()

        logger.info(f"  {name} complete: {total_rows:,} rows total")

    time_end = time.time()
    logger.info(f"Done — splits saved to {SPLITS_DIR}/")
    logger.info(f"Time taken: {time_end - time_start:.1f}s")


if __name__ == "__main__":
    main()
