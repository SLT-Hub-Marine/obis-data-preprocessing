#!/usr/bin/env python3
"""
Create reproducible train/dev/test splits by hashing each record's _id.

Unlike create_splits.py (which hashes by dataset_id), this script assigns
individual records to splits, producing distributionally balanced partitions.

Each row's `_id` (a UUID, always non-null) is hashed to a bucket 0-99:
  - Train: 80%  (buckets 0-79)
  - Dev:   10%  (buckets 80-89)
  - Test:  10%  (buckets 90-99)

Streams row groups via PyArrow for constant memory usage.

Outputs:
  - splits/record_splits/train.parquet
  - splits/record_splits/dev.parquet
  - splits/record_splits/test.parquet

Usage:
    python scripts/create_splits_record.py
    python scripts/create_splits_record.py --keep-missing
"""

import argparse
import hashlib
import sys
import time
import logging

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from config import DATA_DIR, SPLITS_DIR, INTERESTED_FIELDS

TARGET_COLUMN = "interpreted"
TARGET_FIELDS = INTERESTED_FIELDS

# Columns to keep — same as create_splits.py
KEEP_COLUMNS = [
    "_id", "_event_id", "_occurrence_id", "dataset_id", "node_ids",
    "interpreted", "missing", "invalid", "flags",
    "dropped", "absence", "tags", "geometry",
]

OUT_DIR = SPLITS_DIR / "record_splits"


def compute_buckets(id_array: pa.Array) -> pa.Array:
    """Hash each _id string to a bucket 0-99.
    
    Uses SHA-256 truncated to 8 bytes for speed.
    Returns a UInt8 array of bucket indices.
    """
    buckets = []
    for val in id_array.to_pylist():
        h = int(hashlib.sha256(val.encode()).hexdigest()[:16], 16)
        buckets.append(h % 100)
    return pa.array(buckets, type=pa.uint8())


def main():
    parser = argparse.ArgumentParser(
        description="Create reproducible train/dev/test splits (record-level)"
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Keep all rows, even those with missing interested fields",
    )
    args = parser.parse_args()

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

    all_files = sorted(DATA_DIR.glob("*.parquet"))
    logger.info(f"Found {len(all_files):,} parquet files in {DATA_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Open 3 writers simultaneously — one per split
    writers: dict[str, pq.ParquetWriter | None] = {
        "train": None, "dev": None, "test": None
    }
    row_counts = {"train": 0, "dev": 0, "test": 0}

    try:
        for fi, fpath in enumerate(all_files, 1):
            pf = pq.ParquetFile(str(fpath))

            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=KEEP_COLUMNS)

                # Apply non-null filter unless --keep-missing
                if not args.keep_missing:
                    interp = table.column(TARGET_COLUMN)
                    mask = None
                    for field_name in TARGET_FIELDS:
                        col = pc.struct_field(interp, field_name)
                        not_null = pc.is_valid(col)
                        mask = not_null if mask is None else pc.and_(mask, not_null)
                    if mask is not None:
                        table = pc.filter(table, mask)

                if table.num_rows == 0:
                    del table
                    continue

                # Compute bucket for each row
                buckets = compute_buckets(table.column("_id"))

                # Split table into 3 sub-tables by bucket range
                for split_name, lo, hi in [
                    ("train", 0, 79), ("dev", 80, 89), ("test", 90, 99)
                ]:
                    split_mask = pc.and_(
                        pc.greater_equal(buckets, pa.scalar(lo, pa.uint8())),
                        pc.less_equal(buckets, pa.scalar(hi, pa.uint8()))
                    )
                    sub = pc.filter(table, split_mask)
                    n = sub.num_rows
                    if n > 0:
                        if writers[split_name] is None:
                            out_path = OUT_DIR / f"{split_name}.parquet"
                            writers[split_name] = pq.ParquetWriter(
                                str(out_path), sub.schema
                            )
                        writers[split_name].write_table(sub)
                        row_counts[split_name] += n
                    del sub

                del table, buckets

            if fi % 100 == 0 or fi == len(all_files):
                total = sum(row_counts.values())
                logger.info(
                    f"  {fi:,}/{len(all_files):,} files — "
                    f"train={row_counts['train']:,} "
                    f"dev={row_counts['dev']:,} "
                    f"test={row_counts['test']:,} "
                    f"(total={total:,})"
                )
    finally:
        for name, w in writers.items():
            if w is not None:
                w.close()

    total = sum(row_counts.values())
    time_end = time.time()
    logger.info(f"Done — record-level splits saved to {OUT_DIR}/")
    for name in ("train", "dev", "test"):
        pct = row_counts[name] / total * 100 if total else 0
        logger.info(f"  {name}: {row_counts[name]:,} rows ({pct:.1f}%)")
    logger.info(f"Time taken: {time_end - time_start:.1f}s")


if __name__ == "__main__":
    main()
