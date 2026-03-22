#!/usr/bin/env python3
"""
Create reproducible train/dev/test splits.

Two splitting strategies are available:

  Dataset-level (default):
    Hashes each parquet filename (dataset_id) to assign entire datasets
    to a split. Prevents data leakage between correlated records.

  Record-level (--by-record):
    Hashes each row's _id to independently assign records.
    Produces distributionally balanced splits.

Split ratios (bucket 0-99):
  - Train: 80%  (buckets 0-79)
  - Dev:   10%  (buckets 80-89)
  - Test:  10%  (buckets 90-99)

By default, only rows where ALL interested fields are non-null are kept.
Use --keep-missing to include all rows.

Outputs:
  Dataset-level → splits/train.parquet, splits/dev.parquet, splits/test.parquet
  Record-level  → splits/record_splits/train.parquet, …

Usage:
    python scripts/create_splits.py                # dataset-level
    python scripts/create_splits.py --by-record     # record-level
    python scripts/create_splits.py --keep-missing  # include nulls
"""

import argparse
import hashlib
import sys
import time
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from config import DATA_DIR, SPLITS_DIR, INTERESTED_FIELDS

TARGET_COLUMN = "interpreted"
TARGET_FIELDS = INTERESTED_FIELDS

# Exclude 'extensions' and 'source' — their nested struct schemas vary
# across dataset files, causing ParquetWriter schema mismatches.
KEEP_COLUMNS = [
    "_id", "_event_id", "_occurrence_id", "dataset_id", "node_ids",
    "interpreted", "missing", "invalid", "flags",
    "dropped", "absence", "tags", "geometry",
]


def bucket_for_file(path: Path) -> int:
    """Deterministic bucket 0-99 from a filename (dataset_id)."""
    dataset_id = path.stem
    h = int(hashlib.sha256(dataset_id.encode()).hexdigest(), 16)
    return h % 100


def compute_row_buckets(id_array: pa.Array) -> pa.Array:
    """Hash each _id string to a bucket 0-99."""
    buckets = []
    for val in id_array.to_pylist():
        h = int(hashlib.sha256(val.encode()).hexdigest()[:16], 16)
        buckets.append(h % 100)
    return pa.array(buckets, type=pa.uint8())


def apply_null_filter(table, keep_missing: bool):
    """Filter out rows where any interested field is null (unless keep_missing)."""
    if keep_missing:
        return table
    interp = table.column(TARGET_COLUMN)
    mask = None
    for field_name in TARGET_FIELDS:
        col = pc.struct_field(interp, field_name)
        not_null = pc.is_valid(col)
        mask = not_null if mask is None else pc.and_(mask, not_null)
    if mask is not None:
        table = pc.filter(table, mask)
    return table


def split_dataset_level(all_files, out_dir, keep_missing, logger):
    """Assign entire files to splits by hashing the dataset_id."""
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

    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train", "dev", "test"):
        files = split_files[name]
        out_path = out_dir / f"{name}.parquet"
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
                    table = apply_null_filter(table, keep_missing)

                    n = table.num_rows
                    if n > 0:
                        if writer is None:
                            writer = pq.ParquetWriter(str(out_path), table.schema)
                        writer.write_table(table)
                    file_rows += n
                    del table

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

    return {name: len(split_files[name]) for name in ("train", "dev", "test")}


def split_record_level(all_files, out_dir, keep_missing, logger):
    """Assign individual rows to splits by hashing each _id."""
    out_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[str, pq.ParquetWriter | None] = {
        "train": None, "dev": None, "test": None
    }
    row_counts = {"train": 0, "dev": 0, "test": 0}

    try:
        for fi, fpath in enumerate(all_files, 1):
            pf = pq.ParquetFile(str(fpath))

            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=KEEP_COLUMNS)
                table = apply_null_filter(table, keep_missing)

                if table.num_rows == 0:
                    del table
                    continue

                buckets = compute_row_buckets(table.column("_id"))

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
                            out_path = out_dir / f"{split_name}.parquet"
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
        for w in writers.values():
            if w is not None:
                w.close()

    return row_counts


def main():
    parser = argparse.ArgumentParser(
        description="Create reproducible train/dev/test splits"
    )
    parser.add_argument(
        "--by-record",
        action="store_true",
        help="Split by individual record _id instead of dataset "
             "(balanced distributions, but allows data leakage)",
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
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    time_start = time.time()

    mode = "record-level" if args.by_record else "dataset-level"
    logger.info(f"Splitting mode: {mode}")
    if args.keep_missing:
        logger.info("--keep-missing: all rows will be included")
    else:
        logger.info("Filtering to rows where all interested fields are non-null")

    all_files = sorted(DATA_DIR.glob("*.parquet"))
    logger.info(f"Found {len(all_files):,} parquet files in {DATA_DIR}")

    if args.by_record:
        out_dir = SPLITS_DIR / "record_splits"
        row_counts = split_record_level(all_files, out_dir, args.keep_missing, logger)
    else:
        out_dir = SPLITS_DIR
        split_dataset_level(all_files, out_dir, args.keep_missing, logger)
        row_counts = None  # logged inline

    time_end = time.time()
    logger.info(f"Done — splits saved to {out_dir}/")
    if row_counts:
        total = sum(row_counts.values())
        for name in ("train", "dev", "test"):
            pct = row_counts[name] / total * 100 if total else 0
            logger.info(f"  {name}: {row_counts[name]:,} rows ({pct:.1f}%)")
    logger.info(f"Time taken: {time_end - time_start:.1f}s")


if __name__ == "__main__":
    main()
