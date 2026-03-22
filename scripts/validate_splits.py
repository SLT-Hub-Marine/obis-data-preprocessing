#!/usr/bin/env python3
"""
Validate train/validation/test splits for distributional equivalence.

For each key feature, runs:
  - KS test (numerical features): tests if two samples share the same distribution
  - Chi-squared test (categorical features): tests if category frequencies differ

Uses sampling (default 50k per split) because with millions of records even
trivial differences become "significant". Reports effect sizes alongside p-values.

Usage:
    python scripts/validate_splits.py [--sample-per-split 50000]
"""

import os
import glob
import time
import argparse
import warnings
from itertools import combinations

import numpy as np
import polars as pl
from scipy import stats

from config import DATA_DIR, SPLITS_DIR

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Features to validate — chosen for ML relevance
NUMERICAL_FEATURES = [
    ('source', 'decimalLatitude', 'Float64'),
    ('source', 'decimalLongitude', 'Float64'),
    ('source', 'minimumDepthInMeters', 'Float64'),
    ('source', 'maximumDepthInMeters', 'Float64'),
    ('source', 'coordinateUncertaintyInMeters', 'Float64'),
]

CATEGORICAL_FEATURES = [
    ('source', 'basisOfRecord'),
    ('source', 'kingdom'),
    ('source', 'phylum'),
    ('source', 'class'),
    ('source', 'order'),
    ('source', 'family'),
]


def load_split_files(split_name):
    """Load the list of filenames for a given split."""
    path = SPLITS_DIR / f'{split_name}.txt'
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def extract_features(files, sample_n, seed=42):
    """Extract numerical and categorical features from parquet files, with sampling.
    
    Handles heterogeneous schemas — only extracts fields that exist in each batch.
    """
    all_paths = [str(DATA_DIR / f) for f in files]

    num_fields = [field for _, field, _ in NUMERICAL_FEATURES]
    cat_fields = [field for _, field in CATEGORICAL_FEATURES]
    all_fields = num_fields + cat_fields

    BATCH_SIZE = 100
    dfs = []
    total_rows = 0

    for batch_start in range(0, len(all_paths), BATCH_SIZE):
        batch_paths = all_paths[batch_start:batch_start + BATCH_SIZE]
        try:
            # Check which fields exist in this batch's source struct
            schema = pl.read_parquet_schema(batch_paths[0])
            source_dtype = schema.get('source')
            if source_dtype is None or not hasattr(source_dtype, 'fields'):
                continue
            available = {f.name for f in source_dtype.fields}

            # Build expressions only for available fields
            exprs = []
            batch_num_fields = []
            batch_cat_fields = []
            for _, field, _ in NUMERICAL_FEATURES:
                if field in available:
                    exprs.append(
                        pl.col('source').struct.field(field)
                        .cast(pl.Float64, strict=False).alias(field)
                    )
                    batch_num_fields.append(field)
            for _, field in CATEGORICAL_FEATURES:
                if field in available:
                    exprs.append(
                        pl.col('source').struct.field(field)
                        .cast(pl.Utf8, strict=False).alias(field)
                    )
                    batch_cat_fields.append(field)

            if not exprs:
                continue

            batch_n = pl.scan_parquet(batch_paths).select(pl.len()).collect().item()
            total_rows += batch_n

            batch_df = (
                pl.scan_parquet(batch_paths)
                .select(exprs)
                .collect(engine='streaming')
            )

            # Add null columns for missing fields so concat works
            for field in all_fields:
                if field not in batch_df.columns:
                    if field in num_fields:
                        batch_df = batch_df.with_columns(pl.lit(None).cast(pl.Float64).alias(field))
                    else:
                        batch_df = batch_df.with_columns(pl.lit(None).cast(pl.Utf8).alias(field))

            # Reorder columns consistently
            batch_df = batch_df.select(all_fields)
            dfs.append(batch_df)

        except Exception as e:
            print(f"    Warning: batch {batch_start//BATCH_SIZE + 1} failed: {e}")
            continue

    if not dfs:
        return pl.DataFrame({f: [] for f in all_fields}), 0

    df = pl.concat(dfs)
    del dfs

    # Sample if needed
    if len(df) > sample_n:
        df = df.sample(n=sample_n, seed=seed)

    return df, total_rows


def ks_test_feature(df_a, df_b, feature):
    """Run KS test on a numerical feature between two splits."""
    a = df_a[feature].drop_nulls().to_numpy()
    b = df_b[feature].drop_nulls().to_numpy()

    if len(a) < 10 or len(b) < 10:
        return None

    stat, pval = stats.ks_2samp(a, b)
    return {'feature': feature, 'test': 'KS', 'statistic': stat, 'p_value': pval,
            'n_a': len(a), 'n_b': len(b)}


def chi2_test_feature(df_a, df_b, feature, max_categories=50):
    """Run chi-squared test on a categorical feature between two splits."""
    a = df_a[feature].drop_nulls().to_list()
    b = df_b[feature].drop_nulls().to_list()

    if len(a) < 10 or len(b) < 10:
        return None

    # Get union of categories, keep top N by frequency
    from collections import Counter
    counts_a = Counter(a)
    counts_b = Counter(b)
    all_cats = set(counts_a.keys()) | set(counts_b.keys())

    # Keep only top categories by combined frequency
    combined = Counter()
    combined.update(counts_a)
    combined.update(counts_b)
    top_cats = [c for c, _ in combined.most_common(max_categories)]

    # Build contingency vectors
    obs_a = np.array([counts_a.get(c, 0) for c in top_cats], dtype=float)
    obs_b = np.array([counts_b.get(c, 0) for c in top_cats], dtype=float)

    # Remove zero-count categories
    mask = (obs_a + obs_b) > 0
    obs_a = obs_a[mask]
    obs_b = obs_b[mask]

    if len(obs_a) < 2:
        return None

    # Cramér's V as effect size
    contingency = np.array([obs_a, obs_b])
    stat, pval, dof, _ = stats.chi2_contingency(contingency)
    n_total = contingency.sum()
    k = min(contingency.shape) - 1
    cramers_v = np.sqrt(stat / (n_total * k)) if k > 0 and n_total > 0 else 0

    return {'feature': feature, 'test': 'Chi2', 'statistic': stat, 'p_value': pval,
            'cramers_v': cramers_v, 'n_categories': len(obs_a),
            'n_a': int(obs_a.sum()), 'n_b': int(obs_b.sum())}


def main():
    parser = argparse.ArgumentParser(description='Validate split distributions')
    parser.add_argument('--sample-per-split', type=int, default=50000,
                        help='Max records to sample per split (default: 50000)')
    args = parser.parse_args()

    splits = {}
    split_rows = {}

    print("Loading splits...\n")
    for name in ['train', 'val', 'test']:
        files = load_split_files(name)
        t0 = time.time()
        df, n_rows = extract_features(files, sample_n=args.sample_per_split)
        elapsed = time.time() - t0
        splits[name] = df
        split_rows[name] = n_rows
        print(f"  {name:>5}: {n_rows:>12,} total records, "
              f"{len(df):>8,} sampled  ({elapsed:.1f}s)")

    total = sum(split_rows.values())
    print(f"\n  Total: {total:,} records")
    print(f"  Ratios: " + "  ".join(
        f"{name}={split_rows[name]/total:.1%}" for name in ['train', 'val', 'test']))

    # ── Summary Statistics ──────────────────────────────────────────────────
    split_names = ['train', 'val', 'test']

    print("\n" + "=" * 95)
    print("SUMMARY STATISTICS — NUMERICAL FEATURES")
    print("=" * 95)
    header = f"  {'Feature':<30}"
    for s in split_names:
        header += f" │ {'miss%':>6} {'mean':>10} {'std':>10} {'median':>10} {'min':>10} {'max':>10}"
    # Print per-split, one feature at a time (transposed for readability)
    for _, field, _ in NUMERICAL_FEATURES:
        print(f"\n  ── {field} ──")
        print(f"  {'Split':<8} {'miss%':>7} {'mean':>12} {'std':>12} "
              f"{'median':>12} {'min':>12} {'max':>12}")
        for s in split_names:
            col = splits[s][field]
            n_total = len(col)
            n_null = col.null_count()
            miss_pct = n_null / n_total * 100 if n_total > 0 else 0
            valid = col.drop_nulls()
            if len(valid) > 0:
                arr = valid.to_numpy()
                print(f"  {s:<8} {miss_pct:>6.1f}% {np.mean(arr):>12.2f} {np.std(arr):>12.2f} "
                      f"{np.median(arr):>12.2f} {np.min(arr):>12.2f} {np.max(arr):>12.2f}")
            else:
                print(f"  {s:<8} {miss_pct:>6.1f}%  {'(all null)':>62}")

    print("\n" + "=" * 95)
    print("SUMMARY STATISTICS — CATEGORICAL FEATURES")
    print("=" * 95)
    for _, field in CATEGORICAL_FEATURES:
        print(f"\n  ── {field} ──")
        print(f"  {'Split':<8} {'miss%':>7} {'unique':>8}   {'top 3 categories'}")
        for s in split_names:
            col = splits[s][field]
            n_total = len(col)
            n_null = col.null_count()
            miss_pct = n_null / n_total * 100 if n_total > 0 else 0
            valid = col.drop_nulls()
            n_unique = valid.n_unique()
            # Top 3 categories
            if len(valid) > 0:
                from collections import Counter
                counts = Counter(valid.to_list())
                top3 = counts.most_common(3)
                top3_str = ", ".join(f"{cat} ({cnt/len(valid)*100:.1f}%)" for cat, cnt in top3)
            else:
                top3_str = "(all null)"
            print(f"  {s:<8} {miss_pct:>6.1f}% {n_unique:>8}   {top3_str}")

    # ── Statistical Tests ───────────────────────────────────────────────────
    pairs = list(combinations(['train', 'val', 'test'], 2))

    print("\n" + "=" * 85)
    print("NUMERICAL FEATURES (Kolmogorov-Smirnov test)")
    print("=" * 85)
    print(f"  {'Feature':<35} {'Pair':<12} {'KS stat':>8} {'p-value':>10} {'Result':>8}")
    print(f"  {'─'*35} {'─'*12} {'─'*8} {'─'*10} {'─'*8}")

    num_results = []
    for _, field, _ in NUMERICAL_FEATURES:
        for a_name, b_name in pairs:
            result = ks_test_feature(splits[a_name], splits[b_name], field)
            if result is None:
                continue
            result['pair'] = f"{a_name}↔{b_name}"
            passed = result['p_value'] > 0.05
            marker = "  ✓" if passed else "  ✗"
            print(f"  {field:<35} {result['pair']:<12} "
                  f"{result['statistic']:>8.4f} {result['p_value']:>10.4f} {marker:>8}")
            num_results.append(result)

    print("\n" + "=" * 85)
    print("CATEGORICAL FEATURES (Chi-squared test)")
    print("=" * 85)
    print(f"  {'Feature':<35} {'Pair':<12} {'Cramér V':>8} {'p-value':>10} {'Result':>8}")
    print(f"  {'─'*35} {'─'*12} {'─'*8} {'─'*10} {'─'*8}")

    cat_results = []
    for _, field in CATEGORICAL_FEATURES:
        for a_name, b_name in pairs:
            result = chi2_test_feature(splits[a_name], splits[b_name], field)
            if result is None:
                continue
            result['pair'] = f"{a_name}↔{b_name}"
            # Use Cramér's V < 0.1 as practical significance threshold
            passed = result['cramers_v'] < 0.1
            marker = "  ✓" if passed else "  ✗"
            print(f"  {field:<35} {result['pair']:<12} "
                  f"{result['cramers_v']:>8.4f} {result['p_value']:>10.4f} {marker:>8}")
            cat_results.append(result)

    # Summary
    n_num_pass = sum(1 for r in num_results if r['p_value'] > 0.05)
    n_cat_pass = sum(1 for r in cat_results if r['cramers_v'] < 0.1)
    n_total = len(num_results) + len(cat_results)
    n_pass = n_num_pass + n_cat_pass

    print("\n" + "=" * 85)
    print("SUMMARY")
    print("=" * 85)
    print(f"  Numerical tests passed (p > 0.05):     {n_num_pass}/{len(num_results)}")
    print(f"  Categorical tests passed (V < 0.1):    {n_cat_pass}/{len(cat_results)}")
    print(f"  Overall:                               {n_pass}/{n_total}")

    if n_pass == n_total:
        print("\n  ✓ All splits are statistically equivalent — safe for ML.")
    else:
        print("\n  ⚠ Some features show distributional differences between splits.")
        print("    Consider stratified splitting if these features are important.")


if __name__ == '__main__':
    main()
