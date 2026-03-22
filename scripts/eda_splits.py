#!/usr/bin/env python3
"""
Exploratory Data Analysis on train/dev/test splits.

Profiles each split (row counts, null rates, summary statistics), runs
distributional-equivalence tests, and generates geospatial density maps.

Tests:
  - Kolmogorov-Smirnov (numerical features): H₀ = same continuous distribution
  - Chi-squared / Cramér's V (categorical features): effect-size measure

Uses sampling (default 50 000 per split) because with millions of rows even
trivial differences become "statistically significant".

Usage:
    python scripts/eda_splits.py
    python scripts/eda_splits.py --splits-dir splits/record_splits
    python scripts/eda_splits.py --sample 100000
"""

import argparse
import time
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow.compute as pc
from scipy import stats
from collections import Counter

from config import SPLITS_DIR, OUTPUTS_DIR, INTERESTED_FIELDS

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Feature lists (from interpreted struct) ─────────────────────────────────
NUMERICAL = ["bathymetry", "shoredistance", "decimalLatitude", "decimalLongitude"]
CATEGORICAL = [
    "samplingProtocol", "kingdom", "phylum", "class",
    "order", "family", "genus", "species",
]

SPLIT_NAMES = ["train", "dev", "test"]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_split_sample(name: str, sample_n: int, splits_dir: Path = SPLITS_DIR,
                      seed: int = 42):
    """Read a split parquet, extract interested fields, and sample.

    Returns a dict of {field: np.array or list} plus total row count.
    Streams row-groups to avoid loading the whole file.
    """
    path = splits_dir / f"{name}.parquet"
    pf = pq.ParquetFile(str(path))
    total_rows = pf.metadata.num_rows

    all_fields = NUMERICAL + CATEGORICAL
    # accumulate per-field lists
    accum = {f: [] for f in all_fields}

    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["interpreted"])
        interp = tbl.column("interpreted")
        for field in all_fields:
            col = pc.struct_field(interp, field)
            accum[field].append(col)
        del tbl

    # concatenate chunks (struct_field returns ChunkedArrays, flatten first)
    import pyarrow as pa
    arrays = {}
    for field in all_fields:
        chunks = []
        for ca in accum[field]:
            if isinstance(ca, pa.ChunkedArray):
                chunks.extend(ca.chunks)
            else:
                chunks.append(ca)
        arrays[field] = pa.concat_arrays(chunks)
    del accum

    # sample indices
    rng = np.random.default_rng(seed)
    n = len(arrays[all_fields[0]])
    if n > sample_n:
        idx = np.sort(rng.choice(n, size=sample_n, replace=False))
    else:
        idx = None

    result = {}
    for field in NUMERICAL:
        arr = arrays[field]
        if idx is not None:
            arr = arr.take(idx)
        # drop nulls → numpy
        valid = pc.filter(arr, pc.is_valid(arr))
        result[field] = valid.to_numpy(zero_copy_only=False).astype(np.float64)

    for field in CATEGORICAL:
        arr = arrays[field]
        if idx is not None:
            arr = arr.take(idx)
        valid = pc.filter(arr, pc.is_valid(arr))
        result[field] = valid.to_pylist()

    # also keep null counts (from full data, not sampled)
    null_counts = {}
    for field in all_fields:
        null_counts[field] = arrays[field].null_count
    del arrays

    return result, total_rows, null_counts, n


# ─────────────────────────────────────────────────────────────────────────────
# pretty printers
# ─────────────────────────────────────────────────────────────────────────────
def print_overview(split_data):
    """Print row counts and split ratios."""
    print("=" * 72)
    print("SPLIT OVERVIEW")
    print("=" * 72)
    total = sum(d["total_rows"] for d in split_data.values())
    for name in SPLIT_NAMES:
        d = split_data[name]
        pct = d["total_rows"] / total * 100 if total else 0
        print(f"  {name:<6}  {d['total_rows']:>14,} rows  ({pct:5.1f}%)")
    print(f"  {'total':<6}  {total:>14,} rows")


def print_numerical_stats(split_data):
    """Print summary statistics for numerical features."""
    print("\n" + "=" * 72)
    print("NUMERICAL FEATURES — SUMMARY STATISTICS")
    print("=" * 72)

    for field in NUMERICAL:
        print(f"\n  ── {field} ──")
        print(f"  {'split':<6} {'null%':>6} {'count':>10} {'mean':>12} "
              f"{'std':>12} {'median':>12} {'min':>12} {'max':>12}")
        for name in SPLIT_NAMES:
            d = split_data[name]
            arr = d["data"][field]
            nc = d["null_counts"][field]
            total = d["sampled_n"]
            null_pct = nc / d["total_rows"] * 100 if d["total_rows"] else 0
            if len(arr) > 0:
                print(f"  {name:<6} {null_pct:>5.1f}% {len(arr):>10,} "
                      f"{np.mean(arr):>12.2f} {np.std(arr):>12.2f} "
                      f"{np.median(arr):>12.2f} {np.min(arr):>12.2f} "
                      f"{np.max(arr):>12.2f}")
            else:
                print(f"  {name:<6} {null_pct:>5.1f}% {'(all null)':>66}")


def print_categorical_stats(split_data):
    """Print summary statistics for categorical features."""
    print("\n" + "=" * 72)
    print("CATEGORICAL FEATURES — SUMMARY STATISTICS")
    print("=" * 72)

    for field in CATEGORICAL:
        print(f"\n  ── {field} ──")
        print(f"  {'split':<6} {'null%':>6} {'unique':>8}   top 3 categories")
        for name in SPLIT_NAMES:
            d = split_data[name]
            vals = d["data"][field]
            nc = d["null_counts"][field]
            null_pct = nc / d["total_rows"] * 100 if d["total_rows"] else 0
            if vals:
                counts = Counter(vals)
                n_unique = len(counts)
                top3 = counts.most_common(3)
                top3_str = ", ".join(
                    f"{cat} ({cnt/len(vals)*100:.1f}%)" for cat, cnt in top3
                )
            else:
                n_unique = 0
                top3_str = "(all null)"
            print(f"  {name:<6} {null_pct:>5.1f}% {n_unique:>8}   {top3_str}")


# ─────────────────────────────────────────────────────────────────────────────
# statistical tests
# ─────────────────────────────────────────────────────────────────────────────
def ks_test(a, b):
    """Two-sample KS test. Returns (statistic, p_value) or None."""
    if len(a) < 20 or len(b) < 20:
        return None
    stat, pval = stats.ks_2samp(a, b)
    return stat, pval


def chi2_test(a_list, b_list, max_cats=50):
    """Chi-squared test + Cramér's V.  Returns (chi2, p, V, k) or None."""
    if len(a_list) < 20 or len(b_list) < 20:
        return None
    ca, cb = Counter(a_list), Counter(b_list)
    combined = Counter()
    combined.update(ca)
    combined.update(cb)
    top = [c for c, _ in combined.most_common(max_cats)]
    obs_a = np.array([ca.get(c, 0) for c in top], dtype=float)
    obs_b = np.array([cb.get(c, 0) for c in top], dtype=float)
    mask = (obs_a + obs_b) > 0
    obs_a, obs_b = obs_a[mask], obs_b[mask]
    if len(obs_a) < 2:
        return None
    cont = np.array([obs_a, obs_b])
    chi2, pval, dof, _ = stats.chi2_contingency(cont)
    n_total = cont.sum()
    k = min(cont.shape) - 1
    V = np.sqrt(chi2 / (n_total * k)) if k > 0 and n_total > 0 else 0
    return chi2, pval, V, len(obs_a)


def print_tests(split_data):
    """Run and print all distributional tests."""
    pairs = list(combinations(SPLIT_NAMES, 2))

    # ── Numerical ──
    print("\n" + "=" * 72)
    print("KS TESTS — NUMERICAL FEATURES")
    print("=" * 72)
    print(f"  {'feature':<22} {'pair':<14} {'KS stat':>8} {'p-value':>10} {'result':>8}")
    print(f"  {'─'*22} {'─'*14} {'─'*8} {'─'*10} {'─'*8}")

    num_results = []
    for field in NUMERICAL:
        for a_name, b_name in pairs:
            r = ks_test(split_data[a_name]["data"][field],
                        split_data[b_name]["data"][field])
            if r is None:
                continue
            stat, pval = r
            passed = pval > 0.05
            mark = "✓" if passed else "✗"
            label = f"{a_name}↔{b_name}"
            print(f"  {field:<22} {label:<14} {stat:>8.4f} {pval:>10.4f} {mark:>8}")
            num_results.append(passed)

    # ── Categorical ──
    print("\n" + "=" * 72)
    print("CHI-SQUARED TESTS — CATEGORICAL FEATURES")
    print("=" * 72)
    print(f"  {'feature':<22} {'pair':<14} {'Cramér V':>8} {'p-value':>10} {'result':>8}")
    print(f"  {'─'*22} {'─'*14} {'─'*8} {'─'*10} {'─'*8}")

    cat_results = []
    for field in CATEGORICAL:
        for a_name, b_name in pairs:
            r = chi2_test(split_data[a_name]["data"][field],
                          split_data[b_name]["data"][field])
            if r is None:
                continue
            chi2_stat, pval, V, k = r
            passed = V < 0.1  # Cramér's V < 0.1 → negligible effect
            mark = "✓" if passed else "✗"
            label = f"{a_name}↔{b_name}"
            print(f"  {field:<22} {label:<14} {V:>8.4f} {pval:>10.4f} {mark:>8}")
            cat_results.append(passed)

    # ── Summary ──
    n_num_pass = sum(num_results)
    n_cat_pass = sum(cat_results)
    n_total = len(num_results) + len(cat_results)
    n_pass = n_num_pass + n_cat_pass

    print("\n" + "=" * 72)
    print("EQUIVALENCE SUMMARY")
    print("=" * 72)
    print(f"  Numerical  (KS p > 0.05):     {n_num_pass}/{len(num_results)}")
    print(f"  Categorical (V < 0.1):         {n_cat_pass}/{len(cat_results)}")
    print(f"  Overall:                       {n_pass}/{n_total}")

    if n_pass == n_total:
        print("\n  ✓ All splits are distributionally equivalent — safe for ML.")
    else:
        n_fail = n_total - n_pass
        print(f"\n  ⚠ {n_fail} test(s) showed differences — inspect above for details.")
        print("    (Small effect sizes are expected with hash-based splitting.)")


# ─────────────────────────────────────────────────────────────────────────────
# geospatial visualisation
# ─────────────────────────────────────────────────────────────────────────────
def generate_geo_map(split_data, out_path: Path):
    """Generate a 3-panel hexbin density map of lat/lon per split."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle("Geographic Distribution by Split (hexbin density, log scale)",
                 fontsize=14, fontweight="bold", y=1.02)

    colors = {"train": "YlOrRd", "dev": "YlGnBu", "test": "PuBuGn"}

    for ax, name in zip(axes, SPLIT_NAMES):
        d = split_data[name]
        lon = d["data"]["decimalLongitude"]
        lat = d["data"]["decimalLatitude"]

        if len(lon) > 0 and len(lat) > 0:
            hb = ax.hexbin(
                lon, lat, gridsize=80, cmap=colors[name],
                mincnt=1, bins="log", extent=[-180, 180, -90, 90],
            )
            fig.colorbar(hb, ax=ax, shrink=0.7, label="log₁₀(count)")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax.set_title(f"{name}  ({d['total_rows']:,} rows)", fontsize=12)
        ax.set_xlabel("Longitude")
        if name == SPLIT_NAMES[0]:
            ax.set_ylabel("Latitude")

        # lightweight grid
        ax.axhline(0, color="grey", lw=0.4, ls="--")
        ax.axvline(0, color="grey", lw=0.4, ls="--")
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  📍 Geospatial map saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EDA on train/dev/test splits")
    parser.add_argument("--sample", type=int, default=50_000,
                        help="Max rows to sample per split (default: 50000)")
    parser.add_argument("--splits-dir", type=str, default=None,
                        help="Path to splits directory (default: splits/)")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir) if args.splits_dir else SPLITS_DIR
    # resolve relative to project root if not absolute
    if not splits_dir.is_absolute():
        from config import PROJECT_ROOT
        splits_dir = PROJECT_ROOT / splits_dir

    print(f"Splits directory: {splits_dir}")
    print(f"Sampling up to {args.sample:,} rows per split.\n")

    split_data = {}
    for name in SPLIT_NAMES:
        t0 = time.time()
        data, total_rows, null_counts, sampled_n = load_split_sample(
            name, sample_n=args.sample, splits_dir=splits_dir
        )
        elapsed = time.time() - t0
        split_data[name] = {
            "data": data,
            "total_rows": total_rows,
            "null_counts": null_counts,
            "sampled_n": sampled_n,
        }
        print(f"  Loaded {name}: {total_rows:>12,} rows, "
              f"sampled {min(sampled_n, args.sample):,}  ({elapsed:.1f}s)")

    print_overview(split_data)
    print_numerical_stats(split_data)
    print_categorical_stats(split_data)
    print_tests(split_data)

    # Determine output filename based on splits dir
    suffix = splits_dir.name if splits_dir.name != "splits" else "dataset_splits"
    map_path = OUTPUTS_DIR / f"eda_geo_map_{suffix}.png"
    generate_geo_map(split_data, map_path)

    print()


if __name__ == "__main__":
    main()
