#!/usr/bin/env python3
"""
Generate comparison charts for the README.

Produces:
  - docs/split_sizes.png         – side-by-side bar chart of split row counts
  - docs/taxonomy_comparison.png – phylum distribution per split, both approaches
  - docs/geo_dataset_splits.png  – geospatial hexbin map (dataset-level)
  - docs/geo_record_splits.png   – geospatial hexbin map (record-level)
  - docs/equivalence_tests.png   – pass/fail heatmap of distributional tests

Usage:
    python scripts/generate_readme_charts.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pyarrow.parquet as pq
import pyarrow.compute as pc
from collections import Counter
from pathlib import Path

from config import SPLITS_DIR, PROJECT_ROOT

DOCS_DIR = PROJECT_ROOT / "docs"
RECORD_DIR = SPLITS_DIR / "record_splits"
SPLIT_NAMES = ["train", "dev", "test"]

# ── Colour palette ──────────────────────────────────────────────────────────
COLORS = {"train": "#4C72B0", "dev": "#55A868", "test": "#C44E52"}
APPROACH_COLORS = {"Dataset-Level": "#4C72B0", "Record-Level": "#DD8452"}


def sample_field(splits_dir, field, n=50_000, seed=42):
    """Sample a categorical field from each split."""
    import pyarrow as pa
    rng = np.random.default_rng(seed)
    result = {}
    for name in SPLIT_NAMES:
        pf = pq.ParquetFile(str(splits_dir / f"{name}.parquet"))
        chunks = []
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["interpreted"])
            col = pc.struct_field(tbl.column("interpreted"), field)
            if isinstance(col, pa.ChunkedArray):
                chunks.extend(col.chunks)
            else:
                chunks.append(col)
            del tbl
        arr = pa.concat_arrays(chunks)
        # sample
        if len(arr) > n:
            idx = rng.choice(len(arr), n, replace=False)
            arr = arr.take(idx)
        valid = pc.filter(arr, pc.is_valid(arr))
        result[name] = valid.to_pylist()
    return result


def sample_coords(splits_dir, n=50_000, seed=42):
    """Sample lat/lon from each split."""
    import pyarrow as pa
    rng = np.random.default_rng(seed)
    result = {}
    for name in SPLIT_NAMES:
        pf = pq.ParquetFile(str(splits_dir / f"{name}.parquet"))
        lat_chunks, lon_chunks = [], []
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["interpreted"])
            interp = tbl.column("interpreted")
            lat = pc.struct_field(interp, "decimalLatitude")
            lon = pc.struct_field(interp, "decimalLongitude")
            for c, lst in [(lat, lat_chunks), (lon, lon_chunks)]:
                if isinstance(c, pa.ChunkedArray):
                    lst.extend(c.chunks)
                else:
                    lst.append(c)
            del tbl
        lat_arr = pa.concat_arrays(lat_chunks)
        lon_arr = pa.concat_arrays(lon_chunks)
        if len(lat_arr) > n:
            idx = np.sort(rng.choice(len(lat_arr), n, replace=False))
            lat_arr = lat_arr.take(idx)
            lon_arr = lon_arr.take(idx)
        # drop nulls from both together
        mask = pc.and_(pc.is_valid(lat_arr), pc.is_valid(lon_arr))
        result[name] = {
            "lat": pc.filter(lat_arr, mask).to_numpy(zero_copy_only=False).astype(np.float64),
            "lon": pc.filter(lon_arr, mask).to_numpy(zero_copy_only=False).astype(np.float64),
        }
    return result


# ── Chart 1: Split Sizes ───────────────────────────────────────────────────
def chart_split_sizes():
    dataset_rows = {"train": 28_158_921, "dev": 6_131_663, "test": 6_880_361}
    record_rows = {"train": 32_935_709, "dev": 4_118_578, "test": 4_116_658}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (title, data) in zip(axes, [
        ("Dataset-Level Splits", dataset_rows),
        ("Record-Level Splits", record_rows),
    ]):
        total = sum(data.values())
        bars = ax.bar(
            data.keys(),
            [v / 1e6 for v in data.values()],
            color=[COLORS[s] for s in data.keys()],
            edgecolor="white", linewidth=1.5
        )
        for bar, (name, v) in zip(bars, data.items()):
            pct = v / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v/1e6:.1f}M\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Rows (millions)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(data.values()) / 1e6 * 1.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Split Size Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(str(DOCS_DIR / "split_sizes.png"), dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  ✓ split_sizes.png")


# ── Chart 2: Taxonomy Comparison ───────────────────────────────────────────
def chart_taxonomy(dataset_phyla, record_phyla):
    # Get top 8 phyla by combined frequency
    combined = Counter()
    for name in SPLIT_NAMES:
        combined.update(dataset_phyla[name])
        combined.update(record_phyla[name])
    top_phyla = [p for p, _ in combined.most_common(8)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (title, data) in zip(axes, [
        ("Dataset-Level", dataset_phyla),
        ("Record-Level", record_phyla),
    ]):
        x = np.arange(len(top_phyla))
        width = 0.25
        for i, name in enumerate(SPLIT_NAMES):
            counts = Counter(data[name])
            total = len(data[name])
            pcts = [counts.get(p, 0) / total * 100 for p in top_phyla]
            ax.bar(x + i * width, pcts, width, label=name,
                   color=COLORS[name], edgecolor="white", linewidth=0.5)

        ax.set_xticks(x + width)
        ax.set_xticklabels(top_phyla, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Phylum Distribution by Split", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(str(DOCS_DIR / "taxonomy_comparison.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ✓ taxonomy_comparison.png")


# ── Chart 3: Geospatial Maps ──────────────────────────────────────────────
def chart_geo(coords, out_name, title_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    cmaps = {"train": "YlOrRd", "dev": "YlGnBu", "test": "PuBuGn"}

    for ax, name in zip(axes, SPLIT_NAMES):
        d = coords[name]
        hb = ax.hexbin(d["lon"], d["lat"], gridsize=80, cmap=cmaps[name],
                       mincnt=1, bins="log", extent=[-180, 180, -90, 90])
        fig.colorbar(hb, ax=ax, shrink=0.7, label="log₁₀(count)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude")
        if name == "train":
            ax.set_ylabel("Latitude")
        ax.axhline(0, color="grey", lw=0.3, ls="--")
        ax.axvline(0, color="grey", lw=0.3, ls="--")
        ax.grid(True, alpha=0.1)

    fig.suptitle(f"{title_prefix} — Geographic Distribution",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(str(DOCS_DIR / out_name), dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out_name}")


# ── Chart 4: Equivalence Test Heatmap ─────────────────────────────────────
def chart_equivalence():
    # Hardcoded from EDA results
    features = [
        "bathymetry", "shoredistance", "decimalLatitude", "decimalLongitude",
        "samplingProtocol", "kingdom", "phylum", "class",
        "order", "family", "genus", "species",
    ]
    pairs = ["train↔dev", "train↔test", "dev↔test"]

    # 1 = pass, 0 = fail
    dataset_results = np.array([
        [0, 0, 0],  # bathymetry
        [0, 0, 0],  # shoredistance
        [0, 0, 0],  # decimalLatitude
        [0, 0, 0],  # decimalLongitude
        [0, 0, 0],  # samplingProtocol
        [0, 1, 0],  # kingdom
        [0, 0, 0],  # phylum
        [0, 0, 0],  # class
        [0, 0, 0],  # order
        [0, 0, 0],  # family
        [0, 0, 0],  # genus
        [0, 0, 0],  # species
    ])

    record_results = np.ones_like(dataset_results)  # all pass

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    cmap = matplotlib.colors.ListedColormap(["#E74C3C", "#2ECC71"])

    for ax, (title, data) in zip(axes, [
        ("Dataset-Level (1/36)", dataset_results),
        ("Record-Level (36/36)", record_results),
    ]):
        ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pairs, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Add ✓/✗ labels
        for i in range(len(features)):
            for j in range(len(pairs)):
                mark = "✓" if data[i, j] else "✗"
                color = "white" if not data[i, j] else "white"
                ax.text(j, i, mark, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

    fig.suptitle("Distributional Equivalence Tests", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(DOCS_DIR / "equivalence_tests.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ✓ equivalence_tests.png")


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating charts...\n")

    # Chart 1: Split sizes (no data loading needed)
    chart_split_sizes()

    # Chart 4: Equivalence heatmap (hardcoded)
    chart_equivalence()

    # Charts requiring data loading
    print("\n  Loading dataset-level phylum data...")
    ds_phyla = sample_field(SPLITS_DIR, "phylum")
    print("  Loading record-level phylum data...")
    rec_phyla = sample_field(RECORD_DIR, "phylum")
    chart_taxonomy(ds_phyla, rec_phyla)

    print("\n  Loading dataset-level coordinates...")
    ds_coords = sample_coords(SPLITS_DIR)
    chart_geo(ds_coords, "geo_dataset_splits.png", "Dataset-Level Splits")

    print("  Loading record-level coordinates...")
    rec_coords = sample_coords(RECORD_DIR)
    chart_geo(rec_coords, "geo_record_splits.png", "Record-Level Splits")

    print(f"\nDone — charts saved to {DOCS_DIR}/")


if __name__ == "__main__":
    main()
