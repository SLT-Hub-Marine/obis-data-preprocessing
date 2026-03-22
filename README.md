# OBIS Occurrence Analysis

Analysis of the [OBIS](https://obis.org/) (Ocean Biodiversity Information System) occurrence dataset.

## Data

The dataset consists of **6,779 GeoParquet files** (~62GB, ~41M filtered records) partitioned by `dataset_id`.  
Data is not included in the repository — download via:

```bash
aws s3 sync s3://obis-products/occurrence data/
```

## Project Structure

```
data/                       → Raw parquet files (one per OBIS dataset)
scripts/                    → Analysis scripts + shared config
splits/                     → Dataset-level train/dev/test splits
splits/record_splits/       → Record-level train/dev/test splits
outputs/                    → Generated reports and visualizations
```

## Splitting Strategies

We provide two approaches for creating train/dev/test splits, each answering a different question.

### Dataset-Level Splitting (`create_splits.py`)

Each parquet file (= one OBIS dataset/survey) is **assigned whole** to a split by hashing its `dataset_id`. All records from a given survey end up in the same split.

```bash
python scripts/create_splits.py
```

**Why:** Records within a single survey are highly correlated — same location, time period, sampling method, and species pool. Keeping them together prevents **data leakage** and tests whether a model can generalise to *entirely new surveys*.

**Trade-off:** Splits have different geographic and taxonomic compositions because each survey covers a specific region/taxon.

### Record-Level Splitting (`create_splits_record.py`)

Each individual row's `_id` (UUID) is hashed to independently assign it to a split. Records from the same survey are spread across all splits.

```bash
python scripts/create_splits_record.py
```

**Why:** Produces **distributionally balanced** splits — identical geographic coverage, taxonomic mix, and exact 80/10/10 row ratios. Useful for benchmarking and hyperparameter tuning.

**Trade-off:** Allows data leakage — the model sees records from the same survey in both training and evaluation, inflating test metrics.

## EDA Results

### Split Sizes

| | Dataset-Level | Record-Level |
|---|---:|---:|
| Train | 28,158,921 (68.3%) | 32,935,709 (80.0%) |
| Dev | 6,131,663 (14.9%) | 4,118,578 (10.0%) |
| Test | 6,880,361 (16.7%) | 4,116,658 (10.0%) |
| **Total** | **41,170,945** | **41,170,945** |

> Dataset-level targets 80/10/10 by *file count*, but datasets have very different row counts, so actual row ratios deviate. Record-level achieves exact 80/10/10 by row.

### Distributional Equivalence Tests

| | Dataset-Level | Record-Level |
|---|:-:|:-:|
| Numerical (KS test, p > 0.05) | 0/12 ✗ | 12/12 ✓ |
| Categorical (Cramér's V < 0.1) | 1/24 ✗ | 24/24 ✓ |
| **Overall** | **1/36** | **36/36** |

- **Dataset-level:** Splits have very different distributions across all features — expected and correct, since each split contains different surveys.
- **Record-level:** All 36 tests pass — splits are statistically identical.

Run the EDA yourself:

```bash
python scripts/eda_splits.py                                    # dataset-level
python scripts/eda_splits.py --splits-dir splits/record_splits  # record-level
```

## Scripts

| Script | Description |
|--------|-------------|
| `config.py` | Shared path configuration and column/field constants |
| `create_splits.py` | Dataset-level 80/10/10 splitting (no data leakage) |
| `create_splits_record.py` | Record-level 80/10/10 splitting (balanced distributions) |
| `eda_splits.py` | EDA with summary stats, equivalence tests, and geospatial maps |
| `validate_splits.py` | Legacy split validation (uses `.txt` file lists) |
| `data_scale_assessment.py` | Field coverage report across all 212 Darwin Core fields |
| `export_to_csv.py` | Extract interested fields from interpreted column to CSV |
| `inspect_interpreted.py` | Pretty-print a single record's interpreted struct |
| `scan_all_fields_polars.py` | Count total/valid samples for a target field |
| `dataset.py` | PyTorch Dataset/DataLoader for marine occurrence data |
| `interactive_map.py` | Leaflet.js interactive map of classified occurrences |
| `export_geojson.py` | Export parquet data to GeoJSON lines for tile generation |

## Dependencies

```
pyarrow
polars
numpy
scipy
matplotlib
tqdm
scikit-learn
torch
```

## Usage

All scripts should be run from the project root:

```bash
# Download data
aws s3 sync s3://obis-products/occurrence data/

# Create splits
python scripts/create_splits.py            # dataset-level
python scripts/create_splits_record.py     # record-level

# Run EDA
python scripts/eda_splits.py
python scripts/eda_splits.py --splits-dir splits/record_splits
```
