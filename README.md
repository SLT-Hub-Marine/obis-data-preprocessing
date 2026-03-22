# OBIS Occurrence Analysis

Analysis of the [OBIS](https://obis.org/) (Ocean Biodiversity Information System) occurrence dataset.

## Data

The dataset consists of **6,779 GeoParquet files** (~62GB, ~248M records) partitioned by `dataset_id`.  
Data is not included in the repository — download via:

```bash
aws s3 sync s3://obis-products/occurrence data/
```

## Project Structure

```
data/           → Raw parquet files (one per OBIS dataset)
scripts/        → Analysis scripts + shared config
splits/         → Reproducible train/dev/test parquet splits
outputs/        → Generated reports and visualizations
```

## Scripts

| Script | Description |
|--------|-------------|
| `config.py` | Shared path configuration and column/field constants |
| `create_splits.py` | Create 80/10/10 train/dev/test splits by dataset_id (Polars) |
| `validate_splits.py` | Statistical validation of split distributions |
| `data_scale_assessment.py` | Field coverage report across all 212 Darwin Core fields |
| `export_to_csv.py` | Extract interested fields from interpreted column to CSV |
| `inspect_interpreted.py` | Pretty-print a single record's interpreted struct |
| `scan_all_fields_polars.py` | Count total/valid samples for a target field |
| `dataset.py` | PyTorch Dataset/DataLoader for marine occurrence data |
| `interactive_map.py` | Leaflet.js interactive map of classified occurrences |

## Dependencies

```
polars
tqdm
numpy
scipy
scikit-learn
torch
```

## Usage

All scripts should be run from the project root:

```bash
python scripts/create_splits.py
python scripts/validate_splits.py
python scripts/data_scale_assessment.py
python scripts/export_to_csv.py
python scripts/dataset.py
```
