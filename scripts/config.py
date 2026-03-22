"""
Shared configuration for the OBIS occurrence project.

All scripts import paths from here so there's a single source of truth.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
SPLITS_DIR = PROJECT_ROOT / 'splits'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'

# Top-level parquet columns in the OBIS occurrence files
PARQUET_COLUMNS = [
    "_id",
    "_event_id",
    "_occurrence_id",
    "dataset_id",
    "node_ids",
    "source",
    "interpreted",
    "extensions",
    "missing",
    "invalid",
    "flags",
    "dropped",
    "absence",
    "tags",
    "geometry",
]

# Fields of interest within the 'interpreted' struct column
INTERESTED_FIELDS = [
    "samplingProtocol",

    "bathymetry",
    "shoredistance",

    "decimalLatitude",
    "decimalLongitude",
    "geodeticDatum",

    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",

    "scientificName",

    "occurrenceID",
]
