#!/usr/bin/env python3
"""
OBIS Data Scale Assessment 

This script analyzes the OBIS parquet dataset to:
1. Count total files and exact total records
2. Measure coverage of key free-text fields
3. Output a summary report

Usage:
    python scripts/data_scale_assessment.py
"""

import polars as pl
import os
import glob
import time
import gc

from config import DATA_DIR, OUTPUTS_DIR

TARGET_FIELDS = [
    "acceptedNameUsage",
    "acceptedNameUsageID",
    "accessRights",
    "associatedMedia",
    "associatedOccurrences",
    "associatedOrganisms",
    "associatedReferences",
    "associatedSequences",
    "associatedTaxa",
    "basisOfRecord",
    "bed",
    "behavior",
    "bibliographicCitation",
    "caste",
    "catalogNumber",
    "class",
    "collectionCode",
    "collectionID",
    "continent",
    "coordinatePrecision",
    "coordinateUncertaintyInMeters",
    "country",
    "countryCode",
    "county",
    "cultivarEpithet",
    "dataGeneralizations",
    "datasetID",
    "datasetName",
    "dateIdentified",
    "day",
    "decimalLatitude",
    "decimalLongitude",
    "degreeOfEstablishment",
    "disposition",
    "dynamicProperties",
    "earliestAgeOrLowestStage",
    "earliestEonOrLowestEonothem",
    "earliestEpochOrLowestSeries",
    "earliestEraOrLowestErathem",
    "earliestPeriodOrLowestSystem",
    "endDayOfYear",
    "establishmentMeans",
    "eventDate",
    "eventID",
    "eventRemarks",
    "eventTime",
    "eventType",
    "family",
    "fieldNotes",
    "fieldNumber",
    "footprintSRS",
    "footprintSpatialFit",
    "footprintWKT",
    "formation",
    "genericName",
    "genus",
    "geodeticDatum",
    "geologicalContextID",
    "georeferenceProtocol",
    "georeferenceRemarks",
    "georeferenceSources",
    "georeferenceVerificationStatus",
    "georeferencedBy",
    "georeferencedDate",
    "group",
    "habitat",
    "higherClassification",
    "higherGeography",
    "higherGeographyID",
    "highestBiostratigraphicZone",
    "identificationID",
    "identificationQualifier",
    "identificationReferences",
    "identificationRemarks",
    "identificationVerificationStatus",
    "identifiedBy",
    "identifiedByID",
    "individualCount",
    "informationWithheld",
    "infragenericEpithet",
    "infraspecificEpithet",
    "institutionCode",
    "institutionID",
    "island",
    "islandGroup",
    "kingdom",
    "language",
    "latestAgeOrHighestStage",
    "latestEonOrHighestEonothem",
    "latestEpochOrHighestSeries",
    "latestEraOrHighestErathem",
    "latestPeriodOrHighestSystem",
    "license",
    "lifeStage",
    "lithostratigraphicTerms",
    "locality",
    "locationAccordingTo",
    "locationID",
    "locationRemarks",
    "lowestBiostratigraphicZone",
    "materialEntityID",
    "materialEntityRemarks",
    "materialSampleID",
    "maximumDepthInMeters",
    "maximumDistanceAboveSurfaceInMeters",
    "maximumElevationInMeters",
    "member",
    "minimumDepthInMeters",
    "minimumDistanceAboveSurfaceInMeters",
    "minimumElevationInMeters",
    "modified",
    "month",
    "municipality",
    "nameAccordingTo",
    "nameAccordingToID",
    "namePublishedIn",
    "namePublishedInID",
    "namePublishedInYear",
    "nomenclaturalCode",
    "nomenclaturalStatus",
    "occurrenceID",
    "occurrenceRemarks",
    "occurrenceStatus",
    "order",
    "organismID",
    "organismName",
    "organismQuantity",
    "organismQuantityType",
    "organismRemarks",
    "organismScope",
    "originalNameUsage",
    "originalNameUsageID",
    "otherCatalogNumbers",
    "ownerInstitutionCode",
    "parentEventID",
    "parentNameUsage",
    "parentNameUsageID",
    "pathway",
    "phylum",
    "pointRadiusSpatialFit",
    "preparations",
    "previousIdentifications",
    "recordNumber",
    "recordedBy",
    "recordedByID",
    "references",
    "reproductiveCondition",
    "rightsHolder",
    "sampleSizeUnit",
    "sampleSizeValue",
    "samplingEffort",
    "samplingProtocol",
    "scientificName",
    "scientificNameAuthorship",
    "scientificNameID",
    "sex",
    "specificEpithet",
    "startDayOfYear",
    "stateProvince",
    "subfamily",
    "subgenus",
    "subtribe",
    "superfamily",
    "taxonConceptID",
    "taxonID",
    "taxonRank",
    "taxonRemarks",
    "taxonomicStatus",
    "tribe",
    "type",
    "typeStatus",
    "verbatimCoordinateSystem",
    "verbatimCoordinates",
    "verbatimDepth",
    "verbatimElevation",
    "verbatimEventDate",
    "verbatimIdentification",
    "verbatimLabel",
    "verbatimLatitude",
    "verbatimLocality",
    "verbatimLongitude",
    "verbatimSRS",
    "verbatimTaxonRank",
    "vernacularName",
    "verticalDatum",
    "vitality",
    "waterBody",
    "year",
]


# Darwin Core field categories
FIELD_CATEGORIES = {
    "Record": [
        "basisOfRecord", "occurrenceID", "occurrenceStatus", "occurrenceRemarks",
        "catalogNumber", "recordNumber", "recordedBy", "recordedByID",
        "individualCount", "organismQuantity", "organismQuantityType",
        "sex", "lifeStage", "reproductiveCondition", "caste", "behavior",
        "vitality", "disposition", "preparations", "associatedMedia",
        "associatedOccurrences", "associatedReferences", "associatedSequences",
        "associatedTaxa", "otherCatalogNumbers", "previousIdentifications",
        "type", "typeStatus", "modified", "language", "license",
        "accessRights", "rightsHolder", "bibliographicCitation",
        "references", "informationWithheld", "dataGeneralizations",
        "datasetID", "datasetName", "collectionCode", "collectionID",
        "institutionCode", "institutionID", "ownerInstitutionCode",
    ],
    "Organism": [
        "organismID", "organismName", "organismRemarks", "organismScope",
        "associatedOrganisms", "degreeOfEstablishment", "establishmentMeans",
        "pathway",
    ],
    "Event": [
        "eventID", "parentEventID", "eventDate", "eventTime", "eventType",
        "eventRemarks", "fieldNotes", "fieldNumber",
        "verbatimEventDate", "year", "month", "day",
        "startDayOfYear", "endDayOfYear",
        "samplingProtocol", "samplingEffort",
        "sampleSizeValue", "sampleSizeUnit",
    ],
    "Location": [
        "continent", "country", "countryCode", "stateProvince", "county",
        "municipality", "island", "islandGroup", "waterBody",
        "locality", "verbatimLocality", "locationID", "locationRemarks",
        "locationAccordingTo", "higherGeography", "higherGeographyID",
    ],
    "Coordinates": [
        "decimalLatitude", "decimalLongitude", "geodeticDatum",
        "coordinateUncertaintyInMeters", "coordinatePrecision",
        "verbatimLatitude", "verbatimLongitude", "verbatimCoordinates",
        "verbatimCoordinateSystem", "verbatimSRS",
        "footprintWKT", "footprintSRS", "footprintSpatialFit",
        "pointRadiusSpatialFit",
        "georeferenceProtocol", "georeferenceSources", "georeferenceRemarks",
        "georeferenceVerificationStatus", "georeferencedBy", "georeferencedDate",
    ],
    "Depth & Elevation": [
        "minimumDepthInMeters", "maximumDepthInMeters", "verbatimDepth",
        "minimumElevationInMeters", "maximumElevationInMeters", "verbatimElevation",
        "minimumDistanceAboveSurfaceInMeters", "maximumDistanceAboveSurfaceInMeters",
        "verticalDatum",
    ],
    "Habitat": [
        "habitat",
    ],
    "Taxonomy": [
        "kingdom", "phylum", "class", "order", "family", "subfamily",
        "superfamily", "tribe", "subtribe",
        "genus", "subgenus", "genericName",
        "specificEpithet", "infraspecificEpithet", "infragenericEpithet",
        "cultivarEpithet",
        "scientificName", "scientificNameAuthorship", "scientificNameID",
        "acceptedNameUsage", "acceptedNameUsageID",
        "originalNameUsage", "originalNameUsageID",
        "parentNameUsage", "parentNameUsageID",
        "nameAccordingTo", "nameAccordingToID",
        "namePublishedIn", "namePublishedInID", "namePublishedInYear",
        "higherClassification", "taxonID", "taxonConceptID",
        "taxonRank", "verbatimTaxonRank", "taxonRemarks",
        "taxonomicStatus", "nomenclaturalCode", "nomenclaturalStatus",
        "vernacularName", "verbatimIdentification",
    ],
    "Identification": [
        "identificationID", "identifiedBy", "identifiedByID",
        "dateIdentified", "identificationRemarks", "identificationQualifier",
        "identificationReferences", "identificationVerificationStatus",
    ],
    "Geological": [
        "geologicalContextID", "group", "formation", "member", "bed",
        "earliestEonOrLowestEonothem", "latestEonOrHighestEonothem",
        "earliestEraOrLowestErathem", "latestEraOrHighestErathem",
        "earliestPeriodOrLowestSystem", "latestPeriodOrHighestSystem",
        "earliestEpochOrLowestSeries", "latestEpochOrHighestSeries",
        "earliestAgeOrLowestStage", "latestAgeOrHighestStage",
        "lowestBiostratigraphicZone", "highestBiostratigraphicZone",
        "lithostratigraphicTerms",
    ],
    "Material": [
        "materialEntityID", "materialEntityRemarks", "materialSampleID",
    ],
    "Other": [
        "dynamicProperties", "verbatimLabel",
    ],
}

# Build reverse lookup: field -> category
_FIELD_TO_CATEGORY = {}
for cat, fields in FIELD_CATEGORIES.items():
    for f in fields:
        _FIELD_TO_CATEGORY[f] = cat


def main():
    files = sorted(glob.glob(str(DATA_DIR / '*.parquet')))
    total_files = len(files)
    print(f"Processing all {total_files} parquet files...\n")

    total_rows = 0
    field_counts = {f: 0 for f in TARGET_FIELDS}

    start_time = time.time()

    for i, file_path in enumerate(files):
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        file_start = time.time()
        try:
            n = pl.scan_parquet(file_path).select(pl.len()).collect().item()
            total_rows += n

            schema = pl.read_parquet_schema(file_path)
            source_dtype = schema.get('source')
            available_fields = []
            if source_dtype is not None and hasattr(source_dtype, 'fields'):
                available_field_names = {f.name for f in source_dtype.fields}
                available_fields = [f for f in TARGET_FIELDS if f in available_field_names]

            # Process fields in small batches to avoid OOM on large files
            BATCH_SIZE = 20
            for batch_start in range(0, len(available_fields), BATCH_SIZE):
                batch_fields = available_fields[batch_start:batch_start + BATCH_SIZE]
                lf = (
                    pl.scan_parquet(file_path)
                    .select(
                        [pl.col('source').struct.field(f).alias(f) for f in batch_fields]
                    )
                    .select(
                        [
                            pl.col(f).cast(pl.Utf8).str.strip_chars()
                            .replace('', None).drop_nulls().len().alias(f)
                            for f in batch_fields
                        ]
                    )
                )
                counts = lf.collect(engine="streaming")

                for field in batch_fields:
                    field_counts[field] += counts[field].item()

                del counts
                gc.collect()

            del n
            gc.collect()

        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)} "
                  f"({file_size_mb:.0f} MB): {e}")

        file_elapsed = time.time() - file_start
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total_files - i - 1) / rate if rate > 0 else 0
            print(f"Processed {i+1}/{total_files} files "
                  f"({total_rows:,} rows) "
                  f"- last file: {file_elapsed:.1f}s ({file_size_mb:.0f} MB) "
                  f"- ETA: {remaining:.0f}s")

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("OBIS DATA REPORT")
    print("=" * 70)
    print(f"\nFiles processed: {total_files:,}")
    print(f"Total records:   {total_rows:,}")
    print(f"Processing time: {elapsed:.1f}s")

    print("\n" + "-" * 70)
    print("FIELD COVERAGE BY CATEGORY")
    print("-" * 70)

    coverage_data = []
    for category, cat_fields in FIELD_CATEGORIES.items():
        # Only show fields that are in TARGET_FIELDS
        fields_in_target = [f for f in cat_fields if f in field_counts]
        if not fields_in_target:
            continue

        cat_total = sum(field_counts.get(f, 0) for f in fields_in_target)
        cat_avg_pct = (cat_total / (total_rows * len(fields_in_target)) * 100) if total_rows > 0 else 0

        print(f"\n  [{category}]  (avg {cat_avg_pct:.1f}%)")
        print(f"  {'─' * 50}")

        for field in fields_in_target:
            count = field_counts[field]
            pct = (count / total_rows * 100) if total_rows > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {field:<40} {count:>12,}  {pct:>6.2f}%  {bar}")
            coverage_data.append({
                'category': category,
                'field': field,
                'records_with_data': count,
                'total_records': total_rows,
                'coverage_pct': round(pct, 4),
            })

    # Save results to CSV
    output_file = OUTPUTS_DIR / 'field_coverage_report.csv'
    pl.DataFrame(coverage_data).write_csv(output_file)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
