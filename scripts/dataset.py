#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader for the OBIS marine occurrence data.

Provides DataSample, DataBatch, and MarineDataset classes for loading
parquet-based splits into a PyTorch training pipeline.

Usage:
    python scripts/dataset.py
"""

from random import sample

import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from config import SPLITS_DIR

TARGET_COLUMN = "interpreted"

class DataSample:
    def __init__(self, sample: dict):
        """
        Initialize a data sample.

        Args:
            sample: A dictionary containing the sample data.
        """
        self.samplingProtocol = sample.get("samplingProtocol")
        self.bathymetry = sample.get("bathymetry")
        self.shoredistance = sample.get("shoredistance")
        self.decimalLatitude = sample.get("decimalLatitude")
        self.decimalLongitude = sample.get("decimalLongitude")
        self.geodeticDatum = sample.get("geodeticDatum")
        self.kingdom = sample.get("kingdom")
        self.phylum = sample.get("phylum")
        self.class_ = sample.get("class")
        self.order = sample.get("order")
        self.family = sample.get("family")
        self.genus = sample.get("genus")
        self.species = sample.get("species")
        self.scientificName = sample.get("scientificName")
        self.occurrenceID = sample.get("occurrenceID")

class DataBatch:
    def __init__(self, samples: list[DataSample]):
        """
        Initialize a batch from a list of samples.
        
        Args:
            samples: A list of samples to include in the batch.
        """
        self.samplingProtocol = [sample.samplingProtocol for sample in samples]
        self.bathymetry = torch.tensor([sample.bathymetry for sample in samples])
        self.shoredistance = torch.tensor([sample.shoredistance for sample in samples])
        self.decimalLatitude = torch.tensor([sample.decimalLatitude for sample in samples])
        self.decimalLongitude = torch.tensor([sample.decimalLongitude for sample in samples])
        self.geodeticDatum = [sample.geodeticDatum for sample in samples]
        self.kingdom = [sample.kingdom for sample in samples]
        self.phylum = [sample.phylum for sample in samples]
        self.class_ = [sample.class_ for sample in samples]
        self.order = [sample.order for sample in samples]
        self.family = [sample.family for sample in samples]
        self.genus = [sample.genus for sample in samples]
        self.species = [sample.species for sample in samples]
        self.scientificName = [sample.scientificName for sample in samples]
        self.occurrenceID = [sample.occurrenceID for sample in samples]

class MarineDataset(Dataset):
    def __init__(self, path: Path):
        """
        Initialize the dataset by loading the Parquet file.

        Args:
            path: The path to the Parquet file.
        """
        self.path = path
        self.df = (
            pl.scan_parquet(path)
        )
        self.total_samples = self.df.select(pl.len()).collect().item()

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            The total number of samples.
        """
        return self.total_samples

    def __getitem__(self, idx: int) -> DataSample:
        """
        Get a single sample by index.
        
        Args:
            idx: The index of the sample to retrieve.
        Returns:
            A DataSample instance containing the sample data.
        """

        # Check if index is within bounds
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of range")

        # Retrieve the sample at the specified index. 
        # We use 'slice' to get a single row as a DataFrame, 
        # then select the target column and unnest it to get the fields as a dictionary.
        sample = (
            self.df.slice(idx, 1)
            .select(pl.col(TARGET_COLUMN))
            .unnest(TARGET_COLUMN)
        ).collect().to_dicts()[0]
        return DataSample(sample)

    def collate_fn(self, batch: list[DataSample]) -> DataBatch:
        """
        Collate a list of samples into a batch.
        
        Args:
            batch: A list of DataSample instances to collate into a batch.
        Returns: 
            A DataBatch instance containing the collated samples.
        """
        data_batch = DataBatch(batch)
        return data_batch

if __name__ == "__main__":
    # Example usage
    dataset = MarineDataset(SPLITS_DIR / "train.parquet")
    print(f"Total samples in dataset: {len(dataset)}")

    # Get the first sample and print its contents
    sample = dataset[0]
    print(f"First sample: {sample.__dict__}")

    # Create a DataLoader to iterate over the dataset in batches
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=dataset.collate_fn)
    for batch in dataloader:
        print(f"First batch: {batch.__dict__}")
        break
