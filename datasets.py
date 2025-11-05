"""Dataset utilities for Pharma‑Neuro Predictor.

This module provides a ``DrugDataset`` for PyTorch which wraps a
processed pandas DataFrame and yields batches of SMILES strings,
molecular descriptor tensors, optional clinical features, and target
labels.  A helper ``split_dataframe`` function is provided to split
data into training, validation and test sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def split_dataframe(df: pd.DataFrame, val_fraction: float, test_fraction: float,
                    random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/validation/test subsets.

    Parameters
    ----------
    df: pd.DataFrame
        Input data frame to split.
    val_fraction: float
        Fraction of the data to use for the validation set.
    test_fraction: float
        Fraction of the data to use for the test set.
    random_state: int or None
        Random seed for reproducibility.

    Returns
    -------
    (train_df, val_df, test_df): tuple of pd.DataFrame
    """
    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n_total = len(df_shuffled)
    n_test = int(n_total * test_fraction)
    n_val = int(n_total * val_fraction)
    test_df = df_shuffled.iloc[:n_test]
    val_df = df_shuffled.iloc[n_test:n_test + n_val]
    train_df = df_shuffled.iloc[n_test + n_val:]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


@dataclass
class DrugDataset(Dataset):
    """PyTorch dataset wrapping a processed DataFrame.

    The data frame must include the following columns:

    * ``smiles`` – SMILES strings of the compounds.
    * Descriptor columns named ``MW``, ``LogP`` etc.  Extended
      connectivity fingerprints are stored in columns named ``ECFP_i``.
    * Target columns: ``efficacy_score``, ``cns_active`` and one
      column per side effect task (e.g. ``sedation``).
    """
    df: pd.DataFrame
    side_effect_tasks: List[str]

    def __post_init__(self) -> None:
        # Identify descriptor columns: everything that starts with ECFP_ or is a standard descriptor
        descriptor_cols = [col for col in self.df.columns if col.startswith('ECFP_') or col in
                           ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'AromaticRings']]
        self.descriptor_cols = descriptor_cols

        # Identify clinical feature columns; optional.  They should be named
        # ``clinical_`` or can be absent.  If absent, a zero tensor will be used.
        self.clinical_cols = [col for col in self.df.columns if col.startswith('clinical_')]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sample: Dict[str, torch.Tensor] = {}
        sample['smiles'] = [row['smiles']]  # keep as list to preserve batch dimension in model
        # Molecular descriptors
        desc = row[self.descriptor_cols].astype(float).values
        sample['molecular'] = torch.tensor(desc, dtype=torch.float32)
        # Clinical features (optional)
        if self.clinical_cols:
            clin = row[self.clinical_cols].astype(float).values
            sample['clinical'] = torch.tensor(clin, dtype=torch.float32)
        else:
            # Create a zero‑dimensional tensor if no clinical features exist
            sample['clinical'] = torch.zeros(1, dtype=torch.float32)
        # Targets
        sample['efficacy_target'] = torch.tensor(row['efficacy_score'], dtype=torch.float32)
        sample['cns_active_target'] = torch.tensor(int(row['cns_active']), dtype=torch.long)
        for task in self.side_effect_tasks:
            sample[f'{task}_target'] = torch.tensor(int(row[task]), dtype=torch.long)
        return sample

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function to combine variable‑length smiles lists."""
        smiles = [item['smiles'][0] for item in batch]
        molecular = torch.stack([item['molecular'] for item in batch])
        # Handle clinical features: some samples may have zero‑dimensional tensors
        clinical_list = [item['clinical'] for item in batch]
        if clinical_list and clinical_list[0].numel() > 1:
            clinical = torch.stack(clinical_list)
        else:
            # No clinical features present; create zero tensor of shape (batch, 1)
            clinical = torch.zeros(len(batch), 1, dtype=torch.float32)
        efficacy_target = torch.stack([item['efficacy_target'] for item in batch])
        cns_active_target = torch.stack([item['cns_active_target'] for item in batch])
        batch_dict: Dict[str, torch.Tensor | List[str]] = {
            'smiles': smiles,
            'molecular': molecular,
            'clinical': clinical,
            'efficacy_target': efficacy_target,
            'cns_active_target': cns_active_target,
        }
        for task in self.side_effect_tasks:
            batch_dict[f'{task}_target'] = torch.stack([item[f'{task}_target'] for item in batch])
        return batch_dict


def make_dataloader(df: pd.DataFrame, side_effect_tasks: List[str], batch_size: int,
                    shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """Create a DataLoader from a processed DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        Processed dataset.
    side_effect_tasks: List[str]
        Names of side effect columns to predict.
    batch_size: int
        Mini‑batch size.
    shuffle: bool
        Whether to shuffle the dataset.
    num_workers: int
        Number of subprocesses to use for data loading.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader yielding batches suitable for ``MultiModalDrugPredictor``.
    """
    dataset = DrugDataset(df, side_effect_tasks=side_effect_tasks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=dataset.collate_fn)