"""Data processing utilities for the Pharma‑Neuro Predictor.

This module defines a ``DataProcessor`` class responsible for loading
and merging raw datasets, computing molecular descriptors and
fingerprints from SMILES strings, deriving labels for efficacy and
neurological side effects, and writing a cleaned dataset to disk.

The processor is intentionally agnostic about where data are stored.
Paths are passed in via a configuration dictionary or command‑line
arguments.  If a Comet ML experiment is supplied, the processor will
log useful metadata about the dataset.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

try:
    # MemVerge is optional.  If not installed, the processor falls back
    # to sequential execution.
    from memverge import MemoryMachine  # type: ignore
except ImportError:
    MemoryMachine = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class DataProcessor:
    """High‑level orchestrator for preparing the input dataset.

    Parameters
    ----------
    raw_dir: str
        Directory containing the raw input CSV files.
    processed_path: str
        Path where the processed CSV will be written.
    comet_experiment: optional
        A ``comet_ml.Experiment`` instance for logging dataset statistics.
    config: Dict[str, Any]
        Additional configuration options (unused for now).
    """

    raw_dir: str
    processed_path: str
    comet_experiment: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)

    # Known side effect patterns for binary classification tasks.
    side_effect_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "sedation": ["somnolence", "sedation", "sleepiness", "drowsiness"],
        "seizure_risk": ["seizure", "convulsion", "epilepsy"],
        "cognitive_impair": ["confusion", "amnesia", "memory loss"],
        "movement_disorder": ["tremor", "dyskinesia", "akathisia", "parkinsonism"],
    })

    def _log(self, msg: str) -> None:
        """Log a message both to the logger and Comet, if configured."""
        logger.info(msg)
        if self.comet_experiment is not None:
            self.comet_experiment.log_other("data_processor_log", msg)

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load a CSV file from the raw data directory, if it exists.

        Parameters
        ----------
        filename: str
            Name of the CSV file to load.  The file is expected to live
            under ``self.raw_dir``.

        Returns
        -------
        pandas.DataFrame
            The loaded data frame, or an empty data frame if the file
            cannot be found.
        """
        path = os.path.join(self.raw_dir, filename)
        if not os.path.exists(path):
            self._log(f"Warning: file '{filename}' not found in {self.raw_dir}; skipping.")
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            self._log(f"Loaded {len(df)} rows from {filename}")
            return df
        except Exception as exc:
            self._log(f"Failed to load {filename}: {exc}")
            return pd.DataFrame()

    def merge_datasets(self, chembl_df: pd.DataFrame, drugbank_df: pd.DataFrame,
                       sider_df: pd.DataFrame, faers_df: pd.DataFrame) -> pd.DataFrame:
        """Merge multiple data sources on a common compound identifier.

        The function assumes that all input data frames contain a column
        called ``compound_id`` that uniquely identifies a molecule across
        datasets.  You may need to pre‑process your raw files to ensure
        this is true.  Any missing data frames are ignored.

        Parameters
        ----------
        chembl_df, drugbank_df, sider_df, faers_df
            Data frames loaded from the respective CSV files.  They can be
            empty if the corresponding file did not exist.

        Returns
        -------
        pandas.DataFrame
            A merged data frame containing the union of available columns.
        """
        dfs: List[pd.DataFrame] = []
        for name, df in [("ChEMBL", chembl_df), ("DrugBank", drugbank_df),
                         ("SIDER", sider_df), ("FAERS", faers_df)]:
            if not df.empty:
                dfs.append(df)
                self._log(f"Merging {len(df)} rows from {name}")
        if not dfs:
            self._log("No input data frames were provided.  Returning empty dataset.")
            return pd.DataFrame()

        # Perform successive merges on the 'compound_id' column.
        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on="compound_id", how="outer")
        self._log(f"Merged dataset contains {len(merged)} rows and {len(merged.columns)} columns")
        return merged

    @staticmethod
    def _compute_descriptors_for_molecule(mol: Chem.Mol) -> Dict[str, Any]:
        """Compute a dictionary of molecular descriptors for a single RDKit molecule.

        If the molecule is invalid (``mol`` is ``None``), an empty
        dictionary is returned.
        """
        if mol is None:
            return {}
        # Standard physicochemical properties.
        descriptors = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol),
        }
        # Extended connectivity fingerprint (ECFP) bits as a tuple of ints (0/1).
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        descriptors.update({f'ECFP_{i}': int(fp.GetBit(i)) for i in range(fp.GetNumBits())})
        return descriptors

    def compute_molecular_features(self, smiles_list: List[str]) -> pd.DataFrame:
        """Compute molecular descriptors for a list of SMILES strings.

        This function will attempt to parallelise computation using MemVerge
        MemoryMachine if available; otherwise it falls back to a sequential
        loop.  The return value is a data frame where each row corresponds
        to a molecule in ``smiles_list``.

        Parameters
        ----------
        smiles_list: List[str]
            List of SMILES strings for which to compute descriptors.

        Returns
        -------
        pandas.DataFrame
            Data frame of descriptors.  Columns include basic
            physicochemical properties and ECFP bits.
        """
        self._log(f"Computing descriptors for {len(smiles_list)} SMILES strings")
        # Worker function to compute descriptors for a batch of SMILES.
        def worker(batch: List[str]) -> List[Dict[str, Any]]:
            results = []
            for smi in batch:
                mol = Chem.MolFromSmiles(str(smi))
                results.append(self._compute_descriptors_for_molecule(mol))
            return results

        # Determine batch size heuristically: 1000 works well on most
        # machines.  You can adjust this in the config.
        batch_size = 1000
        batches = [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]
        all_results: List[Dict[str, Any]] = []
        if MemoryMachine is not None:
            # Use MemVerge to parallelise the worker across multiple processes.
            mm = MemoryMachine(num_workers=os.cpu_count() or 1)
            self._log(f"Using MemVerge MemoryMachine with {mm.num_workers} workers")

            @mm.parallelize
            def process_batch(batch: List[str]) -> List[Dict[str, Any]]:
                return worker(batch)
            results = process_batch.map(batches)
            # results is a list of lists; flatten it.
            for res in results:
                all_results.extend(res)
        else:
            # Fallback: sequential processing.
            self._log("MemVerge not available; computing descriptors sequentially")
            for batch in batches:
                all_results.extend(worker(batch))
        return pd.DataFrame(all_results)

    def create_side_effect_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary labels for side effects based on free‑text descriptions.

        The input data frame must contain a ``side_effects`` column with a
        string of comma‑separated side effect names.  For each pattern
        defined in ``side_effect_patterns`` a boolean label is created.
        """
        labels: Dict[str, List[bool]] = {name: [] for name in self.side_effect_patterns}
        for _, row in df.iterrows():
            text = str(row.get('side_effects', '')).lower()
            for name, patterns in self.side_effect_patterns.items():
                match = any(pattern in text for pattern in patterns)
                labels[name].append(match)
        label_df = pd.DataFrame(labels)
        self._log("Computed side effect labels")
        return label_df

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create efficacy and CNS activity targets from activity data.

        The input data frame must contain a column ``pchembl_value`` which
        encodes a potency measurement (higher is better).  A binary label
        ``cns_active`` is defined by thresholding this value at 6.0.  A
        regression target ``efficacy_score`` is equal to the pChEMBL value.
        """
        if 'pchembl_value' not in df.columns:
            # If no pChEMBL values are available, create dummy values to allow
            # downstream processing.  In a production context you should
            # replace this with real activity measurements.
            self._log("Warning: pchembl_value column not found; creating dummy targets")
            df['pchembl_value'] = 0.0
        targets = pd.DataFrame()
        targets['efficacy_score'] = df['pchembl_value'].astype(float)
        targets['cns_active'] = (df['pchembl_value'] > 6.0).astype(int)
        self._log("Created efficacy and CNS activity targets")
        return targets

    def process(self) -> pd.DataFrame:
        """Run the full data processing pipeline.

        This method loads all available raw datasets, merges them, computes
        molecular descriptors, derives targets and side effect labels, and
        writes the final processed dataset to ``self.processed_path``.

        Returns
        -------
        pandas.DataFrame
            The processed data frame ready for model training.
        """
        self._log("Starting data processing pipeline")
        chembl_df = self.load_csv('chembl_data.csv')
        drugbank_df = self.load_csv('drugbank_data.csv')
        sider_df = self.load_csv('sider_data.csv')
        faers_df = self.load_csv('faers_data.csv')

        merged = self.merge_datasets(chembl_df, drugbank_df, sider_df, faers_df)
        if merged.empty:
            self._log("No data to process; exiting")
            return pd.DataFrame()

        # Compute features
        smiles_list = merged.get('smiles') if 'smiles' in merged.columns else None
        if smiles_list is None:
            self._log("Error: no 'smiles' column found in merged data.  Cannot compute descriptors.")
            return pd.DataFrame()
        features_df = self.compute_molecular_features(list(smiles_list))
        self._log(f"Computed descriptor matrix of shape {features_df.shape}")

        # Create targets and side effect labels
        targets_df = self.create_targets(merged)
        side_effects_df = self.create_side_effect_labels(merged)

        # Concatenate original metadata, features and targets
        processed = pd.concat([merged.reset_index(drop=True), features_df, targets_df, side_effects_df], axis=1)
        self._log(f"Final processed dataset shape: {processed.shape}")

        # Write to disk
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        processed.to_csv(self.processed_path, index=False)
        self._log(f"Wrote processed data to {self.processed_path}")

        # Log dataset statistics to Comet
        if self.comet_experiment is not None:
            self.comet_experiment.log_parameters({
                'n_rows': len(processed),
                'n_features': processed.shape[1],
            })
        return processed