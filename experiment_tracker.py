#!/usr/bin/env python3
"""Comet ML experiment tracking utilities.

This module provides helper functions and a command‑line interface for
logging dataset statistics, recording model performance and comparing
multiple experiments.  It can be used standalone or imported by other
scripts.
"""

from __future__ import annotations

import argparse
import logging
from typing import List

import pandas as pd
import numpy as np

try:
    import comet_ml  # type: ignore
except ImportError:
    comet_ml = None

from src.utils import load_config


def log_data_stats(experiment: Any, df: pd.DataFrame) -> None:
    """Log basic dataset statistics to a Comet experiment."""
    if experiment is None:
        return
    experiment.log_parameters({
        'dataset_rows': len(df),
        'dataset_columns': len(df.columns),
    })
    # Log numeric column means
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        experiment.log_metric(f'mean_{col}', float(df[col].mean()))


def log_model_performance(experiment: Any, metrics: dict) -> None:
    """Log performance metrics (accuracy, AUC, etc.) to Comet."""
    if experiment is None:
        return
    experiment.log_metrics(metrics)


def compare_experiments(experiment_ids: List[str]) -> pd.DataFrame:
    """Compare multiple Comet experiments by extracting key metrics.

    Parameters
    ----------
    experiment_ids: List[str]
        List of Comet experiment IDs (not URLs).  You can copy these
        identifiers from the Comet UI.

    Returns
    -------
    pandas.DataFrame
        DataFrame summarising the final validation loss and other metrics
        for each experiment.
    """
    if comet_ml is None:
        raise ImportError("comet_ml is not installed")
    api = comet_ml.api.API()
    records = []
    for exp_id in experiment_ids:
        exp = api.get_experiment_by_id(exp_id)
        metrics = exp.get_metrics_summary()
        record = {
            'experiment_id': exp_id,
            'name': exp.get_name(),
            'val_loss': float(metrics.get('val_loss', {}).get('value', np.nan)),
            'val_mse': float(metrics.get('val_mse', {}).get('value', np.nan)),
        }
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comet ML experiment tracker")
    parser.add_argument('--compare', nargs='+', type=str, help='Experiment IDs to compare')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.compare:
        df = compare_experiments(args.compare)
        print(df)


if __name__ == '__main__':
    main()