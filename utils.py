"""Utility functions for Pharma‑Neuro Predictor.

This module contains helper functions used across the project,
including configuration loading, experiment initialisation and
reproducibility helpers.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import yaml

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    load_dotenv = None  # type: ignore


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file into a Python dictionary."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def init_comet(config: Dict[str, Any]) -> Any:
    """Initialise a Comet ML experiment using environment variables.

    The function reads the COMET_API_KEY, COMET_WORKSPACE and
    COMET_PROJECT_NAME environment variables (optionally loaded from
    a .env file) and returns a ``comet_ml.Experiment`` instance.  If
    the API key is not set the function returns ``None``.
    """
    try:
        import comet_ml  # type: ignore
    except ImportError:
        print("Comet ML is not installed; experiment tracking is disabled.")
        return None
    # Load environment variables from .env if python-dotenv is available
    if load_dotenv is not None:
        load_dotenv()
    api_key = os.getenv('COMET_API_KEY')
    workspace = os.getenv('COMET_WORKSPACE')
    project_name = os.getenv('COMET_PROJECT_NAME', config.get('comet', {}).get('project_name'))
    if not api_key:
        print("COMET_API_KEY not found; skipping Comet experiment initialisation.")
        return None
    experiment = comet_ml.Experiment(api_key=api_key,
                                     workspace=workspace,
                                     project_name=project_name)
    if config.get('comet', {}).get('log_code'):
        experiment.log_code = True
    if config.get('comet', {}).get('log_graph'):
        experiment.log_graph = True
    if config.get('comet', {}).get('log_git_metadata'):
        experiment.log_git_metadata = True
    return experiment


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass