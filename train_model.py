#!/usr/bin/env python3
"""Train the multi‑modal drug prediction model.

This script reads the processed dataset, prepares data loaders, instantiates
the ``MultiModalDrugPredictor`` model and trains it using PyTorch
Lightning.  Training and validation metrics are logged to Comet ML if
configured via environment variables.  A model checkpoint is saved to
the ``models/`` directory upon completion.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger
import torch

from src.utils import load_config, init_comet, set_seed
from src.datasets import split_dataframe, make_dataloader
from src.models import MultiModalDrugPredictor
from src.data_processor import DataProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNS drug prediction model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--data-path', type=str, default=None, help='Override processed data path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-epochs', type=int, default=None, help='Override maximum number of epochs')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)
    # Paths
    data_path = args.data_path or config['data']['processed']
    # Load processed data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}.  Run data_pipeline.py first.")
    df = pd.read_csv(data_path)
    # Determine side effect tasks from DataProcessor defaults
    # Use the same patterns as the data processor to ensure consistency.
    processor = DataProcessor(raw_dir=config['data']['raw'], processed_path=data_path)
    side_effect_tasks: List[str] = list(processor.side_effect_patterns.keys())
    # Split dataset
    train_df, val_df, test_df = split_dataframe(
        df,
        val_fraction=config['training'].get('val_size', 0.1),
        test_fraction=config['training'].get('test_size', 0.1),
        random_state=args.seed
    )
    logging.info(f"Train/Val/Test sizes: {len(train_df)}, {len(val_df)}, {len(test_df)}")
    # Create DataLoaders
    train_loader = make_dataloader(train_df, side_effect_tasks,
                                   batch_size=config['training']['batch_size'],
                                   shuffle=True,
                                   num_workers=config['training'].get('num_workers', 0))
    val_loader = make_dataloader(val_df, side_effect_tasks,
                                 batch_size=config['training']['batch_size'],
                                 shuffle=False,
                                 num_workers=config['training'].get('num_workers', 0))
    # Initialise Comet experiment and logger
    comet_exp = init_comet(config)
    if comet_exp is not None:
        comet_logger = CometLogger(experiment=comet_exp)
    else:
        comet_logger = None
    # Instantiate model
    model = MultiModalDrugPredictor(
        smiles_encoder=config['model']['smiles_encoder'],
        molecular_feat_dim=config['model']['molecular_feat_dim'],
        clinical_feat_dim=config['model']['clinical_feat_dim'],
        hidden_dim=config['model']['hidden_dim'],
        side_effect_tasks=side_effect_tasks,
        learning_rate=config['training'].get('learning_rate', 1e-4),
    )
    # Callbacks: checkpoint and early stopping
    checkpoint_callback = ModelCheckpoint(
        dirpath='models',
        filename='pharma_neuro_{epoch:02d}_{val_loss:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['training'].get('patience', 5),
        mode='min'
    )
    # Trainer configuration
    trainer_args = {
        'max_epochs': args.max_epochs or config['training'].get('max_epochs', 50),
        'callbacks': [checkpoint_callback, early_stopping],
        'logger': comet_logger,
    }
    # Use GPU if available
    if torch.cuda.is_available():
        trainer_args['accelerator'] = 'gpu'
        trainer_args['devices'] = 1
    else:
        trainer_args['accelerator'] = 'auto'
    trainer = pl.Trainer(**trainer_args)
    # Train
    trainer.fit(model, train_loader, val_loader)
    # Save best model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logging.info(f"Best model saved at {best_model_path}")
    else:
        logging.warning("Training completed but no checkpoint was saved.")
    # Finish experiment
    if comet_exp is not None:
        comet_exp.end()


if __name__ == '__main__':
    main()