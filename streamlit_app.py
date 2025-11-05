#!/usr/bin/env python3
"""Streamlit application for Pharma‑Neuro Predictor.

This app loads a trained checkpoint and provides an interactive user
interface for predicting efficacy, CNS activity and multiple
neurological side effect risks from a single SMILES string.  It
computes molecular descriptors on the fly using RDKit.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch
import streamlit as st
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

from src.utils import load_config
from src.models import MultiModalDrugPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Streamlit app for PharmaNeuro Predictor")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint (.ckpt)')
    return parser.parse_args()


@st.cache_resource
def load_model(checkpoint_path: str, config: Dict[str, any], side_effect_tasks: List[str]) -> MultiModalDrugPredictor:
    """Load the trained model from a checkpoint."""
    model = MultiModalDrugPredictor.load_from_checkpoint(
        checkpoint_path,
        smiles_encoder=config['model']['smiles_encoder'],
        molecular_feat_dim=config['model']['molecular_feat_dim'],
        clinical_feat_dim=config['model']['clinical_feat_dim'],
        hidden_dim=config['model']['hidden_dim'],
        side_effect_tasks=side_effect_tasks,
    )
    model.eval()
    return model


def compute_features(smiles: str) -> torch.Tensor:
    """Compute molecular descriptor tensor for a single SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    # Physicochemical descriptors
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
    ]
    # ECFP fingerprint bits
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    descriptors.extend(int(fp.GetBit(i)) for i in range(fp.GetNumBits()))
    return torch.tensor(descriptors, dtype=torch.float32).unsqueeze(0)  # shape (1, feat_dim)


def main_app(config: Dict[str, any], checkpoint: str) -> None:
    side_effect_tasks = list(config['model'].get('side_effect_tasks', ['sedation', 'seizure_risk', 'cognitive_impair', 'movement_disorder']))
    # Load model
    model = load_model(checkpoint, config, side_effect_tasks)
    st.title(config['app'].get('title', 'PharmaNeuro Predictor'))
    st.markdown(config['app'].get('description', ''))
    # User input
    smiles = st.text_input("Enter a SMILES string", "CC(C)Cc1ccc(cc1)C(C)C(O)=O")
    if st.button("Predict"):
        try:
            features = compute_features(smiles)
        except Exception as exc:
            st.error(f"Error computing descriptors: {exc}")
            return
        # Dummy clinical tensor (no clinical features used in this example)
        clinical = torch.zeros((1, config['model']['clinical_feat_dim']), dtype=torch.float32)
        with torch.no_grad():
            outputs = model([smiles], features, clinical)
        # Display results
        st.subheader("Predictions")
        efficacy = outputs['efficacy'].item()
        cns_probs = torch.softmax(outputs['cns_active'], dim=1).cpu().numpy()[0]
        st.metric("Efficacy Score", f"{efficacy:.2f}")
        st.metric("CNS Active Probability", f"{cns_probs[1]:.1%}")
        # Side effect probabilities
        st.subheader("Neurological Side Effect Risks")
        side_effect_probs = {}
        for task in side_effect_tasks:
            logits = outputs[task]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            side_effect_probs[task] = probs[1]
        # Display as dataframe for nicer formatting
        df = pd.DataFrame({
            'Side Effect': list(side_effect_probs.keys()),
            'Probability': [f"{p:.1%}" for p in side_effect_probs.values()]
        })
        st.table(df)
        # Display molecule properties
        st.subheader("Molecular Properties")
        props = {
            "Molecular Weight": Descriptors.MolWt(Chem.MolFromSmiles(smiles)),
            "LogP": Descriptors.MolLogP(Chem.MolFromSmiles(smiles)),
            "TPSA": Descriptors.TPSA(Chem.MolFromSmiles(smiles)),
            "H‑Bond Donors": Descriptors.NumHDonors(Chem.MolFromSmiles(smiles)),
            "H‑Bond Acceptors": Descriptors.NumHAcceptors(Chem.MolFromSmiles(smiles)),
        }
        for k, v in props.items():
            st.write(f"**{k}:** {v:.2f}")
        # Blood brain barrier rule of thumb
        bbb_pass = (
            props["Molecular Weight"] < 450 and props["LogP"] < 5 and
            props["H‑Bond Donors"] < 5 and props["H‑Bond Acceptors"] < 10
        )
        if bbb_pass:
            st.success("Likely to penetrate blood–brain barrier")
        else:
            st.warning("May not cross the blood–brain barrier")


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    main_app(config, args.checkpoint)