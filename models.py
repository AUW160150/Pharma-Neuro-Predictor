"""Model definitions for Pharma‑Neuro Predictor.

This module defines a PyTorch Lightning module ``MultiModalDrugPredictor``
that combines a pre‑trained SMILES transformer with feed‑forward networks
for molecular descriptors and optional clinical features.  The output
consists of a regression head for efficacy prediction, a binary
classification head for CNS activity, and multiple binary heads for
neurological side effect risks.
"""

from __future__ import annotations

from typing import List, Dict, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer


class MultiModalDrugPredictor(pl.LightningModule):
    """Multi‑task neural network for CNS drug prediction.

    Parameters
    ----------
    smiles_encoder: str
        Name or path of a pre‑trained transformer model capable of encoding
        SMILES strings (e.g. ``seyonec/ChemBERTa-zinc-base-v1``).
    molecular_feat_dim: int
        Dimensionality of the molecular descriptor input.  This should
        match the number of descriptor columns produced by the data
        processor (including ECFP bits).
    clinical_feat_dim: int
        Dimensionality of the optional clinical features.  If you do not
        provide clinical features, set this to zero; the corresponding
        branch will be skipped.
    hidden_dim: int
        Number of units in the shared fusion layer.
    side_effect_tasks: List[str]
        Names of the side effect classification tasks.  The length of
        this list determines the number of side effect heads.
    learning_rate: float
        Initial learning rate for AdamW.
    """

    def __init__(self,
                 smiles_encoder: str,
                 molecular_feat_dim: int,
                 clinical_feat_dim: int,
                 hidden_dim: int,
                 side_effect_tasks: List[str],
                 learning_rate: float = 1e-4,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.side_effect_tasks = side_effect_tasks

        # SMILES encoder: a transformer model from Hugging Face.  It
        # returns hidden states for each token; we take the [CLS] token
        # representation (first token) as the sequence embedding.
        self.smiles_encoder = AutoModel.from_pretrained(smiles_encoder)
        self.smiles_tokenizer = AutoTokenizer.from_pretrained(smiles_encoder)
        smiles_hidden_dim = self.smiles_encoder.config.hidden_size

        # Feed‑forward network for molecular descriptors
        self.molecular_net = nn.Sequential(
            nn.Linear(molecular_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        ) if molecular_feat_dim > 0 else None

        # Feed‑forward network for optional clinical features
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        ) if clinical_feat_dim > 0 else None

        # Fusion layer combines all embeddings
        fusion_input_dim = smiles_hidden_dim
        if self.molecular_net is not None:
            fusion_input_dim += 256
        if self.clinical_net is not None:
            fusion_input_dim += 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output heads
        self.efficacy_head = nn.Linear(hidden_dim, 1)          # Regression
        self.cns_active_head = nn.Linear(hidden_dim, 2)        # Binary classification
        self.side_effect_heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, 2) for name in side_effect_tasks
        })

    def encode_smiles(self, smiles: List[str]) -> torch.Tensor:
        """Encode a batch of SMILES strings into fixed‑size embeddings.

        The SMILES encoder returns a sequence of hidden states; we use the
        first token's embedding ([CLS]) as a summary representation.
        """
        # Tokenise SMILES strings.  ``return_tensors='pt'`` produces a
        # dictionary with ``input_ids`` and ``attention_mask`` tensors.
        tokens = self.smiles_tokenizer(smiles,
                                       return_tensors='pt',
                                       padding=True,
                                       truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        # Forward pass through the transformer
        outputs = self.smiles_encoder(**tokens)
        # Use the [CLS] token (position 0) as the embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def forward(self, smiles: List[str], molecular_feats: torch.Tensor,
                clinical_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for multi‑modal inputs.

        Parameters
        ----------
        smiles: List[str]
            Batch of SMILES strings.
        molecular_feats: torch.Tensor
            Tensor of shape (batch_size, molecular_feat_dim).
        clinical_feats: torch.Tensor
            Tensor of shape (batch_size, clinical_feat_dim) or a zero‑tensor
            if no clinical data are provided.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with keys ``efficacy`` (regression output),
            ``cns_active`` (logits) and one key per side effect.
        """
        # Encode SMILES
        smiles_embed = self.encode_smiles(smiles)
        # Process molecular descriptors
        if self.molecular_net is not None:
            mol_embed = self.molecular_net(molecular_feats)
        else:
            mol_embed = torch.zeros(smiles_embed.size(0), 0, device=self.device)
        # Process clinical features
        if self.clinical_net is not None:
            clin_embed = self.clinical_net(clinical_feats)
        else:
            clin_embed = torch.zeros(smiles_embed.size(0), 0, device=self.device)
        # Concatenate all embeddings
        fused = torch.cat([emb for emb in [smiles_embed, mol_embed, clin_embed] if emb.numel() > 0], dim=1)
        fused = self.fusion(fused)
        outputs = {
            'efficacy': self.efficacy_head(fused).squeeze(dim=-1),
            'cns_active': self.cns_active_head(fused),
        }
        for name, head in self.side_effect_heads.items():
            outputs[name] = head(fused)
        return outputs

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step called by PyTorch Lightning."""
        outputs = self(batch['smiles'], batch['molecular'], batch['clinical'])
        # Compute losses: MSE for efficacy, cross‑entropy for classification
        loss = 0.0
        mse_loss = nn.MSELoss()(outputs['efficacy'], batch['efficacy_target'])
        cns_loss = nn.CrossEntropyLoss()(outputs['cns_active'], batch['cns_active_target'])
        loss = mse_loss + cns_loss
        for name in self.side_effect_tasks:
            ce_loss = nn.CrossEntropyLoss()(outputs[name], batch[f'{name}_target'])
            loss = loss + ce_loss
        # Log individual losses
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse_loss, on_step=False, on_epoch=True)
        self.log('train_cns_loss', cns_loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        outputs = self(batch['smiles'], batch['molecular'], batch['clinical'])
        loss = 0.0
        mse_loss = nn.MSELoss()(outputs['efficacy'], batch['efficacy_target'])
        cns_loss = nn.CrossEntropyLoss()(outputs['cns_active'], batch['cns_active_target'])
        loss = mse_loss + cns_loss
        for name in self.side_effect_tasks:
            ce_loss = nn.CrossEntropyLoss()(outputs[name], batch[f'{name}_target'])
            loss = loss + ce_loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mse', mse_loss, on_epoch=True)
        self.log('val_cns_loss', cns_loss, on_epoch=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        outputs = self(batch['smiles'], batch['molecular'], batch['clinical'])
        mse_loss = nn.MSELoss()(outputs['efficacy'], batch['efficacy_target'])
        cns_loss = nn.CrossEntropyLoss()(outputs['cns_active'], batch['cns_active_target'])
        self.log('test_mse', mse_loss)
        self.log('test_cns_loss', cns_loss)
        for name in self.side_effect_tasks:
            ce_loss = nn.CrossEntropyLoss()(outputs[name], batch[f'{name}_target'])
            self.log(f'test_{name}_loss', ce_loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)