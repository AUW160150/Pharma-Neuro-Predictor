import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List
import comet_ml

class MultiModalDrugPredictor(pl.LightningModule):
    '''Multi-modal neural network for CNS drug prediction'''
    
    def __init__(
        self,
        smiles_encoder: str = 'seyonec/ChemBERTa-zinc-base-v1',
        molecular_feat_dim: int = 8,  # Our molecular descriptors
        hidden_dim: int = 256,
        num_side_effects: int = 4,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # SMILES encoder (pretrained transformer)
        print(f'Loading SMILES encoder: {smiles_encoder}...')
        self.smiles_tokenizer = AutoTokenizer.from_pretrained(smiles_encoder)
        self.smiles_encoder = AutoModel.from_pretrained(smiles_encoder)
        
        # Get SMILES embedding dimension
        smiles_dim = self.smiles_encoder.config.hidden_size
        
        # Molecular descriptor network
        self.molecular_net = nn.Sequential(
            nn.Linear(molecular_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Fusion layer (combine SMILES + molecular features)
        self.fusion = nn.Sequential(
            nn.Linear(smiles_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Multi-task heads
        self.efficacy_head = nn.Linear(hidden_dim // 2, 1)  # Regression
        self.cns_active_head = nn.Linear(hidden_dim // 2, 2)  # Binary classification
        
        # Side effect heads (4 binary classifiers)
        self.side_effect_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 2) for _ in range(num_side_effects)
        ])
        
        self.learning_rate = learning_rate
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, smiles: List[str], molecular_feats: torch.Tensor):
        # Encode SMILES
        tokens = self.smiles_tokenizer(
            smiles,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to same device as model
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Get SMILES embeddings
        smiles_output = self.smiles_encoder(**tokens)
        smiles_embed = smiles_output.last_hidden_state[:, 0, :]  # CLS token
        
        # Process molecular descriptors
        mol_embed = self.molecular_net(molecular_feats)
        
        # Fuse modalities
        fused = torch.cat([smiles_embed, mol_embed], dim=1)
        fused = self.fusion(fused)
        
        # Multi-task predictions
        outputs = {
            'efficacy': self.efficacy_head(fused).squeeze(-1),
            'cns_active': self.cns_active_head(fused),
            'side_effects': [head(fused) for head in self.side_effect_heads]
        }
        
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch['smiles'], batch['molecular_feats'])
        
        # Multi-task loss
        loss = 0
        
        # Efficacy loss (MSE)
        efficacy_loss = self.mse_loss(outputs['efficacy'], batch['efficacy'])
        loss += efficacy_loss
        
        # CNS active loss (CrossEntropy)
        cns_loss = self.ce_loss(outputs['cns_active'], batch['cns_active'])
        loss += cns_loss
        
        # Side effect losses
        se_names = ['sedation', 'seizure_risk', 'cognitive_impair', 'movement_disorder']
        for i, (head_output, se_name) in enumerate(zip(outputs['side_effects'], se_names)):
            se_loss = self.ce_loss(head_output, batch[se_name])
            loss += se_loss
            self.log(f'train_{se_name}_loss', se_loss, prog_bar=False)
        
        # Log losses
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_efficacy_loss', efficacy_loss, prog_bar=True)
        self.log('train_cns_loss', cns_loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch['smiles'], batch['molecular_feats'])
        
        # Calculate losses
        efficacy_loss = self.mse_loss(outputs['efficacy'], batch['efficacy'])
        cns_loss = self.ce_loss(outputs['cns_active'], batch['cns_active'])
        
        val_loss = efficacy_loss + cns_loss
        
        se_names = ['sedation', 'seizure_risk', 'cognitive_impair', 'movement_disorder']
        for i, (head_output, se_name) in enumerate(zip(outputs['side_effects'], se_names)):
            se_loss = self.ce_loss(head_output, batch[se_name])
            val_loss += se_loss
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_efficacy_loss', efficacy_loss, prog_bar=True)
        self.log('val_cns_loss', cns_loss, prog_bar=True)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

if __name__ == '__main__':
    # Test model creation
    print('Testing model initialization...')
    model = MultiModalDrugPredictor()
    
    print(f'\nModel architecture:')
    print(model)
    
    print(f'\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    # Test forward pass
    print('\nTesting forward pass...')
    dummy_smiles = ['CC(C)Cc1ccc(cc1)C(C)C(O)=O']
    dummy_feats = torch.randn(1, 8)
    
    with torch.no_grad():
        outputs = model(dummy_smiles, dummy_feats)
    
    print(f'Efficacy output shape: {outputs["efficacy"].shape}')
    print(f'CNS active output shape: {outputs["cns_active"].shape}')
    print(f'Number of side effect heads: {len(outputs["side_effects"])}')
    
    print('\n✅ Model test passed!')