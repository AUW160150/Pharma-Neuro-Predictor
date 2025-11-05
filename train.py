import os
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from dotenv import load_dotenv

# Import from src directory
from src.data.dataset import CNSDrugDataset, collate_fn
from src.models.multimodal_model import MultiModalDrugPredictor

# Load environment variables
load_dotenv()

def train():
    '''Main training function'''
    
    print('='*50)
    print('PharmaNeuro Predictor - Training')
    print('='*50)
    
    # Hyperparameters
    BATCH_SIZE = 16
    MAX_EPOCHS = 20
    LEARNING_RATE = 1e-4
    
    # Initialize Comet Logger
    comet_logger = CometLogger(
        api_key=os.getenv('COMET_API_KEY'),
        project_name=os.getenv('COMET_PROJECT_NAME'),
        workspace=os.getenv('COMET_WORKSPACE'),
        experiment_name='multimodal_training_v1'
    )
    
    # Log hyperparameters
    comet_logger.experiment.log_parameters({
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'model': 'MultiModalDrugPredictor',
        'smiles_encoder': 'ChemBERTa'
    })
    
    print('\n📊 Comet ML experiment initialized!')
    print(f'View at: {comet_logger.experiment.url}')
    
    # Load dataset
    print('\n📂 Loading dataset...')
    dataset = CNSDrugDataset('data/processed/cns_drugs_with_features.csv')
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize model
    print('\n🧠 Initializing model...')
    model = MultiModalDrugPredictor(
        learning_rate=LEARNING_RATE
    )
    
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints',
        filename='pharma-neuro-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=comet_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',
        devices=1,
        log_every_n_steps=10
    )
    
    # Train!
    print('\n🚀 Starting training...')
    print('='*50)
    
    trainer.fit(model, train_loader, val_loader)
    
    print('\n✅ Training complete!')
    print(f'Best model saved to: {checkpoint_callback.best_model_path}')
    print(f'📊 View results: {comet_logger.experiment.url}')
    
    # Log final model to Comet
    comet_logger.experiment.log_model(
        'pharma_neuro_final',
        checkpoint_callback.best_model_path
    )
    
    return model, trainer

if __name__ == '__main__':
    train()