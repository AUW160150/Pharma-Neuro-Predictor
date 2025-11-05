from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import os
from dotenv import load_dotenv
import uvicorn

from src.models.multimodal_model import MultiModalDrugPredictor
from src.data.data_loader import MolecularFeatureExtractor
from rdkit import Chem
from rdkit.Chem import Descriptors

load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="PharmaNeuro Predictor API",
    description="MemVerge-Powered CNS Drug Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model (loaded once at startup)
MODEL = None
FEATURE_EXTRACTOR = None

# Request/Response models
class DrugInput(BaseModel):
    smiles: str = Field(..., description="SMILES string of the drug molecule")
    
    class Config:
        schema_extra = {
            "example": {
                "smiles": "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
            }
        }

class BatchDrugInput(BaseModel):
    smiles_list: List[str] = Field(..., description="List of SMILES strings")
    
    class Config:
        schema_extra = {
            "example": {
                "smiles_list": [
                    "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
                    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
                ]
            }
        }

class PredictionOutput(BaseModel):
    smiles: str
    efficacy_score: float
    cns_active_prob: float
    side_effects: dict
    molecular_properties: dict
    bbb_penetrant: bool

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total_processed: int
    successful: int
    failed: int

# Startup event
@app.on_event("startup")
async def load_model():
    """Load model at startup"""
    global MODEL, FEATURE_EXTRACTOR
    
    print("🚀 Loading PharmaNeuro Predictor model...")
    
    # Find latest checkpoint
    checkpoint_dir = 'models/checkpoints'
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
            MODEL = MultiModalDrugPredictor.load_from_checkpoint(latest_checkpoint)
            MODEL.eval()
            print(f"✅ Model loaded: {latest_checkpoint}")
        else:
            print("⚠️ No checkpoint found, using untrained model")
            MODEL = MultiModalDrugPredictor()
            MODEL.eval()
    else:
        print("⚠️ No checkpoint directory, using untrained model")
        MODEL = MultiModalDrugPredictor()
        MODEL.eval()
    
    FEATURE_EXTRACTOR = MolecularFeatureExtractor()
    print("✅ Feature extractor initialized")

# Helper function
def predict_single(smiles: str) -> Optional[dict]:
    """Make prediction for a single compound"""
    try:
        # Extract features
        mol_feats = FEATURE_EXTRACTOR.smiles_to_features(smiles)
        
        if mol_feats is None:
            return None
        
        # Prepare tensors
        feat_tensor = torch.tensor([
            mol_feats['MW'],
            mol_feats['LogP'],
            mol_feats['TPSA'],
            mol_feats['HBD'],
            mol_feats['HBA'],
            mol_feats['RotBonds'],
            mol_feats['AromaticRings'],
            mol_feats['FractionCsp3']
        ], dtype=torch.float32).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = MODEL([smiles], feat_tensor)
        
        # Extract predictions
        efficacy = outputs['efficacy'].item()
        cns_prob = torch.softmax(outputs['cns_active'], dim=1)[0, 1].item()
        
        side_effects = {
            'sedation': torch.softmax(outputs['side_effects'][0], dim=1)[0, 1].item(),
            'seizure_risk': torch.softmax(outputs['side_effects'][1], dim=1)[0, 1].item(),
            'cognitive_impairment': torch.softmax(outputs['side_effects'][2], dim=1)[0, 1].item(),
            'movement_disorder': torch.softmax(outputs['side_effects'][3], dim=1)[0, 1].item()
        }
        
        # Molecular properties
        mol = Chem.MolFromSmiles(smiles)
        properties = {
            'molecular_weight': float(mol_feats['MW']),
            'logp': float(mol_feats['LogP']),
            'tpsa': float(mol_feats['TPSA']),
            'h_bond_donors': int(mol_feats['HBD']),
            'h_bond_acceptors': int(mol_feats['HBA']),
            'rotatable_bonds': int(mol_feats['RotBonds']),
            'aromatic_rings': int(mol_feats['AromaticRings'])
        }
        
        # BBB penetration (Lipinski's rule)
        bbb_penetrant = (
            mol_feats['MW'] < 450 and
            mol_feats['LogP'] < 5 and
            mol_feats['HBD'] < 5 and
            mol_feats['HBA'] < 10
        )
        
        return {
            'smiles': smiles,
            'efficacy_score': round(efficacy, 3),
            'cns_active_prob': round(cns_prob, 3),
            'side_effects': {k: round(v, 3) for k, v in side_effects.items()},
            'molecular_properties': properties,
            'bbb_penetrant': bool(bbb_penetrant)
        }
        
    except Exception as e:
        print(f"Error predicting {smiles}: {e}")
        return None

# API Endpoints

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "PharmaNeuro Predictor API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(drug: DrugInput):
    """
    Predict CNS drug efficacy and side effects for a single compound
    
    **Input:**
    - SMILES string of the molecule
    
    **Output:**
    - Efficacy score (0-10)
    - CNS active probability (0-1)
    - Side effect risks (0-1 for each)
    - Molecular properties
    - BBB penetration assessment
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = predict_single(drug.smiles)
    
    if result is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    return result

@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(batch: BatchDrugInput):
    """
    Batch prediction endpoint
    
    Process multiple compounds in a single request (max 100)
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(batch.smiles_list) > 100:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 100 compounds per batch. Use /predict/large-batch for more."
        )
    
    predictions = []
    failed = 0
    
    for smiles in batch.smiles_list:
        result = predict_single(smiles)
        if result:
            predictions.append(result)
        else:
            failed += 1
    
    return {
        "predictions": predictions,
        "total_processed": len(batch.smiles_list),
        "successful": len(predictions),
        "failed": failed
    }

@app.get("/model/info")
async def model_info():
    """Get model information and statistics"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Count parameters
    total_params = sum(p.numel() for p in MODEL.parameters())
    trainable_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    
    return {
        "model_type": "MultiModalDrugPredictor",
        "encoder": "ChemBERTa-zinc-base-v1",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "tasks": [
            "efficacy_prediction",
            "cns_activity_classification",
            "sedation_risk",
            "seizure_risk",
            "cognitive_impairment_risk",
            "movement_disorder_risk"
        ]
    }

@app.get("/examples")
async def examples():
    """Get example drugs with known CNS activity"""
    return {
        "examples": [
            {
                "name": "Ibuprofen",
                "smiles": "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
                "description": "NSAID pain reliever"
            },
            {
                "name": "Caffeine",
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "description": "CNS stimulant"
            },
            {
                "name": "Diazepam",
                "smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc12",
                "description": "Benzodiazepine sedative"
            },
            {
                "name": "Fluoxetine",
                "smiles": "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
                "description": "SSRI antidepressant"
            }
        ]
    }

# Run server
if __name__ == "__main__":
    print("="*70)
    print("🚀 Starting PharmaNeuro Predictor API")
    print("="*70)
    print("📖 Documentation: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/health")
    print("="*70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)