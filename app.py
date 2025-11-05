import streamlit as st
import torch
import pandas as pd
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import os
from dotenv import load_dotenv

from src.models.multimodal_model import MultiModalDrugPredictor

# Load environment
load_dotenv()

# Page config
st.set_page_config(
    page_title="PharmaNeuro Predictor",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Try to load the best checkpoint
        checkpoint_path = 'models/checkpoints'
        if os.path.exists(checkpoint_path):
            checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]
            if checkpoints:
                latest_checkpoint = os.path.join(checkpoint_path, sorted(checkpoints)[-1])
                model = MultiModalDrugPredictor.load_from_checkpoint(latest_checkpoint)
                model.eval()
                return model, latest_checkpoint
        
        # If no checkpoint, return untrained model
        model = MultiModalDrugPredictor()
        model.eval()
        return model, "untrained"
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def compute_molecular_features(smiles):
    """Compute molecular descriptors from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = torch.tensor([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol)
    ], dtype=torch.float32).unsqueeze(0)
    
    return features, mol

def predict(model, smiles, molecular_feats):
    """Make prediction"""
    with torch.no_grad():
        outputs = model([smiles], molecular_feats)
    
    # Process outputs
    efficacy = outputs['efficacy'].item()
    cns_prob = torch.softmax(outputs['cns_active'], dim=1)[0, 1].item()
    
    side_effects = {
        'Sedation': torch.softmax(outputs['side_effects'][0], dim=1)[0, 1].item(),
        'Seizure Risk': torch.softmax(outputs['side_effects'][1], dim=1)[0, 1].item(),
        'Cognitive Impairment': torch.softmax(outputs['side_effects'][2], dim=1)[0, 1].item(),
        'Movement Disorder': torch.softmax(outputs['side_effects'][3], dim=1)[0, 1].item()
    }
    
    return efficacy, cns_prob, side_effects

# Header
st.markdown('<h1 class="main-header">🧠 PharmaNeuro Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">AI-Powered CNS Drug Efficacy & Side Effect Prediction</p>', unsafe_allow_html=True)

st.markdown("---")

# Load model
model, model_path = load_model()

if model is None:
    st.error("Failed to load model. Please train the model first.")
    st.stop()

if model_path == "untrained":
    st.warning("⚠️ Using untrained model. Run `python train.py` to train the model.")
else:
    st.success(f"✅ Model loaded from: {model_path}")

# Sidebar
st.sidebar.header("📝 Input Drug Information")

# Example drugs
example_drugs = {
    "Ibuprofen (Pain Reliever)": "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    "Aspirin (Pain Reliever)": "CC(=O)Oc1ccccc1C(O)=O",
    "Caffeine (Stimulant)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Diazepam (Sedative)": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc12",
    "Fluoxetine (Antidepressant)": "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "Donepezil (Alzheimer's)": "COc1cc2c(cc1OC)CC(C2=O)CC1CCN(CC1)Cc1ccccc1",
}

selected_example = st.sidebar.selectbox(
    "Select Example Drug:",
    ["Custom"] + list(example_drugs.keys())
)

if selected_example == "Custom":
    smiles_input = st.sidebar.text_input(
        "Enter SMILES String:",
        value="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
        help="Enter a valid SMILES string"
    )
else:
    smiles_input = example_drugs[selected_example]
    st.sidebar.text_input("SMILES:", value=smiles_input, disabled=True)

# Predict button
predict_button = st.sidebar.button("🔮 Predict", type="primary", use_container_width=True)

# Main content
if predict_button:
    if not smiles_input:
        st.error("Please enter a SMILES string!")
    else:
        # Validate SMILES
        result = compute_molecular_features(smiles_input)
        
        if result is None:
            st.error("❌ Invalid SMILES string. Please check your input.")
        else:
            molecular_feats, mol = result
            
            # Make prediction
            with st.spinner("🔬 Analyzing compound..."):
                efficacy, cns_prob, side_effects = predict(model, smiles_input, molecular_feats)
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Layout: 3 columns
            col1, col2, col3 = st.columns([1, 1, 1])
            
            # Column 1: Molecular Structure
            with col1:
                st.subheader("🔬 Molecular Structure")
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img, use_container_width=True)
                
                st.markdown("**SMILES:**")
                st.code(smiles_input, language=None)
            
            # Column 2: Predictions
            with col2:
                st.subheader("📊 Predictions")
                
                # Efficacy score
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric(
                    "Efficacy Score",
                    f"{efficacy:.2f}",
                    help="Predicted drug efficacy (0-10 scale)"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # CNS Activity
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric(
                    "CNS Active Probability",
                    f"{cns_prob:.1%}",
                    help="Probability of crossing blood-brain barrier"
                )
                
                # Progress bar for CNS activity
                st.progress(cns_prob)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Column 3: Molecular Properties
            with col3:
                st.subheader("⚗️ Molecular Properties")
                
                props = {
                    "Molecular Weight": f"{Descriptors.MolWt(mol):.2f} g/mol",
                    "LogP": f"{Descriptors.MolLogP(mol):.2f}",
                    "TPSA": f"{Descriptors.TPSA(mol):.2f} Ų",
                    "H-Bond Donors": f"{Descriptors.NumHDonors(mol)}",
                    "H-Bond Acceptors": f"{Descriptors.NumHAcceptors(mol)}",
                    "Rotatable Bonds": f"{Descriptors.NumRotatableBonds(mol)}"
                }
                
                for prop, value in props.items():
                    st.text(f"{prop}: {value}")
                
                # BBB penetration (Lipinski's rule)
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                bbb_pass = mw < 450 and logp < 5 and hbd < 5 and hba < 10
                
                st.markdown("---")
                if bbb_pass:
                    st.success("✅ Likely BBB Penetrant")
                else:
                    st.warning("⚠️ May Not Cross BBB")
            
            # Side Effects Section (Full Width)
            st.markdown("---")
            st.subheader("⚠️ Neurological Side Effect Risk Profile")
            
            # Create bar chart
            fig = go.Figure(go.Bar(
                x=list(side_effects.values()),
                y=list(side_effects.keys()),
                orientation='h',
                marker=dict(
                    color=list(side_effects.values()),
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Risk")
                ),
                text=[f"{v:.1%}" for v in side_effects.values()],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Side Effect Risk Assessment",
                xaxis_title="Probability",
                yaxis_title="Side Effect",
                height=400,
                xaxis=dict(range=[0, 1]),
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk interpretation
            st.markdown("### 📋 Risk Interpretation")
            
            risk_cols = st.columns(4)
            for i, (effect, prob) in enumerate(side_effects.items()):
                with risk_cols[i]:
                    if prob > 0.7:
                        st.error(f"**{effect}**\n\n🔴 High Risk ({prob:.1%})")
                    elif prob > 0.4:
                        st.warning(f"**{effect}**\n\n🟡 Moderate Risk ({prob:.1%})")
                    else:
                        st.success(f"**{effect}**\n\n🟢 Low Risk ({prob:.1%})")

else:
    # Landing page
    st.info("👈 Select a drug or enter a SMILES string in the sidebar, then click **Predict**")
    
    # Show example images
    st.subheader("Example Drugs")
    
    ex_cols = st.columns(3)
    
    for i, (name, smiles) in enumerate(list(example_drugs.items())[:3]):
        with ex_cols[i]:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(200, 200))
                st.image(img, caption=name, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>PharmaNeuro Predictor</strong> | Built with Streamlit, PyTorch & Comet ML</p>
    <p>For research and educational purposes only</p>
</div>
""", unsafe_allow_html=True)