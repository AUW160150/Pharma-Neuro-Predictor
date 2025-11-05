import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Compute molecular descriptors
def compute_descriptors(smiles_list):
    descriptors = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            feat = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'AromaticRings': Descriptors.NumAromaticRings(mol),
                # Add more descriptors here as necessary
            }
            descriptors.append(feat)
    return pd.DataFrame(descriptors)

# Merge datasets (ChEMBL, DrugBank)
def merge_datasets(chembl_df, drugbank_df):
    merged = pd.merge(chembl_df, drugbank_df, on="compound_id", how="inner")
    return merged

# Clean the data (handle missing values, etc.)
def clean_data(df):
    df.dropna(subset=['smiles', 'compound_id'], inplace=True)  # Adjust columns as needed
    return df

# Main function to run the entire data pipeline
def run_data_pipeline():
    # Load the datasets
    chembl_df = pd.read_csv("data/raw/chembl_data.csv")
    drugbank_df = pd.read_csv("data/raw/drugbank_data.csv")

    # Merge the datasets
    merged_df = merge_datasets(chembl_df, drugbank_df)

    # Extract molecular descriptors
    smiles_list = merged_df['smiles'].tolist()
    molecular_features = compute_descriptors(smiles_list)

    # Combine the molecular features with the merged data
    full_df = pd.concat([merged_df, molecular_features], axis=1)

    # Clean the data
    cleaned_df = clean_data(full_df)

    # Save the cleaned data to a CSV (or use it for model training)
    cleaned_df.to_csv("data/processed/cleaned_data.csv", index=False)

    return cleaned_df

# Run the data pipeline
if __name__ == "__main__":
    run_data_pipeline()
