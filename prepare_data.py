import sys
sys.path.append('src')

import pandas as pd
from data.data_loader import MolecularFeatureExtractor

def prepare_training_data():
    '''Prepare data with molecular features'''
    
    print('Loading dataset...')
    df = pd.read_csv('data/raw/cns_drugs_demo.csv')
    
    print('Extracting molecular features...')
    extractor = MolecularFeatureExtractor()
    
    # Get unique SMILES
    unique_smiles = df['smiles'].unique()
    
    # Extract features
    features_df = extractor.batch_extract(unique_smiles.tolist())
    
    # Merge features back to main dataframe
    # For each SMILES, add the molecular features
    feature_cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'AromaticRings', 'FractionCsp3']
    
    # Create a mapping from SMILES to features
    smiles_to_features = {}
    for _, row in features_df.iterrows():
        smiles = row['smiles']
        smiles_to_features[smiles] = {col: row[col] for col in feature_cols}
    
    # Add features to main dataframe
    for col in feature_cols:
        df[col] = df['smiles'].map(lambda s: smiles_to_features.get(s, {}).get(col, 0))
    
    # Save enhanced dataset
    output_path = 'data/processed/cns_drugs_with_features.csv'
    df.to_csv(output_path, index=False)
    
    print(f'\n✅ Saved enhanced dataset to {output_path}')
    print(f'Dataset shape: {df.shape}')
    print(f'Columns: {df.columns.tolist()}')
    
    return df

if __name__ == '__main__':
    prepare_training_data()