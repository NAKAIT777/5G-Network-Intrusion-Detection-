import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# CHANGED: centralized list of columns that leak the answer or won't generalize
# across 5G networks. 'Offset' was missing here before — testing showed it alone
# inflates AUC from ~0.87 to ~0.999 because it correlates with *capture session*
# (i.e. when a flow was recorded), not real attack behavior.
LEAKAGE_COLS = [
    'Label', 'Attack Type', 'Attack Tool', 'Seq', 'Offset', 'RunTime',
    'Src IP', 'Dst IP', 'saddr', 'daddr', 'Unnamed: 0'
]

def load_and_clean_data(file_path):
    print(f"Loading dataset from {file_path}...")
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Check the file name and path.")
        return None, None, None

    print(f"\nOriginal Dataset Shape: {df.shape}")
    
    # 1. Identify Target Column
    # 5G-NIDD typically uses 'Label' or 'Attack Type'. Adjust if yours differs.
    target_col = 'Label' 
    if target_col not in df.columns:
        target_col = df.columns[-1] # Fallback to the last column
        
    print(f"Using '{target_col}' as the target variable.")
    
    y = df[target_col]

    # CHANGED: drop all known leakage columns here, not just the target.
    # Previously this only dropped `target_col`, so 'Offset', 'Seq', 'Attack Type'
    # etc. all stayed in X and leaked into the model.
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} leakage/target column(s): {cols_to_drop}")

    # 2. Handle Missing Values
    # NOTE: this fills medians using the WHOLE dataset. That's fine for this
    # exploratory ingestion pass, but phase2_imbalance.py (the pipeline actually
    # used for modeling) fits medians on the training split only — see the
    # CHANGED note there. Don't feed this version's X straight into a model.
    print("Handling missing values...")
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    
    # Fill text/categorical columns with the most frequent value (mode)
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # 3. Encode Categorical Features
    print("Encoding categorical network features...")
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # 4. Encode the Target Variable
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)
    
    print("\n--- Target Class Distribution ---")
    class_counts = y.value_counts()
    for class_name, count in class_counts.items():
        percentage = (count / len(y)) * 100
        print(f"{class_name}: {count} rows ({percentage:.2f}%)")
    
    print("\n✅ Phase 1 Complete! Data is cleaned, encoded, and ready.")
    return X, y_encoded, y_encoder

if __name__ == "__main__":
    # IMPORTANT: Update 'Combined.csv' to the actual name of your dataset file
    dataset_filename = 'Combined.csv' 
    X, y, target_encoder = load_and_clean_data(dataset_filename)