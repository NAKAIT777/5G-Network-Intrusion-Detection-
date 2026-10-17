import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN

def load_and_preprocess(file_path):
    print("Loading data with optimized memory options...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Select target column
    if 'Attack Type' in df.columns:
        target_col = 'Attack Type'
    elif 'Label' in df.columns:
        target_col = 'Label'
    else:
        target_col = df.columns[-1]
    
    print(f"Target column selected: '{target_col}'")
    
    # Columns that give away the answer or won't generalize across 5G networks.
    # CHANGED: added 'Offset' and 'RunTime'. Offset especially was NOT being
    # dropped before — it tracks where in the capture a flow sits (essentially
    # which recording session), not real traffic behavior. Verified directly:
    # removing it drops AUC from 0.9992 to 0.8740. That's leakage, not skill.
    leakage_cols = [
        target_col, 'Attack Type', 'Attack Tool', 'Seq', 'Offset', 'RunTime',
        'Src IP', 'Dst IP', 'Label_', 'Attack Type_', 'saddr', 'daddr',
        'Label', 'Unnamed: 0'
    ]
    
    # Drop only the columns that actually exist in your dataframe
    cols_to_drop = [col for col in leakage_cols if col in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    
    print(f"Dropped {len(cols_to_drop)} leakage/target column(s): {cols_to_drop}")
    print(f"Remaining training features count: {X.shape[1]}")

    # CHANGED: numeric missing-value imputation used to happen right here, on the
    # FULL dataset, before the train/test split existed. That means the median
    # used to fill a test-set row was computed partly FROM the test set itself —
    # a small but real leak. It's now done in prepare_data_splits() below, fit
    # on the training split only. Categorical encoding stays here as a scoped
    # simplification (LabelEncoder sees the full category vocabulary); a fully
    # rigorous version would fit it on train only too, but the categorical
    # columns here are low-cardinality enough that this is a minor concern
    # compared to the numeric leak above.

    # Handle object/string columns cleanly
    cat_cols = X.select_dtypes(include=['object', 'string']).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded, label_encoder

def prepare_data_splits(X, y):
    print("\nSplitting data into Train (70%) and Test (30%) with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape:  {X_test.shape}")

    # CHANGED: numeric imputation now happens here, fit on X_train ONLY, and the
    # same train-derived medians are applied to X_test. Previously this was done
    # on the full dataset before the split (in load_and_preprocess), which let
    # test-set values influence the fill for training rows and vice versa.
    X_train = X_train.copy()
    X_test = X_test.copy()
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    train_medians = X_train[num_cols].median()
    X_train[num_cols] = X_train[num_cols].fillna(train_medians)
    X_test[num_cols] = X_test[num_cols].fillna(train_medians)

    # Scale features using training distribution only (prevents data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def apply_resampling(X_train, y_train, method='smote', cap=30000):
    """
    Applies oversampling techniques strictly to the training data.
    Methods: 'smote', 'adasyn', or 'none'
    """
    # CHANGED: by default SMOTE/ADASYN raise EVERY minority class up to the size
    # of the biggest class (~334K rows here), which inflated the training set
    # from ~851K to ~3M rows - slow, memory-hungry, and mostly synthetic data.
    # Now only genuinely rare classes are topped up to `cap` rows each.
    counts = pd.Series(y_train).value_counts()
    sampling_strategy = {cls: cap for cls, cnt in counts.items() if cnt < cap}

    if method == 'smote':
        print(f"\nApplying SMOTE resampling on training set (cap={cap} per rare class)...")
        resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_res, y_res = resampler.fit_resample(X_train, y_train)
    elif method == 'adasyn':
        print(f"\nApplying ADASYN resampling on training set (cap={cap} per rare class)...")
        resampler = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
        X_res, y_res = resampler.fit_resample(X_train, y_train)
    else:
        print("\nUsing original un-resampled training data...")
        X_res, y_res = X_train, y_train
        
    print(f"Resampled Training set shape: {X_res.shape}")
    return X_res, y_res

if __name__ == "__main__":
    file_path = 'Combined.csv'
    X, y, encoder = load_and_preprocess(file_path)
    X_train, X_test, y_train, y_test = prepare_data_splits(X, y)
    
    # Test SMOTE pipeline on training set
    X_train_smote, y_train_smote = apply_resampling(X_train, y_train, method='smote')
    
    print("\n✅ Phase 2 Preprocessing & Oversampling Pipeline ready!")