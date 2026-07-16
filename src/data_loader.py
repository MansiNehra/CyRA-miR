"""
Data loading and dataset preparation utilities.

This module loads the processed miRNA expression dataset,
extracts features and labels, performs the train-holdout split,
and returns all objects required for downstream analysis.
"""

import config

# DATA LOADING
def load_data(file_path):
    
    # READ CSV
    df = config.pd.read_csv(file_path)

    # SAMPLE IDS
    if "sample_id" in df.columns:
        sample_ids = df["sample_id"].astype(str)
        df = df.drop(columns=["sample_id"])

    else:
        sample_ids = config.pd.Series(
            [f"S{i}" for i in range(len(df))]
        )

    # FEATURES & LABELS
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    
    # TRAIN / HOLDOUT SPLIT 
    X_train, X_hold, y_train, y_hold = config.train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE
    )

    print("=" * 70)
    print("Dataset Loaded Successfully")
    print("=" * 70)
    print(f"Samples  : {len(df)}")
    print(f"Features : {X.shape[1]}")
    print("\nLabel Distribution")
    print(y.value_counts())
    print("\nTraining")
    print(y_train.value_counts())
    print("\nHoldout")
    print(y_hold.value_counts())

    return (
        X,
        y,
        X_train,
        X_hold,
        y_train,
        y_hold,
        sample_ids
    )