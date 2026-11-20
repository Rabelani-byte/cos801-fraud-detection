
"""Reusable preprocessing pipeline for COS801 fraud detection."""

import os
import pandas as pd
import numpy as np
import kagglehub

from sklearn.preprocessing import OrdinalEncoder


def load_fraud_data():
    """Download and load IEEE-CIS transaction data from Kaggle."""

    dataset_path = kagglehub.competition_download(
        "ieee-fraud-detection"
    )

    file_path = os.path.join(
        dataset_path,
        "train_transaction.csv"
    )

    df = pd.read_csv(file_path)

    # Reduce memory usage, as in Notebook 02
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    return df


def temporal_split(df):
    """Split transactions chronologically into train, validation and test."""

    # Sort transactions from earliest to latest
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Calculate split positions
    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)

    # Create three separate datasets
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print("\nTEMPORAL DATA SPLITTING")
    print("-" * 45)

    print("Training set:", train_df.shape)
    print("Validation set:", val_df.shape)
    print("Testing set:", test_df.shape)

    # Verify chronological order
    assert train_df["TransactionDT"].max() <= val_df["TransactionDT"].min()
    assert val_df["TransactionDT"].max() <= test_df["TransactionDT"].min()

    print("\nChronological order verified!")

    return train_df, val_df, test_df


def preprocess_features(train_df, val_df, test_df):
    """Fit preprocessing on training data and apply to all splits."""

    excluded_cols = ["isFraud", "TransactionID", "TransactionDT"]

    feature_cols = [
        col for col in train_df.columns
        if col not in excluded_cols
    ]

    # Identify columns with more than 90% missing values
    missing_percentage = (
        train_df[feature_cols].isna().mean() * 100
    )

    high_missing_cols = missing_percentage[
        missing_percentage > 90
    ].index.tolist()

    # Remove the same columns from all datasets
    train_df = train_df.drop(columns=high_missing_cols).copy()
    val_df = val_df.drop(columns=high_missing_cols).copy()
    test_df = test_df.drop(columns=high_missing_cols).copy()

    # Update feature list
    feature_cols = [
        col for col in train_df.columns
        if col not in excluded_cols
    ]

    # Identify numerical and categorical features
    numerical_features = train_df[feature_cols].select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = train_df[feature_cols].select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Calculate medians using training data only
    train_medians = train_df[numerical_features].median().fillna(0)

    # Fill missing numerical and categorical values
    for data in [train_df, val_df, test_df]:
        data[numerical_features] = data[numerical_features].fillna(
            train_medians
        )
        data[categorical_features] = data[categorical_features].fillna(
            "Missing"
        )

    # Fit categorical encoder using training data only
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float32
    )

    encoder.fit(train_df[categorical_features].astype(str))

    # Encode categorical features in all datasets
    for data in [train_df, val_df, test_df]:
        data[categorical_features] = encoder.transform(
            data[categorical_features].astype(str)
        )

    print("\nPREPROCESSING COMPLETED")
    print("-" * 45)
    print("Removed features:", high_missing_cols)
    print("Numerical features:", len(numerical_features))
    print("Categorical features:", len(categorical_features))

    return train_df, val_df, test_df


def prepare_model_data(train_df, val_df, test_df):
    """Prepare model inputs, fraud labels and transaction times."""

    # Exclude target, identifier and transaction time
    excluded_cols = ["isFraud", "TransactionID", "TransactionDT"]

    feature_cols = [
        col for col in train_df.columns
        if col not in excluded_cols
    ]

    # Prepare model inputs
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    # Prepare fraud labels
    y_train = train_df["isFraud"]
    y_val = val_df["isFraud"]
    y_test = test_df["isFraud"]

    # Preserve transaction times for temporal evaluation
    time_train = train_df["TransactionDT"]
    time_val = val_df["TransactionDT"]
    time_test = test_df["TransactionDT"]

    # Verify matching features
    assert X_train.columns.equals(X_val.columns)
    assert X_train.columns.equals(X_test.columns)

    # Verify no missing model inputs
    assert not X_train.isna().any().any()
    assert not X_val.isna().any().any()
    assert not X_test.isna().any().any()

    print("\nMODEL DATA PREPARED")
    print("-" * 45)
    print("Training features:", X_train.shape)
    print("Validation features:", X_val.shape)
    print("Testing features:", X_test.shape)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "time_train": time_train,
        "time_val": time_val,
        "time_test": time_test,
        "feature_cols": feature_cols
    }


def run_preprocessing_pipeline():
    """Run the complete fraud data preprocessing pipeline."""

    df = load_fraud_data()

    train_df, val_df, test_df = temporal_split(df)

    del df

    train_df, val_df, test_df = preprocess_features(
        train_df, val_df, test_df
    )

    model_data = prepare_model_data(
        train_df, val_df, test_df
    )

    return model_data
