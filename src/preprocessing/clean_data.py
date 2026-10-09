"""
Phase 2: Cleaning & preprocessing.

Takes the combined raw data (from load_data.py) and:
  1. Drops duplicate rows
  2. Fixes broken label text (encoding issues in attack names)
  3. Handles infinite/NaN values (Flow Bytes/s and Flow Packets/s
     can be inf when Flow Duration is 0)
  4. Groups extremely rare classes so training doesn't crash on
     classes with only 1-2 examples
  5. Splits into train/test with stratification (so rare attack
     types appear in both train and test, not just one)
  6. Scales numeric features

Usage:
    python src/preprocessing/clean_data.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

INPUT_PATH = os.path.join(PROCESSED_DIR, "combined_raw.parquet")

# Below this many total examples, a class is too rare to reliably
# stratify/train on separately -> group into "Rare Attack".
MIN_CLASS_COUNT = 50


def fix_label_text(label: str) -> str:
    """Fix mangled unicode dash characters seen in CICIDS2017 Web Attack labels."""
    if not isinstance(label, str):
        return label
    return (
        label.replace("\ufffd", "-")
        .replace("�", "-")
        .strip()
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Starting shape: {df.shape}")

    # 1. Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows -> {len(df)} rows remain")

    # 2. Fix label text
    df["Label"] = df["Label"].apply(fix_label_text)

    # 3. Handle inf/NaN (Flow Bytes/s, Flow Packets/s can be inf when duration = 0)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    print(f"Found {inf_count} infinite values -> replacing with NaN, then dropping those rows")
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    print(f"Dropped {before - len(df)} rows with inf/NaN -> {len(df)} rows remain")

    # 4. Group extremely rare classes
    counts = df["Label"].value_counts()
    rare_classes = counts[counts < MIN_CLASS_COUNT].index.tolist()
    if rare_classes:
        print(f"\nGrouping {len(rare_classes)} rare classes (<{MIN_CLASS_COUNT} examples) "
              f"into 'Rare Attack': {rare_classes}")
        df["Label"] = df["Label"].apply(lambda x: "Rare Attack" if x in rare_classes else x)

    print("\nFinal class distribution:")
    print(df["Label"].value_counts())

    return df


def split_and_scale(df: pd.DataFrame):
    X = df.drop(columns=["Label"])
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_PATH)
    df = clean(df)

    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(X_train, os.path.join(PROCESSED_DIR, "X_train.pkl"))
    joblib.dump(X_test, os.path.join(PROCESSED_DIR, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(PROCESSED_DIR, "y_train.pkl"))
    joblib.dump(y_test, os.path.join(PROCESSED_DIR, "y_test.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(PROCESSED_DIR, "feature_names.pkl"))

    print("\nSaved: X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl, scaler.pkl, feature_names.pkl")