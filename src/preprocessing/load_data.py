"""
Phase 1: Data loading & exploration.

Supports both CSV (original CICIDS2017/NSL-KDD) and Parquet
(e.g. the dhoogla/cicids2017 cleaned version on Kaggle) files
placed in data/raw/.

Usage:
    python src/preprocessing/load_data.py
"""

import glob
import os

import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def load_all_files(raw_dir: str) -> pd.DataFrame:
    """Load and concatenate every CSV or Parquet file in raw_dir into one DataFrame."""
    parquet_files = glob.glob(os.path.join(raw_dir, "*.parquet"))
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))

    if not parquet_files and not csv_files:
        raise FileNotFoundError(
            f"No CSV or Parquet files found in {raw_dir}. "
            "Download CICIDS2017 and place the files there."
        )

    frames = []

    for f in parquet_files:
        print(f"Loading {os.path.basename(f)} ...")
        df = pd.read_parquet(f)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)

    for f in csv_files:
        print(f"Loading {os.path.basename(f)} ...")
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined shape: {combined.shape}")
    return combined


def find_label_column(df: pd.DataFrame) -> str:
    """Try common label column names used across CICIDS2017 versions."""
    candidates = ["Label", "label", "Label ", " Label", "attack_cat", "class"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def explore(df: pd.DataFrame) -> None:
    """Print basic exploratory stats: shape, class balance, missing values."""
    print("\n--- Basic info ---")
    print(df.info(memory_usage="deep"))

    label_col = find_label_column(df)
    if label_col:
        print(f"\n--- Class distribution ({label_col}) ---")
        print(df[label_col].value_counts())
        print("\n--- Class distribution (%) ---")
        print((df[label_col].value_counts(normalize=True) * 100).round(3))
    else:
        print(f"\nWarning: couldn't auto-detect the label column. "
              f"Available columns: {list(df.columns)}")

    print("\n--- Missing values (top 15 columns) ---")
    print(df.isnull().sum().sort_values(ascending=False).head(15))

    print("\n--- Duplicate rows ---")
    print(df.duplicated().sum())


def save_combined(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved combined data to {out_path}")


if __name__ == "__main__":
    df = load_all_files(RAW_DATA_DIR)
    explore(df)
    save_combined(df, os.path.join(PROCESSED_DATA_DIR, "combined_raw.parquet"))
