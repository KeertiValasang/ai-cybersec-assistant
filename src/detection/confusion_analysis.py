"""
Phase 3b: Confusion matrix analysis.

Loads the trained XGBoost model and test set, then shows exactly
WHAT the weak classes (Bot, Web Attack - XSS) are being confused
with -- this is what turns "the model struggles with X" into a
specific, explainable finding.

Usage:
    python src/detection/confusion_analysis.py
"""

import os

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def main():
    X_test = joblib.load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
    y_test = joblib.load(os.path.join(PROCESSED_DIR, "y_test.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    y_pred_enc = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    # Focus on the weak classes we already identified
    weak_classes = ["Bot", "Web Attack - XSS", "Web Attack - Brute Force", "Rare Attack"]

    print("=" * 70)
    print("CONFUSION MATRIX: where do the weak classes' mistakes actually go?")
    print("=" * 70)

    for cls in weak_classes:
        if cls not in cm_df.index:
            continue
        row = cm_df.loc[cls]
        total = row.sum()
        # show what this class actually got predicted as, ranked
        confused_with = row[row > 0].sort_values(ascending=False)
        print(f"\n--- True label: '{cls}' ({total} total test examples) ---")
        for pred_label, count in confused_with.items():
            pct = 100 * count / total
            marker = "  <- correct" if pred_label == cls else ""
            print(f"  predicted as '{pred_label}': {count} ({pct:.1f}%){marker}")

    # Also save full confusion matrix to CSV for reference
    out_path = os.path.join(PROCESSED_DIR, "confusion_matrix.csv")
    cm_df.to_csv(out_path)
    print(f"\nFull confusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()