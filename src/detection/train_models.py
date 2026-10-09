"""
Phase 3: Train detection models.

Trains two models on the preprocessed data (from clean_data.py):

  1. Isolation Forest (unsupervised anomaly detection)
     - Trained ONLY on Benign traffic, so it learns what "normal"
       looks like. Anything that deviates enough gets flagged as
       an anomaly. This doesn't know attack TYPES, just "normal vs weird".

  2. XGBoost (supervised multi-class classification)
     - Trained on all labeled classes, learns to tell DoS Hulk apart
       from DDoS apart from PortScan, etc. Uses sample weighting to
       fight the class imbalance (so rare attacks aren't ignored).

Both get evaluated with precision/recall/F1 PER CLASS, not just
overall accuracy -- overall accuracy is misleading here since Benign
dominates the data.

Usage:
    python src/detection/train_models.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import xgboost as xgb

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def load_data():
    X_train = joblib.load(os.path.join(PROCESSED_DIR, "X_train.pkl"))
    X_test = joblib.load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
    y_train = joblib.load(os.path.join(PROCESSED_DIR, "y_train.pkl"))
    y_test = joblib.load(os.path.join(PROCESSED_DIR, "y_test.pkl"))
    feature_names = joblib.load(os.path.join(PROCESSED_DIR, "feature_names.pkl"))
    return X_train, X_test, y_train, y_test, feature_names


# ---------------------------------------------------------------------
# 1. Isolation Forest: unsupervised anomaly detection
# ---------------------------------------------------------------------
def train_isolation_forest(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 70)
    print("ISOLATION FOREST (unsupervised anomaly detection)")
    print("=" * 70)

    # Train ONLY on benign traffic -> model learns what "normal" looks like
    benign_mask_train = (y_train == "Benign").values
    X_train_benign = X_train[benign_mask_train]
    print(f"Training on {X_train_benign.shape[0]} benign-only samples "
          f"(out of {X_train.shape[0]} total training rows)")

    # contamination = expected proportion of anomalies. We estimate this
    # from the real attack ratio in training data.
    attack_ratio = 1 - benign_mask_train.mean()
    contamination = min(max(attack_ratio, 0.01), 0.5)  # keep it in a sane range
    print(f"Using contamination={contamination:.4f} (estimated attack ratio in training data)")

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_benign)

    # Predict on test set: -1 = anomaly (attack), 1 = normal (benign)
    preds = iso_forest.predict(X_test)
    y_pred_binary = np.where(preds == -1, "Attack", "Benign")
    y_true_binary = np.where((y_test == "Benign").values, "Benign", "Attack")

    print("\n--- Isolation Forest: Binary (Attack vs Benign) evaluation ---")
    print(classification_report(y_true_binary, y_pred_binary, digits=3))

    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    print(f"Saved isolation_forest.pkl")

    return iso_forest


# ---------------------------------------------------------------------
# 2. XGBoost: supervised multi-class attack classification
# ---------------------------------------------------------------------
def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 70)
    print("XGBOOST (supervised multi-class classification)")
    print("=" * 70)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Sample weighting fights class imbalance: rare classes get higher
    # weight per example, so the model doesn't just learn to always
    # predict "Benign".
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_enc)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )

    print(f"Training on {X_train.shape[0]} rows, {len(le.classes_)} classes...")
    model.fit(X_train, y_train_enc, sample_weight=sample_weights)

    y_pred_enc = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    print("\n--- XGBoost: Per-class evaluation ---")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    print("\n--- Precision / Recall / F1 summary (sorted by support) ---")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=le.classes_, zero_division=0
    )
    summary = pd.DataFrame({
        "class": le.classes_,
        "precision": precision.round(3),
        "recall": recall.round(3),
        "f1": f1.round(3),
        "support": support,
    }).sort_values("support", ascending=False)
    print(summary.to_string(index=False))

    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print(f"\nSaved xgboost_model.pkl and label_encoder.pkl")

    return model, le


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test, feature_names = load_data()

    iso_forest = train_isolation_forest(X_train, y_train, X_test, y_test)
    xgb_model, label_encoder = train_xgboost(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 70)
    print("DONE. Both models trained and saved to /models")
    print("=" * 70)