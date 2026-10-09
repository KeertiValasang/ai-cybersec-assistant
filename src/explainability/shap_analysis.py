"""
Phase 4: SHAP explainability.

Explains the XGBoost model's predictions using SHAP (SHapley Additive
exPlanations). This answers "WHY did the model predict this?" at both
a global level (which features matter most overall) and a per-prediction
level (which features drove THIS specific prediction).

We specifically dig into the XSS vs Brute Force confusion found in
confusion_analysis.py, to see which features the model leans on when
it gets these two confused.

NOTE: SHAP on the full 446k-row test set would be very slow. We use a
stratified sample instead -- this is standard practice, not cutting
corners: SHAP values are about explaining model *behavior*, and a
representative sample is enough to characterize that.

Usage:
    python src/explainability/shap_analysis.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import shap

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

SAMPLE_SIZE = 3000  # rows to sample for SHAP (keeps runtime reasonable)
TOP_N_FEATURES = 15


def load_everything():
    X_test = joblib.load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
    y_test = joblib.load(os.path.join(PROCESSED_DIR, "y_test.pkl"))
    feature_names = joblib.load(os.path.join(PROCESSED_DIR, "feature_names.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    return X_test, y_test, feature_names, model, le


def stratified_sample(X, y, feature_names, sample_size, random_state=42):
    """Sample rows, trying to keep some representation from every class
    (important since rare classes would otherwise vanish from a random sample)."""
    df = pd.DataFrame(X, columns=feature_names)
    df["_label"] = y.values

    frames = []
    classes = df["_label"].unique()
    per_class = max(sample_size // len(classes), 5)

    for cls in classes:
        cls_rows = df[df["_label"] == cls]
        n = min(per_class, len(cls_rows))
        frames.append(cls_rows.sample(n=n, random_state=random_state))

    sampled = pd.concat(frames).sample(frac=1, random_state=random_state)  # shuffle
    labels = sampled["_label"]
    features = sampled.drop(columns=["_label"])
    return features, labels


def global_importance(shap_values, feature_names, class_names):
    """Print top features overall (averaged across all classes)."""
    # shap_values shape for multiclass: (n_samples, n_features, n_classes)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))
    importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)

    print("\n" + "=" * 70)
    print(f"TOP {TOP_N_FEATURES} MOST IMPORTANT FEATURES (overall, across all classes)")
    print("=" * 70)
    print(importance.head(TOP_N_FEATURES).to_string())
    return importance


def xss_vs_bruteforce_deepdive(shap_values, X_sample, labels, feature_names, class_names):
    """Compare which features drive XSS vs Brute Force predictions specifically --
    this directly investigates the confusion found earlier."""
    print("\n" + "=" * 70)
    print("DEEP DIVE: Web Attack - XSS vs Web Attack - Brute Force")
    print("=" * 70)

    for target_class in ["Web Attack - XSS", "Web Attack - Brute Force"]:
        if target_class not in class_names:
            continue
        class_idx = list(class_names).index(target_class)
        mask = (labels == target_class).values
        if mask.sum() == 0:
            print(f"\nNo samples for '{target_class}' in this sample -- skipping.")
            continue

        # SHAP values for this class's predictions, on rows truly belonging to this class
        class_shap = shap_values.values[mask, :, class_idx]
        mean_abs = np.abs(class_shap).mean(axis=0)
        top_features = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False).head(8)

        print(f"\n--- Top features driving '{target_class}' predictions ({mask.sum()} samples) ---")
        print(top_features.to_string())


def save_summary_csv(importance, out_path):
    importance.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"}).to_csv(
        out_path, index=False
    )
    print(f"\nSaved feature importance to {out_path}")


if __name__ == "__main__":
    X_test, y_test, feature_names, model, le = load_everything()

    print(f"Sampling ~{SAMPLE_SIZE} rows (stratified across classes) for SHAP analysis...")
    X_sample, y_sample = stratified_sample(X_test, y_test, feature_names, SAMPLE_SIZE)
    print(f"Sampled {len(X_sample)} rows.")

    print("\nComputing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    importance = global_importance(shap_values, feature_names, le.classes_)
    xss_vs_bruteforce_deepdive(shap_values, X_sample, y_sample, feature_names, le.classes_)

    out_path = os.path.join(PROCESSED_DIR, "shap_feature_importance.csv")
    save_summary_csv(importance, out_path)

    print("\nDone.")
