"""
Phase 7: Streamlit dashboard.

A clickable web UI for the whole pipeline:
  - Pick a flow from the existing test set (demo mode), OR upload a new
    CSV with the same 77 feature columns
  - See the model's prediction + confidence
  - See the SHAP explanation as a bar chart
  - See the matched MITRE ATT&CK knowledge base entry
  - Generate a full LLM SOC report on demand (button, not automatic,
    to avoid burning API calls on every click)

Usage (from the project root, venv active):
    streamlit run src\\app\\dashboard.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.rag.retrieve import KnowledgeRetriever  # noqa: E402
from src.app.generate_report import (  # noqa: E402
    softmax,
    build_prompt,
    call_llm,
    KNOWN_CONFUSIONS,
    TOP_K_FEATURES,
)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

st.set_page_config(page_title="AI Cyber Security Assistant", layout="wide")


@st.cache_resource
def load_artifacts():
    X_test = joblib.load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
    y_test = joblib.load(os.path.join(PROCESSED_DIR, "y_test.pkl"))
    feature_names = joblib.load(os.path.join(PROCESSED_DIR, "feature_names.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return X_test, y_test, feature_names, model, le, scaler


@st.cache_resource
def get_retriever():
    return KnowledgeRetriever()


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


def predict_batch(X_scaled, model, le, n=50, seed=0):
    """Predict on a sample of n rows, return a results dataframe."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_scaled.shape[0], size=min(n, X_scaled.shape[0]), replace=False)
    X_sample = X_scaled[idx]

    margins = model.predict(X_sample, output_margin=True)
    probs = softmax(np.asarray(margins))
    pred_idx = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    pred_labels = le.classes_[pred_idx]

    return idx, pd.DataFrame({
        "row_id": idx,
        "predicted_label": pred_labels,
        "confidence": confidences,
    })


def analyse_single(row_idx, X_scaled, feature_names, model, le, scaler, explainer):
    X_row_scaled = X_scaled[row_idx : row_idx + 1]
    X_row_df = pd.DataFrame(X_row_scaled, columns=feature_names)

    margins = model.predict(X_row_scaled, output_margin=True)
    probs = softmax(np.asarray(margins))[0]
    pred_idx = int(probs.argmax())
    pred_label = le.classes_[pred_idx]
    confidence = float(probs[pred_idx])

    order = probs.argsort()[::-1][:3]
    top_classes = [(le.classes_[i], float(probs[i])) for i in order]

    sv = explainer(X_row_df).values[0, :, pred_idx]
    raw_values = scaler.inverse_transform(X_row_scaled)[0]

    top_feat_idx = np.abs(sv).argsort()[::-1][:TOP_K_FEATURES]
    top_features = [
        {
            "feature": feature_names[i],
            "raw_value": float(raw_values[i]),
            "shap": float(sv[i]),
            "direction": "pushed toward" if sv[i] > 0 else "pushed away from",
        }
        for i in top_feat_idx
    ]

    return {
        "pred_label": pred_label,
        "confidence": confidence,
        "top_classes": top_classes,
        "top_features": top_features,
    }


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("🛡️ AI Cyber Security Assistant")
st.caption("Upload network flow data → detect attacks → explain → get a grounded SOC report")

X_test, y_test, feature_names, model, le, scaler = load_artifacts()
retriever = get_retriever()
explainer = get_explainer(model)

with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose data", ["Use sample test data", "Upload CSV"])

    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "CSV with the same feature columns used in training", type="csv"
        )
        if uploaded is not None:
            raw_df = pd.read_csv(uploaded)
            missing = set(feature_names) - set(raw_df.columns)
            if missing:
                st.error(f"Uploaded file is missing {len(missing)} expected columns, "
                          f"e.g. {list(missing)[:5]}")
                st.stop()
            X_scaled = scaler.transform(raw_df[feature_names])
            active_labels = None
        else:
            st.info("Upload a CSV to continue, or switch to sample test data.")
            st.stop()
    else:
        X_scaled = X_test
        active_labels = y_test
        sample_size = st.slider("Number of flows to scan", 10, 200, 50)
        if "sample_seed" not in st.session_state:
            st.session_state["sample_seed"] = 0
        if st.button("🔄 Shuffle sample"):
            st.session_state["sample_seed"] = np.random.randint(0, 1_000_000)
        sample_seed = st.session_state["sample_seed"]

st.subheader("1. Detection results")

if source == "Use sample test data":
    idx, results_df = predict_batch(X_scaled, model, le, n=sample_size, seed=sample_seed)
else:
    margins = model.predict(X_scaled, output_margin=True)
    probs = softmax(np.asarray(margins))
    pred_idx = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    idx = np.arange(X_scaled.shape[0])
    results_df = pd.DataFrame({
        "row_id": idx,
        "predicted_label": le.classes_[pred_idx],
        "confidence": confidences,
    })

attack_counts = results_df[results_df["predicted_label"] != "Benign"]["predicted_label"].value_counts()
col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(results_df.sort_values("confidence", ascending=False), use_container_width=True, height=300)
with col2:
    st.metric("Flows scanned", len(results_df))
    st.metric("Flagged as attack", int((results_df["predicted_label"] != "Benign").sum()))
    if len(attack_counts) > 0:
        st.bar_chart(attack_counts)

st.subheader("2. Inspect a flow")
attack_rows = results_df[results_df["predicted_label"] != "Benign"]
if len(attack_rows) == 0:
    st.info("No attacks detected in this sample.")
    st.stop()

selected_row_id = st.selectbox(
    "Select a flagged flow to inspect",
    options=attack_rows["row_id"].tolist(),
    format_func=lambda r: f"Row {r} — predicted {results_df.set_index('row_id').loc[r, 'predicted_label']}",
)

analysis = analyse_single(selected_row_id, X_scaled, feature_names, model, le, scaler, explainer)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### Prediction: `{analysis['pred_label']}`")
    st.markdown(f"**Confidence:** {analysis['confidence']:.1%}")
    st.markdown("**Top candidates:**")
    for cls, p in analysis["top_classes"]:
        st.progress(p, text=f"{cls}: {p:.1%}")

with col2:
    st.markdown("### SHAP: top contributing features")
    feat_df = pd.DataFrame(analysis["top_features"])
    feat_df["abs_shap"] = feat_df["shap"].abs()
    st.bar_chart(feat_df.set_index("feature")["shap"])

st.subheader("3. MITRE ATT&CK reference")
kb_entry = retriever.get_by_label(analysis["pred_label"])
if kb_entry:
    st.markdown(f"**{kb_entry['mitre_technique_id']} — {kb_entry['mitre_technique_name']}**")
    st.write(kb_entry["description"])
    with st.expander("Detection indicators & mitigation"):
        st.markdown("**Detection indicators:**")
        for d in kb_entry["detection_indicators"]:
            st.markdown(f"- {d}")
        st.markdown("**Mitigation:**")
        for m in kb_entry["mitigation"]:
            st.markdown(f"- {m}")

    confused = KNOWN_CONFUSIONS.get(analysis["pred_label"])
    if confused:
        st.warning(f"⚠️ This detector sometimes confuses '{analysis['pred_label']}' with "
                   f"'{confused}' — verify with additional logs before acting.")

st.subheader("4. Generate SOC report")
if st.button("Generate full report with LLM", type="primary"):
    with st.spinner("Calling LLM..."):
        # Build a lightweight analysis dict matching generate_report's expected shape
        full_analysis = {
            "true_label": active_labels.values[selected_row_id] if active_labels is not None else "Unknown",
            "pred_label": analysis["pred_label"],
            "confidence": analysis["confidence"],
            "top_classes": analysis["top_classes"],
            "top_features": analysis["top_features"],
        }
        prompt = build_prompt(full_analysis, kb_entry)
        try:
            report = call_llm(prompt)
            st.markdown(report)
        except SystemExit:
            st.error("GROQ_API_KEY not found. Make sure your .env file is set up correctly.")
