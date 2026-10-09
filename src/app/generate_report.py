"""
Phase 6: Generate a SOC incident report.

Full pipeline for ONE network flow:
  1. Pick a flow from the test set (optionally of a chosen true label)
  2. XGBoost predicts the attack type + confidence
  3. SHAP finds the top features that drove the prediction
  4. RAG retrieves the matching MITRE ATT&CK entry
  5. LLM (Groq, free) writes a structured report grounded ONLY in
     the information from steps 2-4

Usage (from the project root, venv active):
    python src\\app\\generate_report.py
    python src\\app\\generate_report.py "SSH-Patator"
    python src\\app\\generate_report.py "DDoS"

Needs a .env file in the project root containing:
    GROQ_API_KEY=your_key_here
"""

import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.rag.retrieve import KnowledgeRetriever  # noqa: E402

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K_FEATURES = 6

# Known limitation found in our own analysis (confusion matrix + SHAP).
KNOWN_CONFUSIONS = {
    "Web Attack - XSS": "Web Attack - Brute Force",
    "Web Attack - Brute Force": "Web Attack - XSS",
}


def softmax(margins: np.ndarray) -> np.ndarray:
    e = np.exp(margins - margins.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def pick_flow(X_test, y_test, wanted_label=None, seed=7):
    rng = np.random.default_rng(seed)
    labels = y_test.values
    if wanted_label:
        idx = np.where(labels == wanted_label)[0]
        if len(idx) == 0:
            print(f"No test rows with label '{wanted_label}'. Available labels:")
            print(sorted(set(labels)))
            sys.exit(1)
    else:
        idx = np.where(labels != "Benign")[0]
    return int(rng.choice(idx))


def analyse_flow(row_idx, X_test, y_test, feature_names, model, le, scaler):
    X_row_scaled = X_test[row_idx : row_idx + 1]
    X_row_df = pd.DataFrame(X_row_scaled, columns=feature_names)

    # Probabilities from raw margins (robust for softmax objective)
    margins = model.predict(X_row_scaled, output_margin=True)
    probs = softmax(np.asarray(margins))[0]
    pred_idx = int(probs.argmax())
    pred_label = le.classes_[pred_idx]
    confidence = float(probs[pred_idx])

    order = probs.argsort()[::-1][:3]
    top_classes = [(le.classes_[i], float(probs[i])) for i in order]

    # SHAP for the predicted class
    explainer = shap.TreeExplainer(model)
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
        "true_label": y_test.values[row_idx],
        "pred_label": pred_label,
        "confidence": confidence,
        "top_classes": top_classes,
        "top_features": top_features,
    }


def build_prompt(analysis, kb_entry):
    feat_lines = "\n".join(
        f"- {f['feature']} = {f['raw_value']:.4g} ({f['direction']} '{analysis['pred_label']}', SHAP {f['shap']:+.2f})"
        for f in analysis["top_features"]
    )
    alt_lines = "\n".join(f"- {c}: {p:.1%}" for c, p in analysis["top_classes"])

    caveat = ""
    confused = KNOWN_CONFUSIONS.get(analysis["pred_label"])
    if confused:
        caveat = (
            f"\nKNOWN MODEL LIMITATION: this detector often confuses '{analysis['pred_label']}' "
            f"with '{confused}' because it only sees network-flow statistics, not HTTP payloads. "
            "Mention that an analyst should verify using web server logs."
        )

    kb_text = (
        f"Attack: {kb_entry['attack_label']}\n"
        f"MITRE ATT&CK: {kb_entry['mitre_technique_id']} - {kb_entry['mitre_technique_name']}\n"
        f"Description: {kb_entry['description']}\n"
        f"Detection indicators: {'; '.join(kb_entry['detection_indicators'])}\n"
        f"Mitigation: {'; '.join(kb_entry['mitigation'])}"
    )

    return f"""You are a SOC analyst assistant. Write an incident report using ONLY the information below.
Do not invent IP addresses, timestamps, hostnames or facts that are not provided.

DETECTION RESULT
- Predicted attack: {analysis['pred_label']}
- Model confidence: {analysis['confidence']:.1%}
- Top model candidates:
{alt_lines}

EVIDENCE (top features from SHAP explainability, raw values from the flow)
{feat_lines}

KNOWLEDGE BASE (MITRE ATT&CK grounded reference)
{kb_text}
{caveat}

Write the report in Markdown with exactly these sections:
1. Summary (2 sentences)
2. Severity (Low/Medium/High/Critical, with a one-line justification)
3. What the evidence shows (explain the SHAP features in plain English)
4. MITRE ATT&CK mapping
5. Recommended actions (prioritised, taken from the knowledge base)
6. Confidence and limitations
"""


def call_llm(prompt):
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found. Create a .env file in the project root with:")
        print("GROQ_API_KEY=your_key_here")
        sys.exit(1)

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You write precise, factual security incident reports."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    return response.choices[0].message.content


def main():
    wanted = sys.argv[1] if len(sys.argv) > 1 else None

    X_test = joblib.load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
    y_test = joblib.load(os.path.join(PROCESSED_DIR, "y_test.pkl"))
    feature_names = joblib.load(os.path.join(PROCESSED_DIR, "feature_names.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    row_idx = pick_flow(X_test, y_test, wanted)
    analysis = analyse_flow(row_idx, X_test, y_test, feature_names, model, le, scaler)

    print(f"Flow #{row_idx} | true label: {analysis['true_label']} | "
          f"predicted: {analysis['pred_label']} ({analysis['confidence']:.1%})")

    if analysis["pred_label"] == "Benign":
        print("Flow classified as Benign - no incident report needed.")
        return

    retriever = KnowledgeRetriever()
    kb_entry = retriever.get_by_label(analysis["pred_label"])
    if kb_entry is None:
        print(f"No knowledge base entry for '{analysis['pred_label']}'.")
        return

    prompt = build_prompt(analysis, kb_entry)
    print("Calling LLM...\n")
    report = call_llm(prompt)

    print("=" * 70)
    print(report)
    print("=" * 70)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REPORTS_DIR, f"report_{analysis['pred_label'].replace(' ', '_')}_{stamp}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
