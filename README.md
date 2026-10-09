# AI Cyber Security Assistant

An AI-powered SOC (Security Operations Center) assistant that ingests network
traffic data (CICIDS2017 flow records), detects attacks using machine
learning, explains *why* each detection fired using SHAP, and generates a
grounded, human-readable incident report using an LLM retrieving from a
MITRE ATT&CK knowledge base.

Built as a final-year AI/ML engineering placement project, combining
classical ML, explainable AI, and GenAI/RAG into one working, demoable
pipeline.

---

## Architecture

```
Upload (CSV) / Sample test data
        │
        ▼
Preprocessing & Feature Engineering   (src/preprocessing/)
   - dedupe, fix label encoding, group rare classes, scale features
        │
        ▼
Detection Layer                        (src/detection/)
   ├── Isolation Forest   (unsupervised anomaly detection)
   └── XGBoost            (supervised multi-class classification,
                            class-weighted for imbalance)
        │
        ▼
Explainability (SHAP)                  (src/explainability/)
   - per-prediction feature attribution
        │
        ▼
RAG over MITRE ATT&CK knowledge base    (src/rag/)
   - ChromaDB + sentence-transformer embeddings
        │
        ▼
LLM (Groq, openai/gpt-oss-120b)
   - grounded SOC report: summary, severity, evidence, MITRE
     mapping, recommended actions, confidence & limitations
        │
        ▼
Streamlit Dashboard                     (src/app/dashboard.py)
```

---

## Dataset

**CICIDS2017** (via Kaggle: `dhoogla/cicids2017`, cleaned parquet version)

- 2,313,810 raw flow records, 78 columns
- 15 original classes (1 Benign + 14 attack types)
- After cleaning: 2,231,806 rows (82,004 duplicates removed)
- Train/test split: 1,785,444 / 446,362 (stratified)

### Class distribution (severe imbalance — the core challenge of this project)

| Class | Count | % |
|---|---|---|
| Benign | 1,895,314 | 84.9% |
| DoS Hulk | 172,846 | 7.7% |
| DDoS | 128,014 | 5.7% |
| DoS GoldenEye | 10,286 | 0.46% |
| FTP-Patator | 5,931 | 0.27% |
| DoS slowloris | 5,385 | 0.24% |
| DoS Slowhttptest | 5,228 | 0.23% |
| SSH-Patator | 3,219 | 0.14% |
| PortScan | 1,956 | 0.09% |
| Web Attack - Brute Force | 1,470 | 0.07% |
| Bot | 1,437 | 0.06% |
| Web Attack - XSS | 652 | 0.03% |
| Rare Attack (Infiltration, SQLi, Heartbleed combined) | 68 | 0.003% |

**Design decision:** Infiltration, SQL Injection, and Heartbleed (11-36
examples each) were grouped into a single "Rare Attack" class. With
single-digit-to-low-double-digit examples, no model can learn a reliable
per-class pattern — grouping them is an honest acknowledgment of this
limitation rather than reporting a misleading per-class score on 1-2 test
examples.

---

## Model performance

### Isolation Forest (unsupervised, trained on Benign-only traffic)

| | Precision | Recall | F1 |
|---|---|---|---|
| Attack | 0.469 | 0.758 | 0.580 |
| Benign | 0.952 | 0.848 | 0.897 |

Weak precision (47%) is expected: Isolation Forest only distinguishes
"normal vs. unusual," with no concept of attack *type*, so it has a high
false-positive rate. This is why it's paired with a supervised classifier
rather than used alone.

### XGBoost (supervised, class-weighted for imbalance)

Overall accuracy: **99.8%** — but accuracy alone is misleading on this
imbalanced data, so every class was evaluated individually:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Benign | 1.000 | 0.999 | 0.999 | 379,064 |
| DDoS | 0.999 | 1.000 | 1.000 | 25,603 |
| DoS Hulk | 0.998 | 0.999 | 0.998 | 34,569 |
| DoS GoldenEye | 0.987 | 0.998 | 0.993 | 2,057 |
| FTP-Patator | 0.997 | 0.999 | 0.998 | 1,186 |
| SSH-Patator | 1.000 | 0.995 | 0.998 | 644 |
| DoS slowloris | 0.987 | 0.996 | 0.992 | 1,077 |
| DoS Slowhttptest | 0.942 | 0.993 | 0.967 | 1,046 |
| PortScan | 0.915 | 0.987 | 0.950 | 391 |
| Rare Attack | 0.812 | 0.929 | 0.867 | 14 |
| Web Attack - Brute Force | 0.749 | 0.782 | 0.765 | 294 |
| Bot | 0.485 | 0.979 | 0.649 | 287 |
| Web Attack - XSS | 0.449 | 0.408 | 0.427 | 130 |

---

## Key finding: Web Attack - XSS vs Brute Force confusion

Confusion matrix analysis showed a two-way mix-up:
- 51.5% of true XSS attacks were classified as Brute Force
- 20.7% of true Brute Force attacks were classified as XSS

SHAP analysis **proved why**: the top 6 features driving each class's
predictions overlap almost completely (`Init Bwd Win Bytes`, `Fwd IAT Min`,
`Flow IAT Min`, `Fwd IAT Mean`, `Flow Packets/s` all rank highly for both).
Both attacks are delivered over HTTP as short bursts of requests with
similar timing — and the dataset's flow-level features (packet timing,
sizes, TCP window size) don't capture HTTP *payload content*, so the model
has no signal to distinguish `<script>` injection from credential-guessing
at the network-flow level.

**This is a dataset/feature limitation, not a model bug** — confirmed with
SHAP evidence, not assumed. The system surfaces this automatically: when
either class is predicted, the generated report includes a warning that
the finding should be verified against web server/application logs.

---

## Knowledge base & RAG

13 hand-curated knowledge base entries, each mapped to a real MITRE ATT&CK
technique ID (e.g. T1110 Brute Force, T1498 Network DoS, T1046 Network
Service Discovery), with detection indicators and mitigation steps.
Embedded with `sentence-transformers/all-MiniLM-L6-v2` into a persistent
ChromaDB collection, enabling both exact-label lookup and free-text
semantic search (e.g. querying "lots of failed logins" correctly retrieves
the Brute Force entry without any keyword overlap).

---

## LLM report generation

Each report is generated by sending the LLM (Groq, `openai/gpt-oss-120b`,
free tier) **only**: the model's prediction + confidence, the top SHAP
features for that specific flow, and the retrieved knowledge base entry —
explicitly instructed not to invent facts (IPs, timestamps, hostnames) not
present in that data. Reports are structured: Summary, Severity, Evidence
(SHAP explained in plain English), MITRE mapping, Recommended actions,
Confidence & limitations.

---

## Dashboard

Streamlit app (`src/app/dashboard.py`) providing:
- Sample test data scanning (with a shuffle button) or CSV upload of new
  flow data
- A detection results table with confidence scores and attack-type counts
- Per-flow inspection: top-3 candidate labels, SHAP bar chart
- MITRE ATT&CK reference panel, with the XSS/Brute Force limitation
  warning surfaced automatically when relevant
- On-demand full LLM report generation

---

## Project structure

```
data/raw/              # original downloaded parquet files
data/processed/         # cleaned/split/scaled data, confusion matrix, SHAP CSV
models/                 # trained models (isolation_forest, xgboost, scaler, label_encoder)
knowledge_base/         # MITRE ATT&CK KB (JSON) + ChromaDB vector store
reports/                # generated SOC reports (Markdown)
src/preprocessing/      # load_data.py, clean_data.py
src/detection/          # train_models.py, confusion_analysis.py
src/explainability/     # shap_analysis.py
src/rag/                # build_vector_store.py, retrieve.py
src/app/                # generate_report.py, dashboard.py
```

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, ChromaDB, sentence-transformers,
Groq API (LLM), Streamlit

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

## Running it

```bash
# One-time: build data, models, knowledge base (see full steps in /docs or ask)
python src\preprocessing\load_data.py
python src\preprocessing\clean_data.py
python src\detection\train_models.py
python knowledge_base\build_kb.py
python src\rag\build_vector_store.py

# Every time: launch the dashboard
python -m streamlit run src\app\dashboard.py
```

## Known limitations

- Dataset is from 2017; real-world attack traffic patterns evolve, so a
  production system would need continuous retraining on current traffic.
- Web Attack - XSS and Brute Force are confused at the flow-statistics
  level (see finding above) — would need payload/HTTP-layer inspection to
  resolve.
- "Rare Attack" grouping (Infiltration/SQLi/Heartbleed) sacrifices
  per-class granularity for reliability, given extremely low sample counts.
- Isolation Forest alone has a high false-positive rate (47% precision);
  it's used as a complementary signal, not a standalone detector.
