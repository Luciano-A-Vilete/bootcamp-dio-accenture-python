# Credit Card Fraud Detection — Anomaly Detection Benchmark

> Unsupervised anomaly detection on a highly imbalanced credit card transactions dataset, comparing Isolation Forest, Local Outlier Factor and One-Class SVM.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

| Property | Value |
|---|---|
| **Goal** | Detect fraudulent credit card transactions without using labels at training time |
| **Approach** | Unsupervised anomaly detection (novelty detection) |
| **Models** | Isolation Forest, Local Outlier Factor, One-Class SVM (optional) |
| **Dataset** | Credit Card Fraud Detection — 284,807 transactions, 492 frauds (0.172%) |
| **Source** | OpenML id=1597 (mirror of the original Kaggle dataset) |
| **Stack** | Python 3.9+, scikit-learn, pandas, matplotlib, seaborn |

## Why Anomaly Detection Instead of a Classifier?

Credit card fraud is a textbook **extreme class imbalance** problem (~0.17% positive rate). In real production environments, fraud labels are also **delayed, incomplete and noisy** — disputes can take weeks to be confirmed. Unsupervised anomaly detection offers two practical advantages:

1. **No labels needed at training time** — useful when fraud tags are sparse or arrive late.
2. **Detects novel attack patterns** that a supervised model trained on past fraud would miss.

This project benchmarks three classic anomaly detection algorithms and reports honest metrics for imbalanced data (PR-AUC, not just ROC-AUC).

## Dataset

The dataset contains transactions made by European cardholders in September 2013.

- **284,807 transactions**, of which **492 are fraudulent** (0.172%)
- **30 features**: `Time`, `Amount`, and `V1`–`V28` (PCA-transformed for anonymization)
- **Target**: `Class` (0 = legitimate, 1 = fraud) — used **only for evaluation**

The dataset is downloaded automatically from OpenML on first run and cached under `~/scikit_learn_data/`. No Kaggle authentication required.

> Original source: Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced Classification", IEEE SSCI 2015.

## Methodology

### Pipeline

```
Load (OpenML)
   ↓
EDA (Time, Amount distributions)
   ↓
Preprocess (RobustScaler on Time and Amount; V1–V28 already PCA-scaled)
   ↓
Train/test split (70/30, stratified)
   ↓
Train 3 unsupervised models in parallel
   ↓
Evaluate (Classification report, Confusion matrix, ROC-AUC, PR-AUC)
   ↓
Plot Precision-Recall curves
```

### Models

| Model | Approach | Strengths | Trade-offs |
|---|---|---|---|
| **Isolation Forest** | Random partitioning; anomalies require fewer splits to isolate | Fast, scalable, low memory | Less precise on dense local clusters |
| **Local Outlier Factor** | Local density deviation against k-nearest neighbors | Captures local context | O(n²) — does not scale to millions of rows |
| **One-Class SVM** | Learns a boundary around normal data in kernel space | Strong theoretical foundation | Very slow; sensitive to `nu` and `gamma` |

LOF and One-Class SVM are trained on a 50,000-row subsample (configurable) to keep runtime reasonable.

### Why PR-AUC and not just ROC-AUC?

With 0.17% positives, ROC-AUC is misleading — a model that ranks most negatives correctly looks excellent regardless of fraud detection performance. **Average Precision (PR-AUC)** focuses on the positive class and is the more honest metric here.

## Project Structure

```
.
├── anomaly_detection_creditcard.py   # Main script
├── requirements.txt                  # Dependencies
├── README.md                         # This file
└── outputs/                          # Generated on first run
    ├── eda_distribuicoes.png
    ├── precision_recall_curves.png
    └── resumo_modelos.csv
```

## Getting Started

### Requirements

- Python 3.9 or higher
- ~500 MB free disk space (dataset cache + outputs)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/credit-card-anomaly-detection.git
cd credit-card-anomaly-detection

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:

```
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
```

### Run

```bash
python anomaly_detection_creditcard.py
```

First execution downloads the dataset (~150 MB) and takes ~2–5 minutes on a modern laptop. Subsequent runs use the cache.

### Configuration

Edit constants at the top of the script:

| Parameter | Default | Description |
|---|---|---|
| `RANDOM_STATE` | `42` | Seed for reproducibility |
| `USE_OCSVM` | `False` | Set to `True` to include One-Class SVM (adds ~10–20 min) |
| `SAMPLE_FOR_LOF_OCSVM` | `50_000` | Training subsample size for O(n²) algorithms |

## Expected Results

Indicative metrics on the held-out test set (30% stratified split):

| Model | ROC-AUC | Avg Precision (PR-AUC) | Notes |
|---|---|---|---|
| Isolation Forest | ~0.95 | ~0.18 | Best speed/quality trade-off |
| Local Outlier Factor | ~0.78 | ~0.05 | Sensitive to subsampling |
| One-Class SVM | ~0.94 | ~0.13 | Slowest; not always worth it |

> Exact numbers depend on the random seed and the LOF/OCSVM subsample.

For comparison, a supervised XGBoost model with `scale_pos_weight` typically reaches **PR-AUC ≈ 0.85** on this dataset — the gap quantifies the cost of giving up labels.

### Outputs

- `eda_distribuicoes.png` — Histograms of `Time` and `Amount`
- `precision_recall_curves.png` — Precision-Recall curves for all models
- `resumo_modelos.csv` — Summary table with ROC-AUC and Average Precision

## Interpreting the Results

- **High recall, low precision** is the typical regime for unsupervised fraud detection. In production, the score should feed a triage queue for human review, not an auto-block decision.
- The **decision threshold** on the anomaly score is the main lever to trade precision against recall — adjust it according to the cost of false positives (customer friction) versus false negatives (fraud loss).
- Concept drift is real: retrain monthly or whenever the score distribution shifts.

## Limitations

- Features `V1`–`V28` are anonymized via PCA; the model cannot use raw transaction fields (merchant, MCC, geolocation) that drive most production systems.
- The dataset is from 2013 and reflects fraud patterns of that era.
- Subsampling for LOF and OCSVM introduces variance — averaging over multiple runs is recommended for reliable benchmarking.

## Roadmap

- [ ] Add a supervised baseline (XGBoost with `scale_pos_weight`)
- [ ] Add a deep learning baseline (Autoencoder reconstruction error)
- [ ] Threshold tuning notebook with cost-sensitive analysis
- [ ] SHAP explanations for flagged transactions
- [ ] Dockerfile for reproducible runs
- [ ] CI workflow with `pytest` smoke tests

## License

This project is released under the MIT License — see [LICENSE](LICENSE) for details.

The dataset is provided by ULB Machine Learning Group under the [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/).

## Acknowledgements

- **Dataset**: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi. _Calibrating Probability with Undersampling for Unbalanced Classification_. IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.
- **Hosting**: [OpenML](https://www.openml.org/d/1597) and [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Author

**Vilete** — Data Analyst
Recife, Pernambuco, Brazil

[FILL: GitHub profile link] · [FILL: LinkedIn link]
