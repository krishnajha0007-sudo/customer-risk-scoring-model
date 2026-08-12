# Customer AML/KYC Risk Scoring Model

A machine learning project that predicts customer money-laundering risk
from behavioral and profile data — built directly on real KYC/AML risk
logic (country risk, PEP status, cash transaction ratios, cross-border
activity) rather than a generic tutorial dataset.

> **Note:** Uses a synthetically generated dataset of 3,000 customers
> (see `src/generate_risk_data.py`) with country names anonymized
> (Country_A, Country_B, etc.) — not real customer or institutional data.

## What this project does

- Generates a realistic customer dataset with the risk factors used in
  real-world AML/KYC risk rating: country risk category, PEP status,
  cash transaction ratio, cross-border transaction ratio, transaction
  volume/value, KYC document completeness, adverse media flags
- Trains and compares two classification models — Logistic Regression
  (interpretable baseline) and Random Forest (feature importance) — to
  predict which customers should be flagged as high-risk
- Handles class imbalance properly (only ~5% of customers are high-risk,
  matching real-world AML population distributions) using class weighting
  and evaluates with precision/recall/ROC-AUC rather than raw accuracy,
  which would be misleading on an imbalanced dataset
- Surfaces feature importance to show *why* the model flags a customer —
  important for AML explainability, not just prediction

## Tech used

- **Python** — pandas, numpy
- **scikit-learn** — Logistic Regression, Random Forest, StandardScaler,
  train/test split with stratification, classification metrics, ROC-AUC
- **Matplotlib / Seaborn** — confusion matrix, ROC curve comparison,
  feature importance visualization

## Project structure

kyc-risk-scoring-model/
├── data/
│   └── customer_risk_data.csv        # generated dataset (3,000 customers)
├── src/
│   ├── generate_risk_data.py         # synthetic data generator
│   └── train_risk_model.py           # trains + evaluates both models
├── output/
│   ├── model_evaluation_report.txt
│   ├── 01_confusion_matrix.png
│   ├── 02_roc_curve_comparison.png
│   └── 03_feature_importance.png
└── README.md

## How to run

pip install pandas numpy scikit-learn matplotlib seaborn
python src/generate_risk_data.py     # creates data/customer_risk_data.csv
python src/train_risk_model.py       # trains models, saves output/

## Results

| Model | ROC-AUC | High-Risk Recall | High-Risk Precision |
|---|---|---|---|
| Logistic Regression | 0.963 | 0.86 | 0.33 |
| Random Forest | 0.951 | 0.43 | 0.57 |

**Why Logistic Regression's lower precision is actually the right trade-off
here:** in AML risk scoring, missing a genuinely high-risk customer (a
false negative) is far more costly than flagging a low-risk customer for
extra review (a false positive) — regulators penalize misses, not
over-caution. Logistic Regression catches 86% of true high-risk customers
vs. Random Forest's 43%, even though it also raises more false alarms.
This mirrors a real modeling decision AML teams have to make between
precision and recall depending on regulatory risk appetite.

**Top risk drivers identified by the model:** country risk category, PEP
status, cash transaction ratio, and cross-border transaction ratio — all
consistent with standard AML red-flag indicators.

## Why I built this

This project translates ~2 years of hands-on KYC/AML risk assessment
experience into a supervised machine learning problem — the same red
flags (PEP status, high-risk geographies, cash-heavy transaction patterns)
that I evaluated manually are here framed as model features, with the
model learning to replicate and generalize that risk logic.

## Limitations & next steps

- Ground-truth labels are rule-based + noise (synthetic), not real
  investigated SAR (Suspicious Activity Report) outcomes — a production
  model would need labeled historical case data
- Country risk is simplified to 3 categories; production would use
  live FATF/OFAC list integration
- Next: add SHAP values for per-customer explainability (important for
  regulatory audit trails), and test on real anonymized/public AML
  datasets (e.g. IBM's synthetic AML dataset on Kaggle)
