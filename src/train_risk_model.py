"""
train_risk_model.py
---------------------
Trains a supervised classification model to predict customer AML/KYC
risk (high_risk: 1/0) from behavioral and profile features, and
evaluates it properly - including handling class imbalance, since
high-risk customers are a small minority (a realistic AML scenario).

Two models are compared: Logistic Regression (interpretable baseline)
and Random Forest (usually stronger, gives feature importance).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

DATA_PATH = "data/customer_risk_data.csv"
OUT_DIR = "output"

# ---------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# One-hot encode categorical features
df_model = pd.get_dummies(
    df,
    columns=["country_risk_category", "channel"],
    drop_first=True
)
df_model = df_model.drop(columns=["customer_id", "country"])

X = df_model.drop(columns=["high_risk"])
y = df_model["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} customers ({y_train.mean():.1%} high-risk)")
print(f"Test set: {len(X_test)} customers ({y_test.mean():.1%} high-risk)\n")

# ---------------------------------------------------------------
# 2. Logistic Regression baseline
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
logreg_pred = logreg.predict(X_test_scaled)
logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]

print("=== Logistic Regression ===")
print(classification_report(y_test, logreg_pred, zero_division=0))
logreg_auc = roc_auc_score(y_test, logreg_proba)
print(f"ROC-AUC: {logreg_auc:.3f}\n")

# ---------------------------------------------------------------
# 3. Random Forest (main model)
# ---------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("=== Random Forest ===")
print(classification_report(y_test, rf_pred, zero_division=0))
rf_auc = roc_auc_score(y_test, rf_proba)
print(f"ROC-AUC: {rf_auc:.3f}\n")

# Save text report
with open(f"{OUT_DIR}/model_evaluation_report.txt", "w") as f:
    f.write("=== Logistic Regression ===\n")
    f.write(classification_report(y_test, logreg_pred, zero_division=0))
    f.write(f"ROC-AUC: {logreg_auc:.3f}\n\n")
    f.write("=== Random Forest ===\n")
    f.write(classification_report(y_test, rf_pred, zero_division=0))
    f.write(f"ROC-AUC: {rf_auc:.3f}\n")

# ---------------------------------------------------------------
# VISUAL 1 - Confusion matrix (Random Forest)
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"])
plt.title("Confusion Matrix — Random Forest", fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_confusion_matrix.png")
plt.close()

# ---------------------------------------------------------------
# VISUAL 2 - ROC curves comparing both models
# ---------------------------------------------------------------
fpr_lr, tpr_lr, _ = roc_curve(y_test, logreg_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)

plt.figure(figsize=(6.5, 5.5))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={logreg_auc:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Model Comparison", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_roc_curve_comparison.png")
plt.close()

# ---------------------------------------------------------------
# VISUAL 3 - Feature importance (Random Forest)
# ---------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.figure(figsize=(8, 6))
importances.plot(kind="barh", color="#2c6e91")
plt.title("Feature Importance — What Drives Risk Predictions", fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_feature_importance.png")
plt.close()

print("Saved 3 visuals + evaluation report to output/")
print("\nTop 5 most important features:")
print(importances.sort_values(ascending=False).head(5).to_string())
