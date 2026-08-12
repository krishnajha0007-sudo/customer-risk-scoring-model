"""
generate_risk_data.py
-----------------------
Generates a synthetic customer risk dataset modeled on real-world KYC/AML
risk factors: transaction behavior, geography, PEP (Politically Exposed
Person) status, account age, and channel usage. Produces a binary
"high_risk" label based on a realistic (if simplified) rule combination,
which the ML model in train_risk_model.py then learns to predict.

This reflects the kind of customer risk-rating logic used in real KYC/AML
programs, rebuilt here as a supervised learning problem.
"""

import numpy as np
import pandas as pd

np.random.seed(21)

N = 3000

high_risk_countries = ["Country_A", "Country_B", "Country_C"]  # anonymized, stand-in for FATF grey/black list style flags
medium_risk_countries = ["Country_D", "Country_E"]
low_risk_countries = ["Country_F", "Country_G", "Country_H", "Country_I", "Country_J"]

all_countries = high_risk_countries + medium_risk_countries + low_risk_countries
country_weights = [0.03]*3 + [0.07]*2 + [0.152]*5  # skewed toward low-risk countries
country_weights = [w / sum(country_weights) for w in country_weights]  # normalize to exactly 1

rows = []
for i in range(N):
    country = np.random.choice(all_countries, p=country_weights)
    country_risk = (
        "high" if country in high_risk_countries else
        "medium" if country in medium_risk_countries else "low"
    )

    is_pep = np.random.choice([1, 0], p=[0.04, 0.96])
    account_age_months = int(np.random.exponential(24))
    monthly_txn_count = int(np.random.poisson(15))
    avg_txn_amount = round(np.random.lognormal(mean=8.5, sigma=1.2), 2)
    cash_txn_ratio = round(np.random.beta(2, 8), 2)  # most customers low cash ratio
    cross_border_txn_ratio = round(np.random.beta(1.5, 6), 2)
    has_adverse_media = np.random.choice([1, 0], p=[0.02, 0.98])
    kyc_doc_complete = np.random.choice([1, 0], p=[0.92, 0.08])
    channel = np.random.choice(["branch", "online", "mobile"], p=[0.3, 0.35, 0.35])

    # Rule-based ground truth risk score (simplified real-world logic),
    # with noise added so the ML model has a genuine learning problem,
    # not a trivial lookup.
    risk_score = 0
    risk_score += {"high": 3, "medium": 1, "low": 0}[country_risk]
    risk_score += 3 if is_pep else 0
    risk_score += 2 if has_adverse_media else 0
    risk_score += 2 if cash_txn_ratio > 0.4 else 0
    risk_score += 1 if cross_border_txn_ratio > 0.3 else 0
    risk_score += 1 if avg_txn_amount > 100000 else 0
    risk_score += 1 if not kyc_doc_complete else 0
    risk_score += np.random.normal(0, 0.8)  # noise

    high_risk = 1 if risk_score >= 4 else 0

    rows.append({
        "customer_id": f"CUST{i+1:05d}",
        "country": country,
        "country_risk_category": country_risk,
        "is_pep": is_pep,
        "account_age_months": account_age_months,
        "monthly_txn_count": monthly_txn_count,
        "avg_txn_amount": avg_txn_amount,
        "cash_txn_ratio": cash_txn_ratio,
        "cross_border_txn_ratio": cross_border_txn_ratio,
        "has_adverse_media": has_adverse_media,
        "kyc_doc_complete": kyc_doc_complete,
        "channel": channel,
        "high_risk": high_risk,
    })

df = pd.DataFrame(rows)
df.to_csv("data/customer_risk_data.csv", index=False)

print(f"Generated {len(df)} customer records")
print(f"High-risk rate: {df['high_risk'].mean():.1%}")
print(df.head())
