
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

CUSTOMER_PATH = os.path.join(
    "data", "raw", "customer_data.csv"
)

OUTPUT_DIR = os.path.join(
    "reports", "risk_prediction"
)

MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD CUSTOMER DATA
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINWISE - AI-BASED FINANCIAL RISK & EARLY WARNING")
print("=" * 65)

df = pd.read_csv(CUSTOMER_PATH)

print("\nCustomer dataset shape:", df.shape)

# --------------------------------------------------
# 3. IDENTIFY AVAILABLE RISK INDICATORS
# --------------------------------------------------

column_options = {
    "credit_utilization": [
        "credit_utilization_ratio"
    ],
    "failed_transactions": [
        "failed_transactions",
        "failed_transaction_count",
        "failed_transactions_count"
    ],
    "satisfaction": [
        "satisfaction_score"
    ],
    "support_tickets": [
        "support_tickets_count"
    ],
    "ticket_resolution": [
        "resolved_tickets_ratio"
    ]
}

available_columns = {}

for indicator, possible_names in column_options.items():
    for name in possible_names:
        if name in df.columns:
            available_columns[indicator] = name
            break

print("\nAvailable risk indicators:")

for indicator, column in available_columns.items():
    print(f"- {indicator}: {column}")

if len(available_columns) < 2:
    raise ValueError(
        "Not enough risk indicators found in customer data."
    )

# --------------------------------------------------
# 4. PREPARE RISK INDICATOR DATA
# --------------------------------------------------

risk_data = pd.DataFrame(index=df.index)

for indicator, column in available_columns.items():
    risk_data[indicator] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# --------------------------------------------------
# 5. CREATE EXPLAINABLE EARLY WARNING FLAGS
# --------------------------------------------------

risk_flags = pd.DataFrame(index=df.index)

if "credit_utilization" in risk_data.columns:
    risk_flags["high_credit_utilization"] = (
        risk_data["credit_utilization"] >= 0.70
    ).astype(int)

if "failed_transactions" in risk_data.columns:
    risk_flags["failed_transaction_flag"] = (
        risk_data["failed_transactions"] > 0
    ).astype(int)

if "satisfaction" in risk_data.columns:
    risk_flags["low_satisfaction_flag"] = (
        risk_data["satisfaction"] <= 3
    ).astype(int)

if "support_tickets" in risk_data.columns:
    risk_flags["high_support_tickets"] = (
        risk_data["support_tickets"] >= 3
    ).astype(int)

if "ticket_resolution" in risk_data.columns:
    risk_flags["low_ticket_resolution"] = (
        risk_data["ticket_resolution"] < 0.50
    ).astype(int)

# --------------------------------------------------
# 6. CALCULATE TRANSPARENT RISK INDICATOR SCORE
# --------------------------------------------------

weights = {
    "high_credit_utilization": 30,
    "failed_transaction_flag": 25,
    "low_satisfaction_flag": 20,
    "high_support_tickets": 15,
    "low_ticket_resolution": 10
}

available_weights = {
    flag: weights[flag]
    for flag in risk_flags.columns
}

total_available_weight = sum(
    available_weights.values()
)

if total_available_weight == 0:
    raise ValueError(
        "No usable early-warning indicators found."
    )

risk_score = sum(
    risk_flags[flag] * weight
    for flag, weight in available_weights.items()
)

risk_score = (
    risk_score / total_available_weight
) * 100

df["early_warning_score"] = risk_score.round(2)

# --------------------------------------------------
# 7. ASSIGN EARLY WARNING LEVELS
# --------------------------------------------------

def assign_risk_level(score):
    if score >= 60:
        return "High"
    elif score >= 30:
        return "Moderate"
    else:
        return "Low"

df["early_warning_level"] = (
    df["early_warning_score"].apply(assign_risk_level)
)

# --------------------------------------------------
# 8. IDENTIFY REASONS FOR EARLY WARNINGS
# --------------------------------------------------

def identify_reasons(row):
    reasons = []

    if (
        "high_credit_utilization" in risk_flags.columns
        and row.get("high_credit_utilization", 0) == 1
    ):
        reasons.append("High credit utilisation")

    if (
        "failed_transaction_flag" in risk_flags.columns
        and row.get("failed_transaction_flag", 0) == 1
    ):
        reasons.append("Failed transactions recorded")

    if (
        "low_satisfaction_flag" in risk_flags.columns
        and row.get("low_satisfaction_flag", 0) == 1
    ):
        reasons.append("Low customer satisfaction")

    if (
        "high_support_tickets" in risk_flags.columns
        and row.get("high_support_tickets", 0) == 1
    ):
        reasons.append("Multiple support tickets")

    if (
        "low_ticket_resolution" in risk_flags.columns
        and row.get("low_ticket_resolution", 0) == 1
    ):
        reasons.append("Low ticket resolution ratio")

    if not reasons:
        return "No selected warning indicators"

    return "; ".join(reasons)


df["early_warning_reasons"] = risk_flags.apply(
    identify_reasons,
    axis=1
)

# --------------------------------------------------
# 9. MACHINE LEARNING - ISOLATION FOREST
# --------------------------------------------------

print("\nTraining Isolation Forest model...")

ml_features = risk_data.copy()

# Add missing-value indicators for useful information.
for column in ml_features.columns:
    ml_features[column + "_missing"] = (
        ml_features[column].isna().astype(int)
    )

# Remove features with no variation.
ml_features = ml_features.loc[
    :, ml_features.nunique(dropna=True) > 1
]

if ml_features.shape[1] == 0:
    raise ValueError(
        "No suitable features available for anomaly detection."
    )

imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(
    ml_features
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_imputed
)

isolation_model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

anomaly_predictions = isolation_model.fit_predict(
    X_scaled
)

df["anomaly_status"] = np.where(
    anomaly_predictions == -1,
    "Unusual Pattern",
    "Normal Pattern"
)

df["anomaly_score"] = (
    -isolation_model.decision_function(X_scaled)
).round(4)

print("Isolation Forest training completed!")

# --------------------------------------------------
# 10. EARLY WARNING DISTRIBUTION
# --------------------------------------------------

print("\n" + "-" * 65)
print("EARLY WARNING LEVEL DISTRIBUTION")
print("-" * 65)

warning_distribution = (
    df["early_warning_level"]
    .value_counts()
    .reindex(["High", "Moderate", "Low"], fill_value=0)
    .rename_axis("warning_level")
    .reset_index(name="customer_count")
)

warning_distribution["percentage"] = (
    warning_distribution["customer_count"]
    / len(df) * 100
).round(2)

print(warning_distribution.to_string(index=False))

warning_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "early_warning_distribution.csv"
    ),
    index=False
)

# --------------------------------------------------
# 11. ANOMALY DETECTION SUMMARY
# --------------------------------------------------

print("\n" + "-" * 65)
print("ANOMALY DETECTION SUMMARY")
print("-" * 65)

anomaly_summary = (
    df["anomaly_status"]
    .value_counts()
    .rename_axis("anomaly_status")
    .reset_index(name="customer_count")
)

anomaly_summary["percentage"] = (
    anomaly_summary["customer_count"]
    / len(df) * 100
).round(2)

print(anomaly_summary.to_string(index=False))

anomaly_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "anomaly_detection_summary.csv"
    ),
    index=False
)

# --------------------------------------------------
# 12. RISK INDICATOR SUMMARY
# --------------------------------------------------

print("\n" + "-" * 65)
print("INDIVIDUAL RISK INDICATOR SUMMARY")
print("-" * 65)

indicator_summary = pd.DataFrame({
    "indicator": risk_flags.columns,
    "customers_flagged": [
        risk_flags[column].sum()
        for column in risk_flags.columns
    ]
})

indicator_summary["percentage"] = (
    indicator_summary["customers_flagged"]
    / len(df) * 100
).round(2)

print(indicator_summary.to_string(index=False))

indicator_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "risk_indicator_summary.csv"
    ),
    index=False
)

# --------------------------------------------------
# 13. SAVE CUSTOMER EARLY WARNING REPORT
# --------------------------------------------------

output_columns = ["customer_id"]

output_columns += [
    column for column in available_columns.values()
    if column in df.columns
]

output_columns += [
    "early_warning_score",
    "early_warning_level",
    "early_warning_reasons",
    "anomaly_status",
    "anomaly_score"
]

customer_report = df[
    list(dict.fromkeys(output_columns))
].copy()

customer_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "customer_early_warning_report.csv"
    ),
    index=False
)

# --------------------------------------------------
# 14. EARLY WARNING DISTRIBUTION CHART
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    warning_distribution["warning_level"],
    warning_distribution["customer_count"]
)

plt.title("Customer Early Warning Level Distribution")
plt.xlabel("Early Warning Level")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "early_warning_distribution.png"
    ),
    dpi=300
)

plt.close()

# --------------------------------------------------
# 15. RISK INDICATOR CHART
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.barh(
    indicator_summary["indicator"],
    indicator_summary["customers_flagged"]
)

plt.title("Customer Early Warning Indicators")
plt.xlabel("Number of Customers")
plt.ylabel("Risk Indicator")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "risk_indicator_analysis.png"
    ),
    dpi=300
)

plt.close()

# --------------------------------------------------
# 16. ANOMALY DISTRIBUTION CHART
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    anomaly_summary["anomaly_status"],
    anomaly_summary["customer_count"]
)

plt.title("Isolation Forest - Customer Pattern Analysis")
plt.xlabel("Pattern Classification")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "anomaly_distribution.png"
    ),
    dpi=300
)

plt.close()

# --------------------------------------------------
# 17. SAVE MACHINE LEARNING MODEL
# --------------------------------------------------

joblib.dump(
    {
        "model": isolation_model,
        "imputer": imputer,
        "scaler": scaler,
        "features": list(ml_features.columns),
        "risk_indicator_weights": available_weights
    },
    os.path.join(
        MODEL_DIR,
        "finwise_isolation_forest.pkl"
    )
)

# --------------------------------------------------
# 18. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINWISE RISK & EARLY WARNING ANALYSIS COMPLETED!")
print("=" * 65)

print("\nTotal customers analysed:", len(df))

print(
    "Customers with high early-warning score:",
    (df["early_warning_level"] == "High").sum()
)

print(
    "Customers with moderate early-warning score:",
    (df["early_warning_level"] == "Moderate").sum()
)

print(
    "Customers with unusual patterns:",
    (df["anomaly_status"] == "Unusual Pattern").sum()
)

print("\nGenerated reports:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

print("\nSaved model:")
print("models/finwise_isolation_forest.pkl")

print("\nOutput folder:", OUTPUT_DIR)

print(
    "\nNote: These are behavioural early-warning indicators, "
    "not verified financial distress or default predictions."
)
