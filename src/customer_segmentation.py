
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

CUSTOMER_PATH = os.path.join("data", "raw", "customer_data.csv")
TRANSACTION_PATH = os.path.join("data", "raw", "transactions_data.csv")

OUTPUT_DIR = os.path.join("reports", "customer_segmentation")
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINWISE - CUSTOMER SEGMENTATION")
print("=" * 60)

customers = pd.read_csv(CUSTOMER_PATH)
transactions = pd.read_csv(TRANSACTION_PATH)

transactions["date"] = pd.to_datetime(
    transactions["date"], errors="coerce"
)

print("\nCustomer dataset shape:", customers.shape)
print("Transaction dataset shape:", transactions.shape)

# --------------------------------------------------
# 3. CUSTOMER TRANSACTION FEATURES
# --------------------------------------------------

transactions = transactions.dropna(
    subset=["customer_id", "amount"]
)

transaction_features = transactions.groupby(
    "customer_id"
).agg(
    transaction_count=("amount", "count"),
    total_transaction_amount=("amount", "sum"),
    average_transaction_amount=("amount", "mean"),
    median_transaction_amount=("amount", "median")
).reset_index()

# --------------------------------------------------
# 4. MERGE CUSTOMER AND TRANSACTION DATA
# --------------------------------------------------

data = customers.merge(
    transaction_features,
    on="customer_id",
    how="left"
)

transaction_columns = [
    "transaction_count",
    "total_transaction_amount",
    "average_transaction_amount",
    "median_transaction_amount"
]

data[transaction_columns] = data[
    transaction_columns
].fillna(0)

# FIX: Use the actual customer-level failed transaction
# column instead of searching transaction type for "fail".

if "failed_transactions" in data.columns:
    data["failed_transaction_count"] = pd.to_numeric(
        data["failed_transactions"],
        errors="coerce"
    ).fillna(0)

else:
    print(
        "\nWARNING: 'failed_transactions' column "
        "not found in customer dataset."
    )
    data["failed_transaction_count"] = 0

print(
    "\nTotal failed transactions recorded:",
    int(data["failed_transaction_count"].sum())
)

print(
    "Customers with failed transactions:",
    int((data["failed_transaction_count"] > 0).sum())
)

# --------------------------------------------------
# 5. SELECT CUSTOMER BEHAVIOUR FEATURES
# --------------------------------------------------

candidate_features = [
    "transaction_count",
    "total_transaction_amount",
    "average_transaction_amount",
    "median_transaction_amount",
    "failed_transaction_count",
    "credit_utilization_ratio",
    "satisfaction_score",
    "support_tickets_count",
    "savings_account",
    "auto_savings_enabled",
    "credit_card",
    "personal_loan",
    "investment_account",
    "insurance_product",
    "bill_payment_user"
]

available_features = [
    column for column in candidate_features
    if column in data.columns
]

if len(available_features) < 2:
    raise ValueError(
        "Not enough suitable customer features found."
    )

print("\nFeatures selected for segmentation:")
for feature in available_features:
    print("-", feature)

# --------------------------------------------------
# 6. PREPARE FEATURES
# --------------------------------------------------

X = data[available_features].copy()

for column in X.columns:
    X[column] = pd.to_numeric(
        X[column], errors="coerce"
    )

# Preserve missing credit-utilisation information.
if "credit_utilization_ratio" in X.columns:
    X["credit_utilization_missing"] = (
        X["credit_utilization_ratio"].isna().astype(int)
    )

# Log transformation reduces the effect of highly
# skewed transaction amounts and counts.

skewed_features = [
    "transaction_count",
    "total_transaction_amount",
    "average_transaction_amount",
    "median_transaction_amount",
    "failed_transaction_count",
    "support_tickets_count"
]

for column in skewed_features:
    if column in X.columns:
        X[column] = np.log1p(
            X[column].clip(lower=0)
        )

# Remove features that contain no useful variation.
X = X.loc[:, X.nunique(dropna=True) > 1]

if X.empty:
    raise ValueError(
        "No features with sufficient variation found."
    )

# Handle missing values.
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Scale features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("\nPrepared feature matrix:", X_scaled.shape)

# --------------------------------------------------
# 7. FIND THE BEST NUMBER OF CLUSTERS
# --------------------------------------------------

print("\nEvaluating different cluster counts...")

cluster_scores = []
cluster_range = range(2, 7)

sample_size = min(5000, len(X_scaled))
rng = np.random.RandomState(42)

sample_indices = rng.choice(
    len(X_scaled),
    size=sample_size,
    replace=False
)

X_sample = X_scaled[sample_indices]

for k in cluster_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    sample_labels = model.fit_predict(X_sample)

    score = silhouette_score(
        X_sample,
        sample_labels
    )

    cluster_scores.append(score)

    print(
        f"Clusters: {k} | "
        f"Silhouette Score: {score:.4f}"
    )

best_k = list(cluster_range)[
    int(np.argmax(cluster_scores))
]

print("\nSelected number of clusters:", best_k)

# Save cluster evaluation.
evaluation = pd.DataFrame({
    "number_of_clusters": list(cluster_range),
    "silhouette_score": cluster_scores
})

evaluation.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_evaluation.csv"),
    index=False
)

# --------------------------------------------------
# 8. TRAIN FINAL K-MEANS MODEL
# --------------------------------------------------

print("\nTraining final K-Means model...")

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

data["customer_cluster"] = kmeans.fit_predict(
    X_scaled
)

print("Customer segmentation completed!")

# --------------------------------------------------
# 9. CUSTOMER CLUSTER DISTRIBUTION
# --------------------------------------------------

cluster_distribution = (
    data["customer_cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_distribution.columns = [
    "customer_cluster",
    "customer_count"
]

cluster_distribution["percentage"] = (
    cluster_distribution["customer_count"]
    / len(data) * 100
)

print("\nCUSTOMER CLUSTER DISTRIBUTION")
print(cluster_distribution.to_string(index=False))

cluster_distribution.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_distribution.csv"),
    index=False
)

# --------------------------------------------------
# 10. CLUSTER PROFILE ANALYSIS
# --------------------------------------------------

profile_features = [
    column for column in available_features
    if column in data.columns
]

cluster_profiles = data.groupby(
    "customer_cluster"
)[profile_features].mean().round(2)

cluster_profiles.insert(
    0,
    "customer_count",
    data.groupby("customer_cluster").size()
)

print("\nCUSTOMER CLUSTER PROFILES")
print(cluster_profiles.to_string())

cluster_profiles.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_profiles.csv")
)

# --------------------------------------------------
# 11. SAVE SEGMENTED CUSTOMER DATA
# --------------------------------------------------

data.to_csv(
    os.path.join(OUTPUT_DIR, "customer_segments.csv"),
    index=False
)

# --------------------------------------------------
# 12. CLUSTER DISTRIBUTION CHART
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    cluster_distribution["customer_cluster"].astype(str),
    cluster_distribution["customer_count"]
)

plt.title("FinWise Customer Segment Distribution")
plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "cluster_distribution.png"),
    dpi=300
)

plt.close()

# --------------------------------------------------
# 13. SILHOUETTE SCORE CHART
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    list(cluster_range),
    cluster_scores,
    marker="o"
)

plt.title("K-Means Cluster Evaluation")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.xticks(list(cluster_range))
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "cluster_evaluation.png"),
    dpi=300
)

plt.close()

# --------------------------------------------------
# 14. SAVE TRAINED MODEL AND PREPROCESSORS
# --------------------------------------------------

joblib.dump(
    {
        "model": kmeans,
        "scaler": scaler,
        "imputer": imputer,
        "features": list(X.columns),
        "best_k": best_k
    },
    os.path.join(MODEL_DIR, "finwise_kmeans_model.pkl")
)

# --------------------------------------------------
# 15. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("CUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nTotal customers segmented:", len(data))
print("Number of clusters:", best_k)

print("\nGenerated files:")
print("- customer_segments.csv")
print("- cluster_profiles.csv")
print("- cluster_distribution.csv")
print("- cluster_evaluation.csv")
print("- cluster_distribution.png")
print("- cluster_evaluation.png")
print("- finwise_kmeans_model.pkl")

print("\nOutput folder:", OUTPUT_DIR)
print("Model folder:", MODEL_DIR)
