
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

CUSTOMER_PATH = BASE_DIR / "data" / "raw" / "customer_data.csv"
TRANSACTION_PATH = BASE_DIR / "data" / "raw" / "transactions_data.csv"

REPORT_DIR = BASE_DIR / "reports" / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
print("=" * 60)
print("FINWISE - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

customers = pd.read_csv(CUSTOMER_PATH)
transactions = pd.read_csv(TRANSACTION_PATH)

# 1. Dataset overview
print("\n1. DATASET OVERVIEW")
print("-" * 40)
print("Customer dataset shape:", customers.shape)
print("Transaction dataset shape:", transactions.shape)

# 2. Customer demographics
print("\n2. CUSTOMER DEMOGRAPHICS")
print("-" * 40)

for column in ["age", "gender", "income_bracket", "occupation", "location"]:
    if column in customers.columns:
        print(f"\n{column.upper()} DISTRIBUTION")
        print(customers[column].value_counts().head(10))

# 3. Banking product usage
print("\n3. BANKING PRODUCT USAGE")
print("-" * 40)

product_columns = [
    "savings_account",
    "credit_card",
    "personal_loan",
    "investment_account",
    "insurance_product"
]

for column in product_columns:
    if column in customers.columns:
        print(f"{column}: {customers[column].sum()} customers")

# 4. Transaction analysis
print("\n4. TRANSACTION ANALYSIS")
print("-" * 40)

print("\nTransaction types:")
print(transactions["type"].value_counts())

print("\nTransaction amount statistics:")
print(transactions["amount"].describe())

# 5. Customer satisfaction
print("\n5. CUSTOMER SATISFACTION")
print("-" * 40)

if "satisfaction_score" in customers.columns:
    print(customers["satisfaction_score"].describe())

if "feedback_sentiment" in customers.columns:
    print(customers["feedback_sentiment"].value_counts())

# 6. Create visualizations
print("\n6. CREATING VISUALIZATIONS")
print("-" * 40)

# Customer segment distribution
if "customer_segment" in customers.columns:
    plt.figure(figsize=(9, 5))
    customers["customer_segment"].value_counts().plot(
        kind="bar", edgecolor="black"
    )
    plt.title("Customer Distribution by Segment")
    plt.xlabel("Customer Segment")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "customer_segments.png")
    plt.close()

# Banking product usage
product_counts = customers[product_columns].sum()

plt.figure(figsize=(9, 5))
product_counts.plot(kind="bar", edgecolor="black")
plt.title("Banking Product Usage")
plt.xlabel("Banking Products")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(REPORT_DIR / "banking_product_usage.png")
plt.close()

# Transaction type distribution
plt.figure(figsize=(9, 5))
transactions["type"].value_counts().plot(
    kind="bar", edgecolor="black"
)
plt.title("Transaction Type Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(REPORT_DIR / "transaction_types.png")
plt.close()

# Transaction amount distribution
plt.figure(figsize=(9, 5))
plt.hist(transactions["amount"], bins=50, edgecolor="black")
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(REPORT_DIR / "transaction_amounts.png")
plt.close()

print("\nEDA completed successfully!")
print("Visualizations saved in:", REPORT_DIR)
