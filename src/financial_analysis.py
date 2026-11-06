
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CUSTOMER_PATH = BASE_DIR / "data" / "raw" / "customer_data.csv"
TRANSACTION_PATH = BASE_DIR / "data" / "raw" / "transactions_data.csv"

REPORT_DIR = BASE_DIR / "reports" / "financial_analysis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

print("=" * 60)
print("FINWISE - FINANCIAL BEHAVIOUR ANALYSIS")
print("=" * 60)

customers = pd.read_csv(CUSTOMER_PATH)
transactions = pd.read_csv(TRANSACTION_PATH)

transactions["date"] = pd.to_datetime(
    transactions["date"], errors="coerce"
)

print("\nDatasets loaded successfully!")


# --------------------------------------------------
# 3. TRANSACTION SUMMARY
# --------------------------------------------------

print("\n1. TRANSACTION SUMMARY")
print("-" * 40)

print("Total transactions:", len(transactions))
print("Total transaction amount:", transactions["amount"].sum())
print("Average transaction amount:", transactions["amount"].mean())
print("Median transaction amount:", transactions["amount"].median())

print("\nTransaction type summary:")

transaction_summary = transactions.groupby("type").agg(
    transaction_count=("amount", "count"),
    total_amount=("amount", "sum"),
    average_amount=("amount", "mean")
).sort_values("total_amount", ascending=False)

print(transaction_summary)

transaction_summary.to_csv(
    REPORT_DIR / "transaction_summary.csv"
)


# --------------------------------------------------
# 4. CUSTOMER FINANCIAL ACTIVITY
# --------------------------------------------------

print("\n2. CUSTOMER FINANCIAL ACTIVITY")
print("-" * 40)

customer_activity = transactions.groupby("customer_id").agg(
    transaction_count=("amount", "count"),
    total_transaction_amount=("amount", "sum"),
    average_transaction_amount=("amount", "mean"),
    largest_transaction=("amount", "max"),
    smallest_transaction=("amount", "min")
).reset_index()

print(customer_activity.head())

customer_activity.to_csv(
    REPORT_DIR / "customer_financial_activity.csv",
    index=False
)


# --------------------------------------------------
# 5. MERGE CUSTOMER AND TRANSACTION DATA
# --------------------------------------------------

print("\n3. MERGING CUSTOMER AND TRANSACTION DATA")
print("-" * 40)

financial_data = customers.merge(
    customer_activity,
    on="customer_id",
    how="left"
)

financial_data["transaction_count"] = (
    financial_data["transaction_count"].fillna(0)
)

financial_data["total_transaction_amount"] = (
    financial_data["total_transaction_amount"].fillna(0)
)

print("Merged dataset shape:", financial_data.shape)

financial_data.to_csv(
    REPORT_DIR / "finwise_customer_analysis.csv",
    index=False
)


# --------------------------------------------------
# 6. SAVINGS AND BANKING PRODUCT ANALYSIS
# --------------------------------------------------

print("\n4. SAVINGS AND BANKING PRODUCT ANALYSIS")
print("-" * 40)

product_columns = [
    "savings_account",
    "credit_card",
    "personal_loan",
    "investment_account",
    "insurance_product"
]

product_analysis = customers[product_columns].sum().reset_index()

product_analysis.columns = [
    "banking_product",
    "customer_count"
]

print(product_analysis)

product_analysis.to_csv(
    REPORT_DIR / "banking_product_analysis.csv",
    index=False
)


# --------------------------------------------------
# 7. FINANCIAL BEHAVIOUR INDICATORS
# --------------------------------------------------

print("\n5. FINANCIAL BEHAVIOUR INDICATORS")
print("-" * 40)

print(
    "Customers with auto savings enabled:",
    customers["auto_savings_enabled"].sum()
)

print(
    "Customers with bill payment usage:",
    customers["bill_payment_user"].sum()
)

print(
    "Customers with investment accounts:",
    customers["investment_account"].sum()
)

print(
    "Customers with failed transactions:",
    (customers["failed_transactions"] > 0).sum()
)

print(
    "Customers with credit utilization data:",
    customers["credit_utilization_ratio"].notna().sum()
)


# --------------------------------------------------
# 8. MONTHLY TRANSACTION ANALYSIS
# --------------------------------------------------

print("\n6. MONTHLY TRANSACTION ANALYSIS")
print("-" * 40)

monthly_transactions = (
    transactions.dropna(subset=["date"])
    .assign(month=lambda df: df["date"].dt.to_period("M").astype(str))
    .groupby("month")
    .agg(
        transaction_count=("amount", "count"),
        total_amount=("amount", "sum")
    )
    .reset_index()
)

print(monthly_transactions.head(12))

monthly_transactions.to_csv(
    REPORT_DIR / "monthly_transaction_analysis.csv",
    index=False
)


# --------------------------------------------------
# 9. VISUALIZATIONS
# --------------------------------------------------

print("\n7. GENERATING FINANCIAL ANALYSIS CHARTS")
print("-" * 40)

# Chart 1: Transaction type by amount

plt.figure(figsize=(9, 5))

transaction_summary["total_amount"].plot(
    kind="bar", edgecolor="black"
)

plt.title("Total Transaction Amount by Type")
plt.xlabel("Transaction Type")
plt.ylabel("Total Amount")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(REPORT_DIR / "transaction_amount_by_type.png")
plt.close()


# Chart 2: Monthly transaction amount

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_transactions["month"],
    monthly_transactions["total_amount"],
    marker="o"
)

plt.title("Monthly Transaction Amount Trend")
plt.xlabel("Month")
plt.ylabel("Total Transaction Amount")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(REPORT_DIR / "monthly_transaction_trend.png")
plt.close()


# Chart 3: Banking product usage

plt.figure(figsize=(9, 5))

product_analysis.set_index("banking_product")[
    "customer_count"
].plot(kind="bar", edgecolor="black")

plt.title("Customer Banking Product Usage")
plt.xlabel("Banking Product")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(REPORT_DIR / "financial_product_usage.png")
plt.close()


# Chart 4: Customer transaction activity

plt.figure(figsize=(9, 5))

plt.hist(
    customer_activity["transaction_count"],
    bins=40,
    edgecolor="black"
)

plt.title("Customer Transaction Activity Distribution")
plt.xlabel("Number of Transactions")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(REPORT_DIR / "customer_transaction_activity.png")
plt.close()


# --------------------------------------------------
# 10. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINANCIAL BEHAVIOUR ANALYSIS COMPLETED!")
print("=" * 60)

print("\nGenerated reports saved in:")
print(REPORT_DIR)

print("\nGenerated files:")
for file in sorted(REPORT_DIR.iterdir()):
    print("-", file.name)
