
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# FINWISE - CUSTOMER FINANCIAL WELLBEING ANALYSIS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports" / "wellbeing_analysis"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

CUSTOMER_FILE = DATA_DIR / "customer_data.csv"


def main():

    print("=" * 60)
    print("FINWISE - CUSTOMER FINANCIAL WELLBEING ANALYSIS")
    print("=" * 60)

    # 1. Load customer dataset
    customers = pd.read_csv(CUSTOMER_FILE)

    print("\nCustomer dataset loaded successfully!")
    print("Total customers:", len(customers))

    # 2. Savings and financial activity indicators
    print("\n1. SAVINGS AND FINANCIAL ACTIVITY")
    print("-" * 45)

    savings_metrics = {
        "Savings Account": customers["savings_account"].sum(),
        "Auto Savings Enabled": customers["auto_savings_enabled"].sum(),
        "Bill Payment Users": customers["bill_payment_user"].sum(),
        "Investment Account Users": customers["investment_account"].sum(),
        "Credit Card Users": customers["credit_card"].sum(),
        "Personal Loan Users": customers["personal_loan"].sum(),
        "Insurance Product Users": customers["insurance_product"].sum()
    }

    savings_df = pd.DataFrame(
        list(savings_metrics.items()),
        columns=["Financial Indicator", "Customer Count"]
    )

    print(savings_df.to_string(index=False))
    savings_df.to_csv(
        REPORT_DIR / "financial_activity_indicators.csv",
        index=False
    )

    # 3. Credit utilization analysis
    print("\n2. CREDIT UTILIZATION ANALYSIS")
    print("-" * 45)

    credit_data = customers["credit_utilization_ratio"].dropna()

    print("Customers with credit utilization data:", len(credit_data))
    print("Customers without credit utilization data:",
          customers["credit_utilization_ratio"].isna().sum())

    if not credit_data.empty:
        print("Average credit utilization:", round(credit_data.mean(), 3))
        print("Median credit utilization:", round(credit_data.median(), 3))
        print("Minimum:", round(credit_data.min(), 3))
        print("Maximum:", round(credit_data.max(), 3))

        credit_summary = pd.DataFrame({
            "Metric": ["Average", "Median", "Minimum", "Maximum"],
            "Value": [
                credit_data.mean(),
                credit_data.median(),
                credit_data.min(),
                credit_data.max()
            ]
        })

        credit_summary.to_csv(
            REPORT_DIR / "credit_utilization_summary.csv",
            index=False
        )

    # 4. Failed transaction analysis
    print("\n3. FAILED TRANSACTION ANALYSIS")
    print("-" * 45)

    failed_customers = (customers["failed_transactions"] > 0).sum()
    total_failed = customers["failed_transactions"].sum()

    print("Customers with failed transactions:", failed_customers)
    print("Total failed transactions:", total_failed)

    failed_df = pd.DataFrame({
        "Indicator": [
            "Customers with Failed Transactions",
            "Total Failed Transactions"
        ],
        "Value": [failed_customers, total_failed]
    })

    failed_df.to_csv(
        REPORT_DIR / "failed_transaction_analysis.csv",
        index=False
    )

    # 5. Customer satisfaction and support analysis
    print("\n4. CUSTOMER SATISFACTION AND SUPPORT")
    print("-" * 45)

    print("Average satisfaction score:",
          round(customers["satisfaction_score"].mean(), 2))

    print("Average NPS score:",
          round(customers["nps_score"].mean(), 2))

    print("Total support tickets:",
          customers["support_tickets_count"].sum())

    print("\nFeedback sentiment:")
    print(customers["feedback_sentiment"].value_counts())

    satisfaction_summary = pd.DataFrame({
        "Metric": [
            "Average Satisfaction Score",
            "Average NPS Score",
            "Total Support Tickets",
            "Average Resolved Tickets Ratio"
        ],
        "Value": [
            customers["satisfaction_score"].mean(),
            customers["nps_score"].mean(),
            customers["support_tickets_count"].sum(),
            customers["resolved_tickets_ratio"].mean()
        ]
    })

    satisfaction_summary.to_csv(
        REPORT_DIR / "customer_satisfaction_summary.csv",
        index=False
    )

    # 6. Income group analysis
    print("\n5. FINANCIAL ACTIVITY BY INCOME GROUP")
    print("-" * 45)

    income_analysis = customers.groupby("income_bracket").agg(
        customer_count=("customer_id", "count"),
        average_satisfaction=("satisfaction_score", "mean"),
        average_transactions=("tx_count", "mean"),
        savings_account_users=("savings_account", "sum"),
        auto_savings_users=("auto_savings_enabled", "sum")
    ).round(2).reset_index()

    print(income_analysis.to_string(index=False))

    income_analysis.to_csv(
        REPORT_DIR / "income_group_analysis.csv",
        index=False
    )

    # 7. Customer-level wellbeing indicators
    print("\n6. CREATING CUSTOMER WELLBEING INDICATORS")
    print("-" * 45)

    wellbeing = customers[[
        "customer_id",
        "income_bracket",
        "savings_account",
        "auto_savings_enabled",
        "bill_payment_user",
        "investment_account",
        "credit_utilization_ratio",
        "failed_transactions",
        "satisfaction_score",
        "support_tickets_count"
    ]].copy()

    wellbeing["has_savings_activity"] = (
        wellbeing["savings_account"] |
        wellbeing["auto_savings_enabled"]
    )

    wellbeing["has_failed_transactions"] = (
        wellbeing["failed_transactions"] > 0
    )

    wellbeing["has_support_tickets"] = (
        wellbeing["support_tickets_count"] > 0
    )

    wellbeing.to_csv(
        REPORT_DIR / "customer_wellbeing_indicators.csv",
        index=False
    )

    # 8. Generate visualizations
    print("\n7. GENERATING WELLBEING VISUALIZATIONS")
    print("-" * 45)

    # Financial activity chart
    plt.figure(figsize=(10, 6))
    plt.bar(
        savings_df["Financial Indicator"],
        savings_df["Customer Count"]
    )
    plt.title("Customer Financial Activity")
    plt.xlabel("Financial Indicators")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "financial_activity.png")
    plt.close()

    # Credit utilization chart
    if not credit_data.empty:
        plt.figure(figsize=(9, 5))
        plt.hist(credit_data, bins=30, edgecolor="black")
        plt.title("Credit Utilization Distribution")
        plt.xlabel("Credit Utilization Ratio")
        plt.ylabel("Number of Customers")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "credit_utilization.png")
        plt.close()

    # Feedback sentiment chart
    sentiment_counts = customers["feedback_sentiment"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(
        sentiment_counts.index,
        sentiment_counts.values
    )
    plt.title("Customer Feedback Sentiment")
    plt.xlabel("Feedback Sentiment")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feedback_sentiment.png")
    plt.close()

    # Failed transactions chart
    plt.figure(figsize=(7, 5))
    plt.bar(
        ["No Failed Transactions", "Failed Transactions"],
        [
            len(customers) - failed_customers,
            failed_customers
        ]
    )
    plt.title("Customers with Failed Transactions")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "failed_transactions.png")
    plt.close()

    print("\n" + "=" * 60)
    print("FINANCIAL WELLBEING ANALYSIS COMPLETED!")
    print("=" * 60)

    print("\nReports saved in:", REPORT_DIR)

    print("\nGenerated files:")
    for file in sorted(REPORT_DIR.iterdir()):
        print("-", file.name)


if __name__ == "__main__":
    main()
