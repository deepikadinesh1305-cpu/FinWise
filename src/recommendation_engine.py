
import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

CUSTOMER_PATH = os.path.join(
    "data", "raw", "customer_data.csv"
)

OUTPUT_DIR = os.path.join(
    "reports", "recommendations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD CUSTOMER DATA
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINWISE - PERSONALISED FINANCIAL RECOMMENDATION ENGINE")
print("=" * 65)

df = pd.read_csv(CUSTOMER_PATH)

print("\nCustomer dataset shape:", df.shape)

# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def get_value(row, column, default=None):
    """Safely retrieve a customer value."""
    if column not in row.index:
        return default

    return row[column]


def is_enabled(value):
    """Check whether a product or feature is enabled."""
    if pd.isna(value):
        return False

    if isinstance(value, str):
        return value.strip().lower() in [
            "yes", "true", "1", "active", "enabled"
        ]

    return value == 1 or value is True


def get_number(row, column):
    """Convert a value into a number safely."""
    value = get_value(row, column)

    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# --------------------------------------------------
# 4. GENERATE PERSONAL FINANCIAL RECOMMENDATIONS
# --------------------------------------------------

def generate_recommendations(row):

    recommendations = []

    # Savings recommendations
    savings = get_value(row, "savings_account")
    auto_savings = get_value(row, "auto_savings_enabled")

    if not is_enabled(savings):
        recommendations.append(
            "Explore suitable savings account options "
            "and learn about their features and charges."
        )

    elif not is_enabled(auto_savings):
        recommendations.append(
            "Consider setting up automatic savings transfers "
            "based on your income and budget."
        )

    # Credit utilisation recommendations
    credit_utilization = get_number(
        row, "credit_utilization_ratio"
    )

    if credit_utilization is not None:

        if credit_utilization >= 0.70:
            recommendations.append(
                "Review your credit utilisation and consider "
                "reducing outstanding balances where possible."
            )

        elif credit_utilization >= 0.40:
            recommendations.append(
                "Monitor your credit utilisation regularly "
                "and plan repayments within your budget."
            )

    # Failed transaction recommendations
    failed_transactions = get_number(
        row, "failed_transactions"
    )

    if failed_transactions is not None and failed_transactions > 0:
        recommendations.append(
            "Review failed transaction records and contact "
            "bank support if any issue remains unresolved."
        )

    # Customer satisfaction recommendations
    satisfaction = get_number(
        row, "satisfaction_score"
    )

    if satisfaction is not None and satisfaction <= 3:
        recommendations.append(
            "Provide feedback about your banking experience "
            "so customer support can address your concerns."
        )

    # Support ticket recommendations
    support_tickets = get_number(
        row, "support_tickets_count"
    )

    if support_tickets is not None and support_tickets >= 3:
        recommendations.append(
            "Review your existing support requests and "
            "follow up on unresolved issues."
        )

    # Ticket resolution recommendations
    resolution_ratio = get_number(
        row, "resolved_tickets_ratio"
    )

    if resolution_ratio is not None and resolution_ratio < 0.50:
        recommendations.append(
            "Request assistance for unresolved support tickets "
            "and track their resolution status."
        )

    # Investment account recommendations
    investment = get_value(row, "investment_account")

    if not is_enabled(investment):
        recommendations.append(
            "Explore basic financial education resources "
            "about investment options, risks and suitability."
        )

    # Digital banking engagement
    app_usage = get_number(row, "app_logins_frequency")

    if app_usage is not None and app_usage < 5:
        recommendations.append(
            "Explore digital banking features and learn "
            "how to use available online services."
        )

    # Customer feedback
    sentiment = str(
        get_value(row, "feedback_sentiment", "")
    ).strip().lower()

    if sentiment == "negative":
        recommendations.append(
            "Review your recent feedback and contact "
            "customer support if additional help is needed."
        )

    if not recommendations:
        recommendations.append(
            "Continue monitoring your financial activity "
            "and review available banking services regularly."
        )

    return recommendations


# --------------------------------------------------
# 5. GENERATE RECOMMENDATIONS FOR ALL CUSTOMERS
# --------------------------------------------------

print("\nGenerating personalised recommendations...")

recommendation_lists = []

for _, customer in df.iterrows():
    recommendation_lists.append(
        generate_recommendations(customer)
    )

df["recommendations"] = recommendation_lists

df["recommendation_count"] = df[
    "recommendations"
].apply(len)

df["recommendations"] = df[
    "recommendations"
].apply(lambda items: " | ".join(items))

print("Recommendations generated successfully!")

# --------------------------------------------------
# 6. CUSTOMER RECOMMENDATION REPORT
# --------------------------------------------------

output_columns = ["customer_id"]

possible_columns = [
    "savings_account",
    "auto_savings_enabled",
    "credit_utilization_ratio",
    "failed_transactions",
    "satisfaction_score",
    "support_tickets_count",
    "resolved_tickets_ratio",
    "investment_account",
    "feedback_sentiment",
    "recommendations",
    "recommendation_count"
]

for column in possible_columns:
    if column in df.columns:
        output_columns.append(column)

recommendation_report = df[
    list(dict.fromkeys(output_columns))
].copy()

recommendation_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "personalised_recommendations.csv"
    ),
    index=False
)

# --------------------------------------------------
# 7. RECOMMENDATION COUNT ANALYSIS
# --------------------------------------------------

print("\n" + "-" * 65)
print("RECOMMENDATION COUNT ANALYSIS")
print("-" * 65)

recommendation_summary = (
    df["recommendation_count"]
    .value_counts()
    .sort_index()
    .rename_axis("recommendation_count")
    .reset_index(name="customer_count")
)

recommendation_summary["percentage"] = (
    recommendation_summary["customer_count"]
    / len(df) * 100
).round(2)

print(recommendation_summary.to_string(index=False))

recommendation_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "recommendation_count_summary.csv"
    ),
    index=False
)

# --------------------------------------------------
# 8. INDIVIDUAL RECOMMENDATION CATEGORY ANALYSIS
# --------------------------------------------------

print("\n" + "-" * 65)
print("RECOMMENDATION CATEGORY ANALYSIS")
print("-" * 65)

recommendation_categories = {}

for recommendations in recommendation_lists:

    for recommendation in recommendations:

        if "savings" in recommendation.lower():
            category = "Savings"

        elif "credit utilisation" in recommendation.lower():
            category = "Credit Awareness"

        elif "transaction" in recommendation.lower():
            category = "Transaction Support"

        elif "support" in recommendation.lower() or \
             "feedback" in recommendation.lower():
            category = "Customer Support"

        elif "investment" in recommendation.lower():
            category = "Investment Education"

        elif "digital banking" in recommendation.lower():
            category = "Digital Banking"

        else:
            category = "General Financial Guidance"

        recommendation_categories[category] = (
            recommendation_categories.get(category, 0) + 1
        )

category_summary = pd.DataFrame(
    list(recommendation_categories.items()),
    columns=["recommendation_category", "recommendation_count"]
)

category_summary = category_summary.sort_values(
    by="recommendation_count",
    ascending=False
)

print(category_summary.to_string(index=False))

category_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "recommendation_category_summary.csv"
    ),
    index=False
)

# --------------------------------------------------
# 9. TOP CUSTOMERS BY NUMBER OF RECOMMENDATIONS
# --------------------------------------------------

print("\n" + "-" * 65)
print("CUSTOMERS WITH MOST RECOMMENDATIONS")
print("-" * 65)

top_customers = recommendation_report.sort_values(
    by="recommendation_count",
    ascending=False
).head(10)

print(
    top_customers[
        ["customer_id", "recommendation_count"]
    ].to_string(index=False)
)

top_customers.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_customer_recommendations.csv"
    ),
    index=False
)

# --------------------------------------------------
# 10. RECOMMENDATION COUNT CHART
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    recommendation_summary[
        "recommendation_count"
    ].astype(str),
    recommendation_summary["customer_count"]
)

plt.title("Customer Recommendation Count Distribution")
plt.xlabel("Number of Recommendations")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "recommendation_count_distribution.png"
    ),
    dpi=300
)

plt.close()

# --------------------------------------------------
# 11. RECOMMENDATION CATEGORY CHART
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.barh(
    category_summary["recommendation_category"],
    category_summary["recommendation_count"]
)

plt.title("Personalised Recommendation Categories")
plt.xlabel("Number of Recommendations")
plt.ylabel("Recommendation Category")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "recommendation_category_distribution.png"
    ),
    dpi=300
)

plt.close()

# --------------------------------------------------
# 12. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("PERSONALISED RECOMMENDATION ENGINE COMPLETED!")
print("=" * 65)

print("\nTotal customers analysed:", len(df))

print(
    "Total recommendations generated:",
    df["recommendation_count"].sum()
)

print(
    "Average recommendations per customer:",
    round(df["recommendation_count"].mean(), 2)
)

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

print("\nOutput folder:", OUTPUT_DIR)

print(
    "\nNote: Recommendations are general educational "
    "and service guidance, not personalised regulated "
    "financial or investment advice."
)
