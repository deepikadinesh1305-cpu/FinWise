
import os
import re
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

CUSTOMER_PATH = os.path.join(
    "data", "raw", "customer_data.csv"
)

OUTPUT_DIR = os.path.join(
    "reports", "complaint_intelligence"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD CUSTOMER DATA
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINWISE - CUSTOMER COMPLAINT & FEEDBACK INTELLIGENCE")
print("=" * 65)

df = pd.read_csv(CUSTOMER_PATH)

print("\nCustomer dataset shape:", df.shape)

# --------------------------------------------------
# 3. IDENTIFY RELEVANT COLUMNS
# --------------------------------------------------

print("\nAvailable feedback-related columns:")

keywords = [
    "feedback",
    "sentiment",
    "complaint",
    "satisfaction",
    "support",
    "ticket"
]

for column in df.columns:
    if any(word in column.lower() for word in keywords):
        print("-", column)

# --------------------------------------------------
# 4. FEEDBACK SENTIMENT ANALYSIS
# --------------------------------------------------

print("\n" + "-" * 65)
print("CUSTOMER FEEDBACK SENTIMENT ANALYSIS")
print("-" * 65)

if "feedback_sentiment" in df.columns:

    sentiment = (
        df["feedback_sentiment"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.title()
    )

    sentiment_counts = (
        sentiment.value_counts()
        .rename_axis("sentiment")
        .reset_index(name="customer_count")
    )

    sentiment_counts["percentage"] = (
        sentiment_counts["customer_count"]
        / len(df) * 100
    )

    print(sentiment_counts.to_string(index=False))

    sentiment_counts.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feedback_sentiment_analysis.csv"
        ),
        index=False
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        sentiment_counts["sentiment"],
        sentiment_counts["customer_count"]
    )

    plt.title("Customer Feedback Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feedback_sentiment_distribution.png"
        ),
        dpi=300
    )

    plt.close()

else:
    print("Feedback sentiment column not found.")

# --------------------------------------------------
# 5. COMPLAINT TOPIC ANALYSIS
# --------------------------------------------------

print("\n" + "-" * 65)
print("CUSTOMER COMPLAINT TOPIC ANALYSIS")
print("-" * 65)

if "complaint_topics" in df.columns:

    complaints = df["complaint_topics"].dropna().astype(str)

    complaint_counts = (
        complaints.value_counts()
        .rename_axis("complaint_topic")
        .reset_index(name="customer_count")
    )

    print("\nTop 15 complaint topics:")
    print(
        complaint_counts.head(15).to_string(index=False)
    )

    complaint_counts.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "complaint_topic_analysis.csv"
        ),
        index=False
    )

    plt.figure(figsize=(11, 6))

    top_complaints = complaint_counts.head(10)

    plt.barh(
        top_complaints["complaint_topic"].astype(str),
        top_complaints["customer_count"]
    )

    plt.title("Top 10 Customer Complaint Topics")
    plt.xlabel("Number of Customers")
    plt.ylabel("Complaint Topic")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "top_complaint_topics.png"
        ),
        dpi=300
    )

    plt.close()

else:
    print("Complaint topics column not found.")

# --------------------------------------------------
# 6. CUSTOMER SATISFACTION ANALYSIS
# --------------------------------------------------

print("\n" + "-" * 65)
print("CUSTOMER SATISFACTION ANALYSIS")
print("-" * 65)

if "satisfaction_score" in df.columns:

    satisfaction = pd.to_numeric(
        df["satisfaction_score"],
        errors="coerce"
    )

    satisfaction_summary = pd.DataFrame({
        "metric": [
            "Mean Satisfaction",
            "Median Satisfaction",
            "Minimum Satisfaction",
            "Maximum Satisfaction"
        ],
        "value": [
            satisfaction.mean(),
            satisfaction.median(),
            satisfaction.min(),
            satisfaction.max()
        ]
    })

    print(satisfaction_summary.to_string(index=False))

    satisfaction_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "customer_satisfaction_analysis.csv"
        ),
        index=False
    )

    satisfaction_counts = (
        satisfaction.value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        satisfaction_counts.index.astype(str),
        satisfaction_counts.values
    )

    plt.title("Customer Satisfaction Score Distribution")
    plt.xlabel("Satisfaction Score")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "satisfaction_distribution.png"
        ),
        dpi=300
    )

    plt.close()

else:
    print("Satisfaction score column not found.")

# --------------------------------------------------
# 7. SUPPORT TICKET ANALYSIS
# --------------------------------------------------

print("\n" + "-" * 65)
print("CUSTOMER SUPPORT TICKET ANALYSIS")
print("-" * 65)

support_column = None

for column in df.columns:
    if column.lower() == "support_tickets_count":
        support_column = column
        break

if support_column:

    support = pd.to_numeric(
        df[support_column],
        errors="coerce"
    )

    support_summary = pd.DataFrame({
        "metric": [
            "Total Support Tickets",
            "Average Tickets per Customer",
            "Customers with Support Tickets",
            "Customers without Support Tickets",
            "Maximum Tickets for a Customer"
        ],
        "value": [
            support.sum(),
            support.mean(),
            (support > 0).sum(),
            (support == 0).sum(),
            support.max()
        ]
    })

    print(support_summary.to_string(index=False))

    support_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "support_ticket_analysis.csv"
        ),
        index=False
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        support.dropna(),
        bins=20
    )

    plt.title("Distribution of Customer Support Tickets")
    plt.xlabel("Number of Support Tickets")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "support_ticket_distribution.png"
        ),
        dpi=300
    )

    plt.close()

else:
    print("Support ticket column not found.")

# --------------------------------------------------
# 8. SENTIMENT VS SATISFACTION ANALYSIS
# --------------------------------------------------

print("\n" + "-" * 65)
print("SENTIMENT VS CUSTOMER SATISFACTION")
print("-" * 65)

if (
    "feedback_sentiment" in df.columns
    and "satisfaction_score" in df.columns
):

    sentiment_satisfaction = df[
        ["feedback_sentiment", "satisfaction_score"]
    ].copy()

    sentiment_satisfaction[
        "satisfaction_score"
    ] = pd.to_numeric(
        sentiment_satisfaction["satisfaction_score"],
        errors="coerce"
    )

    sentiment_satisfaction = (
        sentiment_satisfaction
        .dropna(subset=["satisfaction_score"])
        .groupby("feedback_sentiment")[
            "satisfaction_score"
        ]
        .agg(
            customer_count="count",
            average_satisfaction="mean",
            median_satisfaction="median"
        )
        .round(2)
        .reset_index()
    )

    print(
        sentiment_satisfaction.to_string(index=False)
    )

    sentiment_satisfaction.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "sentiment_vs_satisfaction.csv"
        ),
        index=False
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        sentiment_satisfaction["feedback_sentiment"].astype(str),
        sentiment_satisfaction["average_satisfaction"]
    )

    plt.title("Average Satisfaction by Feedback Sentiment")
    plt.xlabel("Feedback Sentiment")
    plt.ylabel("Average Satisfaction Score")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "sentiment_satisfaction_comparison.png"
        ),
        dpi=300
    )

    plt.close()

else:
    print("Required columns for comparison not found.")

# --------------------------------------------------
# 9. COMPLAINT AND SUPPORT INDICATORS
# --------------------------------------------------

print("\n" + "-" * 65)
print("CUSTOMER COMPLAINT AND SUPPORT INDICATORS")
print("-" * 65)

indicator_data = pd.DataFrame({
    "customer_id": df["customer_id"]
})

if "complaint_topics" in df.columns:
    indicator_data["has_complaint_topic"] = (
        df["complaint_topics"].notna().astype(int)
    )

if support_column:
    indicator_data["support_ticket_count"] = pd.to_numeric(
        df[support_column],
        errors="coerce"
    ).fillna(0)

if "feedback_sentiment" in df.columns:
    indicator_data["feedback_sentiment"] = (
        df["feedback_sentiment"]
    )

if "satisfaction_score" in df.columns:
    indicator_data["satisfaction_score"] = pd.to_numeric(
        df["satisfaction_score"],
        errors="coerce"
    )

indicator_data.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "customer_complaint_indicators.csv"
    ),
    index=False
)

print(
    "\nCustomer complaint indicator report generated."
)

# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("COMPLAINT INTELLIGENCE COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nTotal customers analysed:", len(df))

if "feedback_sentiment" in df.columns:
    print(
        "Feedback records:",
        df["feedback_sentiment"].notna().sum()
    )

if "complaint_topics" in df.columns:
    print(
        "Customers with recorded complaint topics:",
        df["complaint_topics"].notna().sum()
    )

print("\nGenerated reports are saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

print("\nFinWise complaint intelligence module finished!")
