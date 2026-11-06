from pathlib import Path
import sqlite3
import re
import textwrap

import pandas as pd
import altair as alt
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FinWise | Financial Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"
DATABASE_PATH = BASE_DIR / "data" / "finwise.db"


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f4f7fc;
        color: #172b4d;
    }

    [data-testid="stSidebar"] {
        background-color: #102a43;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-title {
        color: #102a43;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #627d98;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        color: #102a43;
        font-size: 23px;
        font-weight: 700;
        margin-top: 15px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #d9e2ec;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 7px rgba(16, 42, 67, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #627d98;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #102a43;
        font-weight: 800;
    }

    .info-note {
        background-color: #eaf2fb;
        border-left: 4px solid #2874a6;
        padding: 12px 15px;
        border-radius: 5px;
        color: #243b53;
        margin: 12px 0;
    }

    .footer {
        text-align: center;
        color: #829ab1;
        font-size: 12px;
        padding: 25px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATA LOADING - SQLITE DATABASE
# --------------------------------------------------

@st.cache_data
def load_csv(file_path):
    """
    Load report data from the FinWise SQLite database.

    The original report file path is used to identify
    the corresponding SQLite table.
    """

    path = Path(file_path)

    try:
        relative_path = path.relative_to(REPORT_DIR)

        table_name = str(
            relative_path.with_suffix("")
        ).replace("\\", "_").replace("/", "_")

        table_name = re.sub(
            r"[^a-zA-Z0-9_]",
            "_",
            table_name
        )

        table_name = re.sub(
            r"_+",
            "_",
            table_name
        ).strip("_").lower()

        if not DATABASE_PATH.exists():
            st.error(
                "FinWise database not found. "
                "Please run: python src/database_integration.py"
            )
            return pd.DataFrame()

        with sqlite3.connect(DATABASE_PATH) as connection:

            return pd.read_sql_query(
                f'SELECT * FROM "{table_name}"',
                connection
            )

    except Exception as error:

        st.warning(
            f"Could not load {path.name} from SQLite: {error}"
        )

        return pd.DataFrame()


# --------------------------------------------------
# LOAD DATABASE TABLES
# --------------------------------------------------

customers = load_csv(
    REPORT_DIR / "financial_analysis" / "finwise_customer_analysis.csv"
)

segments = load_csv(
    REPORT_DIR / "customer_segmentation" / "customer_segments.csv"
)

cluster_profiles = load_csv(
    REPORT_DIR / "customer_segmentation" / "cluster_profiles.csv"
)

cluster_evaluation = load_csv(
    REPORT_DIR / "customer_segmentation" / "cluster_evaluation.csv"
)

risk = load_csv(
    REPORT_DIR / "risk_prediction" / "customer_early_warning_report.csv"
)

risk_distribution = load_csv(
    REPORT_DIR / "risk_prediction" / "early_warning_distribution.csv"
)

anomaly_summary = load_csv(
    REPORT_DIR / "risk_prediction" / "anomaly_detection_summary.csv"
)

recommendations = load_csv(
    REPORT_DIR / "recommendations" / "personalised_recommendations.csv"
)

recommendation_categories = load_csv(
    REPORT_DIR / "recommendations" / "recommendation_category_summary.csv"
)

recommendation_counts = load_csv(
    REPORT_DIR / "recommendations" / "recommendation_count_summary.csv"
)

complaints = load_csv(
    REPORT_DIR / "complaint_intelligence" / "customer_complaint_indicators.csv"
)

complaint_topics = load_csv(
    REPORT_DIR / "complaint_intelligence" / "complaint_topic_analysis.csv"
)

sentiment = load_csv(
    REPORT_DIR / "complaint_intelligence" / "feedback_sentiment_analysis.csv"
)

support_tickets = load_csv(
    REPORT_DIR / "complaint_intelligence" / "support_ticket_analysis.csv"
)

wellbeing = load_csv(
    REPORT_DIR / "wellbeing_analysis" / "customer_wellbeing_indicators.csv"
)

income_analysis = load_csv(
    REPORT_DIR / "wellbeing_analysis" / "income_group_analysis.csv"
)

credit_summary = load_csv(
    REPORT_DIR / "wellbeing_analysis" / "credit_utilization_summary.csv"
)

transaction_summary = load_csv(
    REPORT_DIR / "financial_analysis" / "transaction_summary.csv"
)

monthly_transactions = load_csv(
    REPORT_DIR / "financial_analysis" / "monthly_transaction_analysis.csv"
)

banking_products = load_csv(
    REPORT_DIR / "financial_analysis" / "banking_product_analysis.csv"
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def numeric_series(df, column):

    if df.empty or column not in df.columns:
        return pd.Series(dtype="float64")

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()


def format_number(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:,.0f}"


def format_decimal(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:,.2f}"


def format_compact_number(value):

    """Format large values in a compact, readable form."""

    if pd.isna(value):
        return "N/A"

    absolute_value = abs(value)

    if absolute_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:,.2f}T"

    if absolute_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}B"

    if absolute_value >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"

    if absolute_value >= 1_000:
        return f"{value / 1_000:,.2f}K"

    return f"{value:,.0f}"


def wrap_chart_label(value, width=18):

    """Wrap long category labels so chart labels remain readable."""

    label = str(value)

    if len(label) <= width:
        return label

    return textwrap.fill(label, width=width, break_long_words=True)


def truthy_series(series):

    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def show_chart(df, column, title):

    if not df.empty and column in df.columns:

        chart_data = df[column].dropna().astype(str).value_counts().head(15)

        if not chart_data.empty:

            chart_data.index = [
                wrap_chart_label(label) for label in chart_data.index
            ]

            st.subheader(title)
            st.bar_chart(chart_data)


def show_numeric_chart(df, column, title):

    values = numeric_series(df, column)

    if not values.empty:

        st.subheader(title)

        # Create equal-width numeric bins and keep their original numeric order.
        bins = pd.cut(values, bins=10, duplicates="drop")
        counts = bins.value_counts(sort=False)

        is_ratio = "ratio" in column.lower()

        def format_edge(number):
            if is_ratio:
                return f"{number:.2f}"
            return f"{number:,.0f}"

        labels = [
            f"{format_edge(interval.left)}–{format_edge(interval.right)}"
            for interval in counts.index
        ]

        # Altair's explicit sort list prevents labels such as "8–16"
        # from being placed after "72–80" by alphabetical sorting.
        chart_data = pd.DataFrame({
            "Range": labels,
            "Customer Count": counts.to_numpy()
        })

        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Range:N",
                    sort=labels,
                    title=column.replace("_", " ").title(),
                    axis=alt.Axis(labelAngle=-35)
                ),
                y=alt.Y(
                    "Customer Count:Q",
                    title="Customer Count"
                ),
                tooltip=[
                    alt.Tooltip("Range:N", title="Range"),
                    alt.Tooltip(
                        "Customer Count:Q",
                        title="Customer Count",
                        format=","
                    )
                ]
            )
            .properties(height=350)
        )

        st.altair_chart(chart, use_container_width=True)


def show_table(df, title, max_rows=20):

    st.subheader(title)

    if df.empty:

        st.info("No report data available.")

    else:

        st.dataframe(
            df.head(max_rows),
            use_container_width=True,
            hide_index=True
        )


def show_dynamic_summary(df, title):

    if df.empty:

        st.info(f"{title}: Report not available.")
        return

    st.subheader(title)

    categorical = [
        col for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    numeric = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    if categorical and numeric:

        chart_data = df.set_index(
            categorical[0]
        )[numeric[0]].head(15)

        chart_data.index = [
            wrap_chart_label(label) for label in chart_data.index
        ]

        st.bar_chart(chart_data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


def metric_row(items):

    columns = st.columns(len(items))

    for column, (label, value) in zip(columns, items):

        with column:

            st.metric(label, value)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.markdown("# 💠 FinWise")
st.sidebar.markdown("### Financial Intelligence Platform")
st.sidebar.markdown("---")

pages = [
    "Dashboard Overview",
    "Customer Analytics",
    "Financial Analysis",
    "Customer Segmentation",
    "Risk Intelligence",
    "Financial Wellbeing",
    "Complaint Intelligence",
    "Recommendations"
]

selected_page = st.sidebar.radio(
    "NAVIGATION",
    pages
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "FinWise | Data-Driven Financial Inclusion"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<p class="main-title">FinWise</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Financial Wellbeing & Inclusion Intelligence Platform'
    '</p>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# PAGE 1: DASHBOARD OVERVIEW
# --------------------------------------------------

if selected_page == "Dashboard Overview":

    st.markdown("## Executive Overview")

    customer_count = (
        customers["customer_id"].nunique()
        if not customers.empty
        and "customer_id" in customers.columns
        else len(customers)
    )

    total_recommendations = (
        numeric_series(
            recommendations,
            "recommendation_count"
        ).sum()
        if "recommendation_count" in recommendations.columns
        else 0
    )

    high_risk_count = 0

    if not risk.empty and "early_warning_level" in risk.columns:

        high_risk_count = (
            risk["early_warning_level"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("high")
            .sum()
        )

    average_satisfaction = numeric_series(
        customers,
        "satisfaction_score"
    ).mean()

    complaint_count = 0

    if (
        not complaints.empty
        and "has_complaint_topic" in complaints.columns
    ):

        complaint_count = truthy_series(
            complaints["has_complaint_topic"]
        ).sum()

    transaction_amount = (
        numeric_series(
            customers,
            "total_transaction_amount"
        ).sum()
        if "total_transaction_amount" in customers.columns
        else 0
    )

    metric_row([
        (
            "Total Customers",
            format_number(customer_count)
        ),
        (
            "Transaction Amount (Source Units)",
            format_compact_number(transaction_amount)
        ),
        (
            "Total Recommendations",
            format_number(total_recommendations)
        )
    ])

    st.write("")

    metric_row([
        (
            "High Early-Warning Records",
            format_number(high_risk_count)
        ),
        (
            "Average Satisfaction",
            format_decimal(average_satisfaction)
        ),
        (
            "Customers with Complaint Topics",
            format_number(complaint_count)
        )
    ])

    st.markdown("---")

    st.markdown("### Business Insights")

    left, right = st.columns(2)

    with left:

        show_chart(
            customers,
            "income_bracket",
            "Customer Income Distribution"
        )

    with right:

        show_chart(
            risk,
            "early_warning_level",
            "Early-Warning Level Distribution"
        )

    left, right = st.columns(2)

    with left:

        show_chart(
            customers,
            "feedback_sentiment",
            "Customer Feedback Sentiment"
        )

    with right:

        show_chart(
            customers,
            "preferred_transaction_type",
            "Preferred Transaction Types"
        )


# --------------------------------------------------
# PAGE 2: CUSTOMER ANALYTICS
# --------------------------------------------------

elif selected_page == "Customer Analytics":

    st.markdown("## Customer Analytics")

    st.write(
        "Understand customer demographics, income groups and activity."
    )

    if not customers.empty:

        metric_row([
            (
                "Customer Records",
                format_number(len(customers))
            ),
            (
                "Average Age",
                format_decimal(
                    numeric_series(customers, "age").mean()
                )
            ),
            (
                "Average Satisfaction",
                format_decimal(
                    numeric_series(
                        customers,
                        "satisfaction_score"
                    ).mean()
                )
            )
        ])

        left, right = st.columns(2)

        with left:

            show_chart(
                customers,
                "gender",
                "Gender Distribution"
            )

        with right:

            show_chart(
                customers,
                "income_bracket",
                "Income Group Distribution"
            )

        left, right = st.columns(2)

        with left:

            show_chart(
                customers,
                "customer_segment",
                "Existing Customer Segment Distribution"
            )

        with right:

            show_chart(
                customers,
                "feedback_sentiment",
                "Feedback Sentiment"
            )

        show_numeric_chart(
            customers,
            "age",
            "Customer Age Distribution"
        )

        show_table(
            customers[
                [
                    col for col in [
                        "customer_id",
                        "age",
                        "gender",
                        "income_bracket",
                        "occupation",
                        "customer_segment",
                        "satisfaction_score"
                    ]
                    if col in customers.columns
                ]
            ],
            "Customer Details"
        )

    else:

        st.warning("Customer analysis report is missing.")


# --------------------------------------------------
# PAGE 3: FINANCIAL ANALYSIS
# --------------------------------------------------

elif selected_page == "Financial Analysis":

    st.markdown("## Financial Analysis")

    st.write(
        "Explore customer transactions and banking product usage."
    )

    if not customers.empty:

        total_amount = numeric_series(
            customers,
            "total_transaction_amount"
        ).sum()

        avg_amount = numeric_series(
            customers,
            "average_transaction_amount"
        ).mean()

        transaction_count = numeric_series(
            customers,
            "transaction_count"
        ).sum()

        metric_row([
            (
                "Total Transaction Amount",
                format_compact_number(total_amount)
            ),
            (
                "Average Transaction Amount",
                format_decimal(avg_amount)
            ),
            (
                "Transaction Count",
                format_number(transaction_count)
            )
        ])

        st.markdown("---")

        left, right = st.columns(2)

        with left:

            show_chart(
                customers,
                "preferred_transaction_type",
                "Transaction Type Distribution"
            )

        with right:

            show_dynamic_summary(
                banking_products,
                "Banking Product Analysis"
            )

        show_dynamic_summary(
            transaction_summary,
            "Transaction Summary"
        )

        show_dynamic_summary(
            monthly_transactions,
            "Monthly Transaction Analysis"
        )

    else:

        st.warning("Financial analysis report is missing.")


# --------------------------------------------------
# PAGE 4: CUSTOMER SEGMENTATION
# --------------------------------------------------

elif selected_page == "Customer Segmentation":

    st.markdown("## Customer Segmentation")

    st.write(
        "Explore customer groups identified using K-Means clustering."
    )

    if not segments.empty:

        if "customer_cluster" in segments.columns:

            cluster_counts = (
                segments["customer_cluster"].value_counts()
            )

            metric_row([
                (
                    "Total Customers",
                    format_number(len(segments))
                ),
                (
                    "Number of Clusters",
                    format_number(
                        segments["customer_cluster"].nunique()
                    )
                ),
                (
                    "Largest Cluster",
                    format_number(
                        cluster_counts.max()
                        if not cluster_counts.empty
                        else 0
                    )
                )
            ])

            st.subheader("Cluster Distribution")

            st.bar_chart(cluster_counts)

        show_table(
            cluster_profiles,
            "Cluster Profiles"
        )

        show_table(
            cluster_evaluation,
            "Cluster Evaluation"
        )

        show_table(
            segments[
                [
                    col for col in [
                        "customer_id",
                        "customer_cluster",
                        "income_bracket",
                        "credit_card",
                        "credit_utilization_ratio",
                        "satisfaction_score"
                    ]
                    if col in segments.columns
                ]
            ],
            "Customer Cluster Records"
        )

        st.markdown("""
        <div class="info-note">
        <b>Interpretation:</b> The selected two-cluster solution is exploratory.
        Its silhouette score is relatively low, so clusters should not be
        treated as strong behavioural customer personas.
        </div>
        """, unsafe_allow_html=True)

    else:

        st.warning("Customer segmentation report is missing.")


# --------------------------------------------------
# PAGE 5: RISK INTELLIGENCE
# --------------------------------------------------

elif selected_page == "Risk Intelligence":

    st.markdown("## Risk Intelligence")

    st.write(
        "Identify customers with indicators that may require attention."
    )

    if not risk.empty:

        high_count = 0
        moderate_count = 0
        low_count = 0

        if "early_warning_level" in risk.columns:

            levels = (
                risk["early_warning_level"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            high_count = (levels == "high").sum()

            moderate_count = (
                levels == "moderate"
            ).sum()

            low_count = (levels == "low").sum()

        metric_row([
            (
                "High Early-Warning",
                format_number(high_count)
            ),
            (
                "Moderate Early-Warning",
                format_number(moderate_count)
            ),
            (
                "Low Early-Warning",
                format_number(low_count)
            )
        ])

        left, right = st.columns(2)

        with left:

            show_chart(
                risk,
                "early_warning_level",
                "Early-Warning Distribution"
            )

        with right:

            show_chart(
                risk,
                "anomaly_status",
                "Anomaly Detection Results"
            )

        show_numeric_chart(
            risk,
            "early_warning_score",
            "Early-Warning Score Distribution"
        )

        if "early_warning_score" in risk.columns:

            top_risk = risk.sort_values(
                "early_warning_score",
                ascending=False
            )

            show_table(
                top_risk,
                "Customers with Highest Early-Warning Scores"
            )

        show_dynamic_summary(
            anomaly_summary,
            "Anomaly Detection Summary"
        )

        st.markdown("""
        <div class="info-note">
        <b>Important:</b> Early-warning levels are based on manually defined
        rules and thresholds. They are preliminary indicators, not validated
        predictions of loan default or financial distress. Anomaly detection
        identifies unusual patterns and does not prove harmful activity.
        </div>
        """, unsafe_allow_html=True)

    else:

        st.warning("Risk intelligence report is missing.")


# --------------------------------------------------
# PAGE 6: FINANCIAL WELLBEING
# --------------------------------------------------

elif selected_page == "Financial Wellbeing":

    st.markdown("## Financial Wellbeing")

    st.write(
        "Understand savings activity, credit usage and customer satisfaction."
    )

    if not wellbeing.empty:

        savings_count = 0
        auto_savings_count = 0
        bill_payment_count = 0
        investment_count = 0

        for column, variable in [
            ("has_savings_activity", "savings"),
            ("auto_savings_enabled", "auto"),
            ("bill_payment_user", "bill"),
            ("investment_account", "investment")
        ]:

            if column in wellbeing.columns:

                count = truthy_series(
                    wellbeing[column]
                ).sum()

                if variable == "savings":

                    savings_count = count

                elif variable == "auto":

                    auto_savings_count = count

                elif variable == "bill":

                    bill_payment_count = count

                elif variable == "investment":

                    investment_count = count

        metric_row([
            (
                "Savings Activity",
                format_number(savings_count)
            ),
            (
                "Auto-Savings Users",
                format_number(auto_savings_count)
            ),
            (
                "Bill Payment Users",
                format_number(bill_payment_count)
            )
        ])

        metric_row([
            (
                "Investment Account Users",
                format_number(investment_count)
            ),
            (
                "Average Satisfaction",
                format_decimal(
                    numeric_series(
                        wellbeing,
                        "satisfaction_score"
                    ).mean()
                )
            ),
            (
                "Customers Analysed",
                format_number(len(wellbeing))
            )
        ])

        left, right = st.columns(2)

        with left:

            show_numeric_chart(
                wellbeing,
                "credit_utilization_ratio",
                "Credit Utilization Distribution"
            )

        with right:

            show_chart(
                wellbeing,
                "income_bracket",
                "Income Group Distribution"
            )

        show_dynamic_summary(
            income_analysis,
            "Income Group Analysis"
        )

        show_dynamic_summary(
            credit_summary,
            "Credit Utilization Summary"
        )

        show_numeric_chart(
            wellbeing,
            "failed_transactions",
            "Failed Transactions Distribution"
        )

        show_numeric_chart(
            wellbeing,
            "support_tickets_count",
            "Support Ticket Distribution"
        )

    else:

        st.warning("Financial wellbeing report is missing.")


# --------------------------------------------------
# PAGE 7: COMPLAINT INTELLIGENCE
# --------------------------------------------------

elif selected_page == "Complaint Intelligence":

    st.markdown("## Complaint Intelligence")

    st.write(
        "Analyse customer feedback, complaint topics and support activity."
    )

    if not complaints.empty:

        complaint_count = (
            truthy_series(
                complaints["has_complaint_topic"]
            ).sum()
            if "has_complaint_topic" in complaints.columns
            else 0
        )

        metric_row([
            (
                "Customers with Complaint Topics",
                format_number(complaint_count)
            ),
            (
                "Customer Records",
                format_number(len(complaints))
            ),
            (
                "Average Satisfaction",
                format_decimal(
                    numeric_series(
                        complaints,
                        "satisfaction_score"
                    ).mean()
                )
            )
        ])

        left, right = st.columns(2)

        with left:

            show_dynamic_summary(
                complaint_topics,
                "Complaint Topic Analysis"
            )

        with right:

            show_dynamic_summary(
                sentiment,
                "Feedback Sentiment Analysis"
            )

        show_dynamic_summary(
            support_tickets,
            "Support Ticket Analysis"
        )

        show_table(
            complaints,
            "Customer Complaint Indicators"
        )

    else:

        st.warning("Complaint intelligence report is missing.")


# --------------------------------------------------
# PAGE 8: RECOMMENDATIONS
# --------------------------------------------------

elif selected_page == "Recommendations":

    st.markdown("## Personalised Recommendations")

    st.write(
        "Explore rule-based financial guidance and customer support recommendations."
    )

    if not recommendations.empty:

        total_count = numeric_series(
            recommendations,
            "recommendation_count"
        ).sum()

        average_count = numeric_series(
            recommendations,
            "recommendation_count"
        ).mean()

        metric_row([
            (
                "Customers Analysed",
                format_number(len(recommendations))
            ),
            (
                "Total Recommendations",
                format_number(total_count)
            ),
            (
                "Average per Customer",
                format_decimal(average_count)
            )
        ])

        show_dynamic_summary(
            recommendation_categories,
            "Recommendation Category Analysis"
        )

        show_dynamic_summary(
            recommendation_counts,
            "Recommendations per Customer"
        )

        show_table(
            recommendations[
                [
                    col for col in [
                        "customer_id",
                        "recommendations",
                        "recommendation_count",
                        "feedback_sentiment",
                        "satisfaction_score"
                    ]
                    if col in recommendations.columns
                ]
            ],
            "Personalised Customer Recommendations"
        )

        st.markdown("""
        <div class="info-note">
        <b>Note:</b> Recommendations are generated using predefined
        rules to provide educational and service-related guidance.
        They are not regulated investment or financial advice.
        </div>
        """, unsafe_allow_html=True)

    else:

        st.warning("Recommendation report is missing.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    '<div class="footer">'
    'FinWise | Financial Wellbeing & Inclusion Intelligence Platform'
    '<br>Data-driven insights for better financial inclusion'
    '</div>',
    unsafe_allow_html=True
)