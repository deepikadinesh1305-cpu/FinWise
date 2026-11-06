# FinWise – Financial Wellbeing & Inclusion Intelligence Platform

## Project Overview

**FinWise** is a data-driven Financial Wellbeing and Inclusion Intelligence Platform designed to analyze customer financial behavior, transaction patterns, financial wellbeing, customer satisfaction, and potential financial risks.

The platform uses data analytics, machine learning, customer segmentation, complaint intelligence, and a rule-based recommendation engine to generate meaningful insights that can help improve financial services and customer experiences.

It transforms large-scale financial and customer datasets into actionable insights through an interactive Streamlit dashboard.

## Problem Statement

Financial institutions handle large volumes of customer and transaction data but often face challenges in understanding customer financial behavior, identifying potential financial risks, analyzing customer complaints, and improving financial inclusion.

FinWise addresses these challenges by combining data analytics and machine learning techniques to provide insights into customer behavior, financial wellbeing, transaction activities, and service-related issues.

## Project Objectives

- Analyze customer financial behavior and transaction patterns.
- Understand customer satisfaction and financial wellbeing.
- Identify customer segments based on financial and behavioral characteristics.
- Analyze customer complaints and service-related issues.
- Develop an early-warning financial risk scoring mechanism.
- Generate personalized, rule-based financial wellbeing recommendations.
- Present analytical insights through an interactive dashboard.

## Dataset

The project uses the **COFINFAD Colombian Fintech Financial Analytics Dataset**.

**Dataset Source:** [Mendeley Data – COFINFAD Dataset](https://data.mendeley.com/datasets/mhb4zn3258/1)

The dataset contains two main files:

| Dataset | Records | Columns |
|---|---:|---:|
| Customer Data | 48,723 | 54 |
| Transaction Data | 3,159,157 | 4 |

The customer dataset contains demographic, financial, product usage, satisfaction, and customer support information.

The transaction dataset contains customer transaction-related information used for financial and behavioral analysis.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Plotly, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Database:** SQLite
- **Development Environment:** Visual Studio Code

## Key Project Modules

### 1. Data Inspection and Exploratory Data Analysis

- Inspected dataset structure, columns, data types, and missing values.
- Performed data cleaning and preprocessing.
- Analyzed customer demographics and financial characteristics.
- Explored transaction patterns and distributions.

### 2. Financial Analysis

- Analyzed total transaction amounts and average transaction values.
- Studied customer transaction behavior.
- Examined financial product usage.
- Identified patterns in customer financial activities.

### 3. Financial Wellbeing Analysis

- Analyzed customer satisfaction and NPS.
- Examined savings behavior and financial product adoption.
- Studied financial wellbeing indicators.
- Evaluated customer support and service-related patterns.

### 4. Customer Segmentation

- Applied K-Means clustering to group customers based on selected characteristics.
- Analyzed customer segments and their behavioral patterns.
- Evaluated clustering performance using the silhouette score.

### 5. Complaint Intelligence

- Analyzed customer complaints and support tickets.
- Identified common complaint categories.
- Examined complaint patterns and customer satisfaction.
- Generated insights into service-related issues.

### 6. Early-Warning Risk Intelligence

- Developed a rule-based early-warning risk scoring mechanism.
- Analyzed selected customer financial and behavioral indicators.
- Categorized customers based on calculated risk scores.
- Presented risk-related insights through dashboard visualizations.

### 7. Recommendation Engine

- Developed a rule-based recommendation engine.
- Generated recommendations using customer financial and behavioral characteristics.
- Provided educational suggestions related to savings, financial habits, and service usage.

### 8. Database Integration

- Integrated analytical reports into an SQLite database.
- Created 29 database tables from generated CSV reports.
- Enabled structured data retrieval for dashboard visualization.

## Key Analytical Results

| Metric | Result |
|---|---:|
| Total Transaction Amount | 12,544,849,568,700 |
| Average Transaction Amount | 3,564,851.69 |
| Average Customer Satisfaction | 4.162 |
| Total Support Tickets | 48,599 |
| Customers with Support Tickets | 30,809 |
| Customers with Complaint Topics | 24,377 |
| Customers Using Savings Accounts | 38,459 |
| Customers Using Auto-Savings | 19,550 |
| Customers Using Bill Payments | 34,170 |
| Customers with Investment Accounts | 20,738 |
| Customers with Failed Transactions | 8,818 |
| Total Failed Transactions | 9,684 |
| Net Promoter Score (NPS) | -26.76 |
| Generated Recommendations | 112,884 |
| Analytical CSV Reports | 29 |
| SQLite Database Tables | 29 |

### Customer Sentiment Distribution

| Sentiment | Number of Customers |
|---|---:|
| Positive | 38,897 |
| Neutral | 8,346 |
| Negative | 1,480 |

### Customer Complaint Categories

The major complaint categories identified include:

- Customer Service
- Fees
- Transaction Issues
- Account Security
- App Performance

## Interactive Dashboard

The FinWise dashboard is developed using Streamlit and provides eight major analytical sections.

1. **Dashboard Overview** – Displays important financial and customer KPIs.
2. **Customer Analytics** – Explores customer demographics and behavioral patterns.
3. **Financial Analysis** – Presents transaction trends and financial insights.
4. **Customer Segmentation** – Displays customer clusters and segment characteristics.
5. **Risk Intelligence** – Presents early-warning risk scores and related indicators.
6. **Financial Wellbeing** – Analyzes savings behavior, satisfaction, and financial wellbeing.
7. **Complaint Intelligence** – Visualizes customer complaints and support-related insights.
8. **Recommendations** – Displays personalized rule-based recommendations.

## Project Structure

```text
FinWise/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── finwise.db
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── sql/
│
├── src/
│   ├── inspect_data.py
│   ├── eda.py
│   ├── financial_analysis.py
│   ├── wellbeing_analysis.py
│   ├── customer_segmentation.py
│   ├── complaint_intelligence.py
│   ├── risk_prediction.py
│   ├── recommendation_engine.py
│   └── database_integration.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/deepikadinesh1305-cpu/FinWise.git
cd FinWise
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the Dataset

Download the COFINFAD dataset from:

https://data.mendeley.com/datasets/mhb4zn3258/1

Place the original dataset files in the appropriate `data/raw/` directory.

### 6. Generate the Analytical Reports and Database

Run the Python scripts in the `src/` directory in the required pipeline order to perform preprocessing, analysis, segmentation, risk scoring, recommendation generation, and database integration.

The SQLite database is generated during database integration.

### 7. Run the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at:

```text
http://localhost:8501
```

**Important:** The raw datasets, processed datasets, and generated SQLite database are excluded from this GitHub repository. Download the dataset and execute the processing pipeline to recreate the required files before launching the dashboard.

## Limitations

- The K-Means clustering model achieved a silhouette score of approximately 0.1372, indicating weak cluster separation.
- The early-warning risk score is based on manually defined rules and has not been validated as a predictive model of financial distress or default.
- Recommendations are generated using predefined rules and are intended for educational and financial wellbeing guidance, not regulated investment advice.
- The analysis depends on the quality and availability of the supplied dataset.
- Customer sentiment and satisfaction indicators may overlap and should be interpreted in context.

## Future Enhancements

- Develop and validate advanced machine learning models for financial risk prediction.
- Improve customer segmentation using additional clustering techniques.
- Integrate real-time financial transaction monitoring.
- Enhance recommendation personalization using machine learning.
- Add multilingual dashboard support.
- Explore integration with secure financial data sources.

## Conclusion

FinWise demonstrates how data analytics, machine learning, and interactive visualization can be combined to understand customer financial behavior and financial wellbeing.

By integrating customer analytics, financial analysis, segmentation, complaint intelligence, early-warning risk scoring, and personalized recommendations, the platform aims to support better financial insights and more informed decision-making.

---

**Project Name:** FinWise – Financial Wellbeing & Inclusion Intelligence Platform

**Domain:** Data Analytics | Machine Learning | Financial Technology

**GitHub Repository:** https://github.com/deepikadinesh1305-cpu/FinWise