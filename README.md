\# FinWise – Financial Wellbeing \& Inclusion Intelligence Platform



\## Project Overview



FinWise is a data analytics and intelligence platform designed to analyse customer financial behaviour, understand financial wellbeing, identify potential risk indicators, and generate personalised recommendations.



The platform combines Python-based data analytics, machine learning techniques, SQL database integration, and an interactive Streamlit dashboard to convert raw financial data into meaningful insights.



\## Problem Statement



Financial institutions handle large amounts of customer and transaction data. However, identifying customer behaviour patterns, understanding financial wellbeing, detecting potential risk indicators, and improving customer experience can be challenging.



FinWise addresses these challenges through data-driven analysis, customer segmentation, risk intelligence, complaint analysis, and recommendation generation.



\## Dataset



\*\*Dataset:\*\* COFINFAD – Colombian Fintech Financial Analytics Dataset



\*\*Source:\*\* https://data.mendeley.com/datasets/mhb4zn3258/1



The project uses two datasets:



\- Customer Data: 48,723 records and 54 columns.

\- Transaction Data: 3,159,157 records and 4 columns.



\## Technologies Used



\- \*\*Programming:\*\* Python

\- \*\*Data Analysis:\*\* Pandas, NumPy

\- \*\*Machine Learning:\*\* Scikit-learn

\- \*\*Database:\*\* SQLite

\- \*\*Dashboard:\*\* Streamlit

\- \*\*Visualisation:\*\* Plotly, Matplotlib, Seaborn

\- \*\*Development Environment:\*\* Visual Studio Code



\## Key Modules



\### 1. Exploratory Data Analysis

Analyses customer and transaction data to understand distributions, patterns, relationships, and missing values.



\### 2. Financial Analysis

Examines transaction amounts, transaction behaviour, banking products, and financial activity.



\### 3. Financial Wellbeing Analysis

Studies indicators such as savings behaviour, credit utilisation, failed transactions, and customer satisfaction.



\### 4. Customer Segmentation

Uses clustering techniques to group customers based on their financial and behavioural characteristics.



\### 5. Complaint Intelligence

Analyses customer complaints, feedback sentiment, complaint categories, and support ticket information.



\### 6. Risk Intelligence

Uses a rule-based early-warning scoring approach to identify customers who may need additional attention.



\### 7. Recommendation Engine

Generates rule-based personalised recommendations based on customer financial and behavioural indicators.



\### 8. Database Integration

Imports generated analytical reports into an SQLite database for structured storage and dashboard access.



\## Key Results



\- Analysed 48,723 customer records.

\- Processed more than 3.15 million transactions.

\- Generated 29 analytical CSV reports.

\- Created 29 SQLite database tables.

\- Identified customer segments using K-Means clustering.

\- Analysed customer complaints, sentiment, and support tickets.

\- Generated 112,884 customer recommendations.

\- Developed an interactive dashboard with eight analytical sections.



\## Dashboard Sections



1\. Dashboard Overview

2\. Customer Analytics

3\. Financial Analysis

4\. Customer Segmentation

5\. Risk Intelligence

6\. Financial Wellbeing

7\. Complaint Intelligence

8\. Recommendations



\## Project Structure



```text

FinWise/

├── dashboard/

├── data/

│   ├── raw/

│   ├── processed/

│   └── finwise.db

├── models/

├── notebooks/

├── reports/

├── sql/

├── src/

└── venv/

```



\## Installation and Execution



\### 1. Clone the Repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd FinWise

```



\### 2. Create a Virtual Environment



```bash

python -m venv venv

```



\### 3. Activate the Environment



Windows:



```bash

venv\\Scripts\\activate

```



\### 4. Install Dependencies



```bash

pip install -r requirements.txt

```



\### 5. Run the Dashboard



```bash

streamlit run dashboard/app.py

```



\## Important Limitations



\- Customer segmentation is exploratory; the observed silhouette score indicates limited separation between clusters.

\- Risk intelligence uses a rule-based early-warning score and is not a validated prediction of loan default or financial distress.

\- Recommendations are rule-based educational and service suggestions, not regulated financial or investment advice.

\- Results depend on the quality and limitations of the available dataset.



\## Conclusion



FinWise demonstrates how data analytics, machine learning techniques, database management, and interactive visualisation can be combined to understand financial behaviour and customer wellbeing.



The platform aims to support better data-driven insights, customer engagement, and financial inclusion.

