import pandas as pd
import os

# Project data folder
DATA_PATH = os.path.join("data", "raw")

# Dataset file paths
customer_file = os.path.join(DATA_PATH, "customer_data.csv")
transaction_file = os.path.join(DATA_PATH, "transactions_data.csv")


def inspect_dataset(file_path, dataset_name):
    print("\n" + "=" * 60)
    print(f"INSPECTING {dataset_name.upper()} DATASET")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(file_path)

    # Display basic information
    print("\nDataset Shape:", df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\nDuplicate Rows:", df.duplicated().sum())

    print("\nDataset Information:")
    df.info()


# Inspect customer dataset
inspect_dataset(customer_file, "Customer")

# Inspect transaction dataset
inspect_dataset(transaction_file, "Transaction")