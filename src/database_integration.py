
"""
FinWise - SQLite Database Integration

Imports all CSV analytics reports into a local SQLite database.
Database location: FinWise/data/finwise.db
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "finwise.db"


def create_database():
    """Create the database directory and SQLite database."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    print("\nFinWise SQLite database created successfully.")
    print(f"Database location: {DATABASE_PATH}")

    return connection


def clean_table_name(file_path):
    """Generate a valid, unique-style table name from a report path."""

    relative_path = file_path.relative_to(REPORTS_DIR)

    table_name = str(relative_path.with_suffix("")).replace("\\", "_").replace("/", "_")

    table_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
    table_name = re.sub(r"_+", "_", table_name).strip("_")

    return table_name.lower()


def import_csv_to_database(connection, csv_path):
    """Import one CSV report into SQLite using chunks."""

    table_name = clean_table_name(csv_path)

    print(f"\nImporting: {csv_path.relative_to(PROJECT_ROOT)}")

    total_rows = 0
    first_chunk = True

    try:
        for chunk in pd.read_csv(csv_path, chunksize=10000):

            chunk.to_sql(
                name=table_name,
                con=connection,
                if_exists="replace" if first_chunk else "append",
                index=False
            )

            total_rows += len(chunk)
            first_chunk = False

        if first_chunk:
            print("Skipped: CSV contains no data.")
            return None

        print(f"Table created: {table_name}")
        print(f"Rows imported: {total_rows:,}")

        return {
            "source_file": str(csv_path.relative_to(PROJECT_ROOT)),
            "table_name": table_name,
            "rows_imported": total_rows,
            "imported_at": datetime.now().isoformat(timespec="seconds")
        }

    except pd.errors.EmptyDataError:
        print("Skipped: CSV file is empty.")
        return None

    except Exception as error:
        print(f"Import failed: {error}")
        raise


def create_import_log(connection, import_records):
    """Store information about imported reports."""

    if not import_records:
        return

    log_df = pd.DataFrame(import_records)

    log_df.to_sql(
        name="database_import_log",
        con=connection,
        if_exists="replace",
        index=False
    )

    print("\nDatabase import log created.")


def verify_database(connection):
    """Display all database tables and their record counts."""

    cursor = connection.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
    """)

    tables = cursor.fetchall()

    print("\n" + "=" * 55)
    print("FINWISE DATABASE VERIFICATION")
    print("=" * 55)

    for (table_name,) in tables:

        if table_name == "database_import_log":
            continue

        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        row_count = cursor.fetchone()[0]

        print(f"{table_name:<40} {row_count:>10,} rows")

    print("=" * 55)
    print(f"Total tables: {len(tables) - 1}")


def main():
    """Run the complete database integration process."""

    print("\n" + "=" * 55)
    print("FINWISE - DATABASE INTEGRATION")
    print("=" * 55)

    if not REPORTS_DIR.exists():
        print(f"Reports folder not found: {REPORTS_DIR}")
        return

    csv_files = sorted(REPORTS_DIR.rglob("*.csv"))

    if not csv_files:
        print("No CSV report files found.")
        return

    print(f"\nCSV reports found: {len(csv_files)}")

    connection = create_database()

    import_records = []

    try:
        for csv_path in csv_files:

            result = import_csv_to_database(connection, csv_path)

            if result:
                import_records.append(result)

        create_import_log(connection, import_records)

        connection.commit()

        verify_database(connection)

        print("\n" + "=" * 55)
        print("DATABASE INTEGRATION COMPLETED SUCCESSFULLY!")
        print(f"Reports imported: {len(import_records)}")
        print(f"Database: {DATABASE_PATH}")
        print("=" * 55)

    except Exception as error:
        connection.rollback()
        print(f"\nDatabase integration stopped: {error}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
