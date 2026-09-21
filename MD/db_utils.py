"""Database utilities for the MD (Mahalanobis) monitor.

Fetches the last completed minute of raw metrics from the database and
appends the rows to the CSV file that the MD monitor reads, so the
monitor can pick up new minute data without manual intervention.

The query output matches the CSV layout (datetime, module, tags, metrics)
that cleaning_utils.py expects.
"""
import csv
import os

import psycopg2


# Path to the SQL query that fetches the last completed minute.
SQL_PATH = os.path.join(os.path.dirname(__file__), '..', 'SQL', 'fetch_last_minute.sql')

# CSV columns matching the data file layout.
CSV_COLUMNS = ['datetime', 'module', 'tags', 'metrics']


def load_query(sql_path=SQL_PATH):
    """Read the SQL query from file."""
    with open(sql_path, encoding='utf-8') as f:
        return f.read()


def fetch_last_minute(conn_params, sql_path=SQL_PATH):
    """Run the SQL query and return the rows for the last completed minute.

    conn_params is a dict passed to psycopg2.connect (host, dbname, user,
    password, port, ...).
    """
    query = load_query(sql_path)

    conn = psycopg2.connect(**conn_params)
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    return rows


def append_rows_to_csv(rows, csv_path):
    """Append the fetched rows to the CSV, creating the header if needed.

    rows: list of (datetime, module, tags, metrics) tuples.
    """
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            writer.writerow(CSV_COLUMNS)

        for row in rows:
            writer.writerow(row)


def append_rows_to_daily_csv(rows, base_dir):
    """Append rows to per-day CSV files (one per calendar day).

    Each row is routed to raw_data_YYYY-MM-DD.csv based on its datetime,
    so every day gets its own file for easier inspection.

    rows: list of (datetime, module, tags, metrics) tuples.
    base_dir: directory where the daily files are created.
    """
    for row in rows:
        date_part = str(row[0])[:10]  # YYYY-MM-DD
        daily_path = os.path.join(base_dir, f"raw_data_{date_part}.csv")
        file_exists = os.path.exists(daily_path) and os.path.getsize(daily_path) > 0
        with open(daily_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            if not file_exists:
                writer.writerow(CSV_COLUMNS)
            writer.writerow(row)


def fetch_and_append(conn_params, csv_path, sql_path=SQL_PATH):
    """Fetch the last minute from the DB and append it to the CSV.

    Appends to both the main CSV (csv_path) and a per-day CSV file
    in the same directory, so every day gets its own raw data file.

    Returns the number of rows appended (0 if nothing new).
    """
    rows = fetch_last_minute(conn_params, sql_path)

    if not rows:
        return 0

    append_rows_to_csv(rows, csv_path)
    append_rows_to_daily_csv(rows, os.path.dirname(csv_path))
    return len(rows)
