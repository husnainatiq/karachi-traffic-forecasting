"""
Karachi Traffic Congestion Forecasting - Preprocessing Pipeline (Week 2)

What this script does, step by step:
1. Reads the RAW data collected by collect_traffic_data.py
   (data/raw/traffic_data.csv) - this file is NEVER modified.
2. Parses the timestamp column into a real datetime (not just text).
3. Removes exact duplicate rows (in case the hourly job ever ran twice
   by accident, e.g. a manual test run plus the scheduled run in the same hour).
4. Converts numeric columns to proper numbers, turning any blank/failed
   readings into a clearly marked "missing" value (NaN) instead of empty text.
5. Saves the result as a NEW file: data/processed/traffic_data_clean.csv

Run it with:
    python src/preprocess_data.py
(assuming you're in the project's root folder, with venv activated)
"""

import os
import pandas as pd

RAW_FILE = os.path.join("data", "raw", "traffic_data_synthetic.csv")
PROCESSED_FILE = os.path.join("data", "processed", "traffic_data_clean.csv")

# Columns that should be treated as numbers
NUMERIC_COLUMNS = [
    "travel_time_with_traffic_sec",
    "free_flow_time_sec",
    "congestion_index",
    "temperature_c",
    "precipitation_mm",
    "weather_code",
]


def load_raw_data():
    """Reads the raw CSV file into a pandas DataFrame (a table, like a spreadsheet in Python)."""
    if not os.path.isfile(RAW_FILE):
        raise FileNotFoundError(
            f"Raw data file not found at {RAW_FILE}. "
            "Make sure collect_traffic_data.py has run at least once."
        )
    df = pd.read_csv(RAW_FILE)
    print(f"Loaded {len(df)} raw rows from {RAW_FILE}")
    return df


def parse_timestamps(df):
    """
    Converts the 'timestamp' column from plain text (e.g. '2026-07-25T17:32:44')
    into a real datetime object, so later scripts can filter by date/hour easily.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def remove_duplicates(df):
    """
    Removes rows that are exact duplicates of another row - e.g. if the
    scheduled task accidentally ran twice for the same route at the same timestamp.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    if removed > 0:
        print(f"Removed {removed} duplicate row(s)")
    else:
        print("No duplicate rows found")
    return df


def cast_numeric_columns(df):
    """
    Makes sure numeric columns are actually stored as numbers.
    Any value that can't be converted (like an empty cell from a failed
    API call) becomes NaN (pandas' way of marking 'missing data') instead
    of silently being treated as text.
    """
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def report_missing_data(df):
    """Prints a quick summary of how much data is missing per column - useful to know before modeling later."""
    print("\nMissing data summary:")
    missing_counts = df[NUMERIC_COLUMNS].isna().sum()
    total_rows = len(df)
    for col, count in missing_counts.items():
        pct = round((count / total_rows) * 100, 1) if total_rows > 0 else 0
        print(f"  {col}: {count} missing ({pct}%)")


def save_processed_data(df):
    """Saves the cleaned DataFrame to the processed data folder."""
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"\nSaved {len(df)} cleaned rows to {PROCESSED_FILE}")


def run_preprocessing():
    df = load_raw_data()
    df = parse_timestamps(df)
    df = remove_duplicates(df)
    df = cast_numeric_columns(df)
    report_missing_data(df)
    save_processed_data(df)


if __name__ == "__main__":
    run_preprocessing()