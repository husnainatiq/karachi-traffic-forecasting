"""
Karachi Traffic Congestion Forecasting - Feature Engineering (Week 3)

What this script does, step by step:
1. Reads the CLEANED data (output of preprocess_data.py) -
   data/processed/traffic_data_clean.csv
2. Handles remaining missing values in congestion_index using interpolation
   (filling gaps using nearby known values, per-route, in time order) -
   better than dropping rows, since dropping would break lag features
   right around each gap.
3. Adds calendar flag columns: hour_of_day, day_of_week, is_weekend, is_friday
4. Adds lag features per route: congestion_1hr_ago, congestion_3hr_ago
   (these need each route's own timestamp-sorted history, so we group by route)
5. Saves the result as data/processed/traffic_data_features.csv

Run it with:
    python src/feature_engineering.py
(assuming you're in the project's root folder, with venv activated)

NOTE: if your data has GAPS in time (e.g. only 2 real hourly readings so far,
or synthetic data with random missing hours), lag features for the very first
few rows of each route will be blank (NaN) - that's expected and correct,
since there's no "1 hour ago" for the very first reading. Later modeling
steps (Prophet/ARIMA) will handle this automatically.
"""

import os
import pandas as pd

INPUT_FILE = os.path.join("data", "processed", "traffic_data_clean.csv")
OUTPUT_FILE = os.path.join("data", "processed", "traffic_data_features.csv")


def load_clean_data():
    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(
            f"Clean data file not found at {INPUT_FILE}. "
            "Make sure preprocess_data.py has been run first."
        )
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} cleaned rows from {INPUT_FILE}")
    return df


def interpolate_missing_values(df):
    """
    Fills small gaps in congestion_index (and related time columns) using
    linear interpolation, done SEPARATELY per route - since Route 1's
    congestion has nothing to do with Route 5's congestion, we must not
    let interpolation blend data across different routes.
    """
    df = df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    cols_to_interpolate = [
        "travel_time_with_traffic_sec", "free_flow_time_sec", "congestion_index"
    ]

    before_missing = df["congestion_index"].isna().sum()

    df[cols_to_interpolate] = (
        df.groupby("route_id")[cols_to_interpolate]
        .transform(lambda group: group.interpolate(method="linear", limit_direction="both"))
    )

    after_missing = df["congestion_index"].isna().sum()
    print(f"Interpolated missing congestion_index values: "
          f"{before_missing} -> {after_missing} still missing "
          f"(only possible if a whole route has zero valid readings)")

    return df


def add_calendar_features(df):
    """Adds simple calendar-based columns that help models and charts group data meaningfully."""
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # Monday=0 ... Sunday=6
    df["is_weekend"] = df["day_of_week"] >= 5           # Saturday(5) or Sunday(6)
    df["is_friday"] = df["day_of_week"] == 4             # Friday, relevant for Jummah traffic dip
    print("Added calendar features: hour_of_day, day_of_week, is_weekend, is_friday")
    return df


def add_lag_features(df):
    """
    Adds 'what was congestion X hours ago' columns, calculated separately
    per route (using each route's own timestamp-sorted history).
    This is the core idea from literature review point 2 (autocorrelation) -
    recent history helps predict the next value.
    """
    df = df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    df["congestion_1hr_ago"] = df.groupby("route_id")["congestion_index"].shift(1)
    df["congestion_3hr_ago"] = df.groupby("route_id")["congestion_index"].shift(3)

    print("Added lag features: congestion_1hr_ago, congestion_3hr_ago")
    return df


def save_features(df):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows with engineered features to {OUTPUT_FILE}")


def run_feature_engineering():
    df = load_clean_data()
    df = interpolate_missing_values(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    save_features(df)


if __name__ == "__main__":
    run_feature_engineering()
