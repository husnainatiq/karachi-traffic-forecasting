"""
Karachi Traffic Congestion Forecasting - Prophet Modeling (Week 4)

What this script does, step by step:
1. Reads the feature-engineered dataset (data/processed/traffic_data_features.csv)
2. For EACH route separately (Prophet trains one model per time series):
   a. Reshapes the data into Prophet's required format: columns 'ds' (datetime)
      and 'y' (the value to predict - here, congestion_index)
   b. Splits the data into TRAIN (everything except the last 14 days) and
      TEST (the last 14 days) - the test set is what we compare
      "actual vs predicted" against, since Prophet never sees it during training
   c. Configures Prophet with daily + weekly seasonality (matches our literature
      review: rush-hour daily cycle + weekday/weekend/Friday weekly cycle),
      and adds precipitation as an extra regressor (rain -> more congestion)
   d. Trains the model, forecasts over the test period
   e. Saves: the trained model, an actual-vs-predicted plot, and a quick
      MAE (Mean Absolute Error) for a first sanity check
      (full MAE/RMSE/MAPE comparison across Prophet vs ARIMA happens in Week 6)
3. Prints a summary table of all 10 routes' quick MAE at the end

Run it with:
    python src/prophet_modeling.py
(assuming you're in the project's root folder, with venv activated)

NOTE: this can take a few minutes to run since it trains 10 separate models.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import pickle
import warnings
warnings.filterwarnings("ignore")  # Prophet is chatty with harmless warnings

INPUT_FILE = os.path.join("data", "processed", "traffic_data_features.csv")
MODELS_DIR = os.path.join("models", "prophet")
REPORTS_DIR = os.path.join("reports", "prophet")

TEST_DAYS = 14  # hold out the last 14 days as the test set for actual-vs-predicted


def load_data():
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows across {df['route_id'].nunique()} routes")
    return df


def prepare_route_data(df, route_id):
    """
    Filters to one route and reshapes into Prophet's required format:
    - 'ds' column: the timestamp
    - 'y' column: the value to forecast (congestion_index)
    - 'precipitation_mm' kept as an extra regressor column
    """
    route_df = df[df["route_id"] == route_id].sort_values("timestamp").copy()
    route_df = route_df.rename(columns={"timestamp": "ds", "congestion_index": "y"})
    route_df = route_df[["ds", "y", "precipitation_mm"]].dropna(subset=["y"])
    return route_df


def split_train_test(route_df):
    """Splits into train (everything except last TEST_DAYS) and test (last TEST_DAYS)."""
    cutoff = route_df["ds"].max() - pd.Timedelta(days=TEST_DAYS)
    train = route_df[route_df["ds"] <= cutoff]
    test = route_df[route_df["ds"] > cutoff]
    return train, test


def train_prophet_model(train_df):
    """
    Configures and trains a Prophet model for one route.
    - daily_seasonality: captures the rush-hour pattern within each day
    - weekly_seasonality: captures weekday vs weekend vs Friday patterns
    - yearly_seasonality=False: with only ~6 months of data, we don't have
      enough history yet to reliably learn a yearly cycle - revisit once
      real data accumulates past 1 year
    - precipitation_mm added as an extra regressor: lets the model directly
      use "is it raining" as a predictive signal, not just time patterns
    """
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        seasonality_mode="multiplicative",  # congestion swings scale with the base level, not a flat +/- amount
    )
    model.add_regressor("precipitation_mm")
    model.fit(train_df)
    return model


def forecast_test_period(model, test_df):
    """
    Uses the trained model to predict over the test period.
    We must supply precipitation_mm for the test dates too, since it's a
    regressor the model expects - we use the ACTUAL historical precipitation
    here (since this is backtesting on real past data, not future forecasting).
    """
    future = test_df[["ds", "precipitation_mm"]].copy()
    forecast = model.predict(future)
    return forecast


def calculate_quick_mae(actual, predicted):
    """A quick Mean Absolute Error for a first sanity check of model quality."""
    return round(float(np.mean(np.abs(actual - predicted))), 4)


def plot_actual_vs_predicted(route_id, route_name, train_df, test_df, forecast):
    """Saves a chart comparing actual congestion vs Prophet's predicted congestion over the test period."""
    plt.figure(figsize=(14, 5))

    # Show a bit of recent training history for context
    recent_train = train_df.tail(24 * 7)  # last 7 days of training data
    plt.plot(recent_train["ds"], recent_train["y"], label="Actual (training history)", color="gray", alpha=0.5)

    plt.plot(test_df["ds"], test_df["y"], label="Actual (test period)", color="#4C72B0", linewidth=2)
    plt.plot(forecast["ds"], forecast["yhat"], label="Predicted (Prophet)", color="#C44E52", linewidth=2)
    plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                      color="#C44E52", alpha=0.15, label="Prediction interval")

    plt.axvline(test_df["ds"].min(), color="black", linestyle="--", alpha=0.5, label="Train/Test split")
    plt.xlabel("Date")
    plt.ylabel("Congestion Index")
    plt.title(f"Route {route_id} ({route_name}) - Actual vs Predicted (Prophet)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    filepath = os.path.join(REPORTS_DIR, f"route_{route_id}_actual_vs_predicted.png")
    plt.savefig(filepath, dpi=100)
    plt.close()
    return filepath


def run_prophet_for_all_routes():
    df = load_data()
    os.makedirs(MODELS_DIR, exist_ok=True)

    route_info = df[["route_id", "route_name"]].drop_duplicates().sort_values("route_id")
    results = []

    for _, row in route_info.iterrows():
        route_id, route_name = row["route_id"], row["route_name"]
        print(f"\nTraining Prophet model for Route {route_id}: {route_name}...")

        route_df = prepare_route_data(df, route_id)
        train_df, test_df = split_train_test(route_df)

        if len(train_df) < 24 * 30 or len(test_df) == 0:
            print(f"  Skipping - not enough data yet (need at least ~30 days train + some test data)")
            continue

        model = train_prophet_model(train_df)
        forecast = forecast_test_period(model, test_df)

        mae = calculate_quick_mae(test_df["y"].values, forecast["yhat"].values)
        print(f"  -> Quick MAE on test period: {mae}")

        plot_path = plot_actual_vs_predicted(route_id, route_name, train_df, test_df, forecast)
        print(f"  -> Saved plot: {plot_path}")

        model_path = os.path.join(MODELS_DIR, f"route_{route_id}_prophet.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  -> Saved model: {model_path}")

        results.append({"route_id": route_id, "route_name": route_name, "quick_mae": mae})

    print("\n=== Summary: Quick MAE per route (Prophet baseline) ===")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(REPORTS_DIR, "prophet_baseline_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    run_prophet_for_all_routes()
