"""
Karachi Traffic Congestion Forecasting - ARIMA/SARIMA Modeling (Week 5)

What this script does, step by step:
1. Reads the feature-engineered dataset (data/processed/traffic_data_features.csv)
2. For EACH route separately:
   a. Builds a regular hourly time series of congestion_index (ARIMA requires
      evenly-spaced time steps, unlike Prophet which tolerates gaps)
   b. Runs the Augmented Dickey-Fuller (ADF) test to check STATIONARITY -
      tells us whether the series' average level drifts over time or holds
      steady, which guides how much differencing ARIMA needs
   c. Uses the last 45 days of TRAINING data (not the full 6 months) - this
      is a deliberate choice: ARIMA/SARIMAX with a 24-hour seasonal period
      is computationally expensive, and older data is less relevant for
      near-term forecasting anyway. Prophet (Week 4) used the full history
      since it handles that more efficiently - this difference is worth
      noting in your model comparison write-up.
   d. Fits a small, sensible SET of candidate SARIMA(p,d,q)(P,D,Q,24) models
      and picks the one with the lowest AIC (Akaike Information Criterion -
      a standard statistical score balancing model fit vs. complexity).
      This is the "hyperparameter tuning" step for this script.
   e. Forecasts over the same 14-day test period used in Week 4's Prophet
      script, calculates quick MAE, and saves an actual-vs-predicted plot
3. Combines ARIMA's results with the Week 4 Prophet summary into one
   side-by-side comparison table - your plan's "Comparative Model Benchmark."


NOTE: each route takes roughly 20-30 seconds (fitting 3 candidate models),
so the full 10-route run takes about 3-5 minutes - much faster than a full
automated grid search, while still being a defensible, documented tuning process.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import warnings
warnings.filterwarnings("ignore") 

INPUT_FILE = os.path.join("data", "processed", "traffic_data_features.csv")
MODELS_DIR = os.path.join("models", "arima")
REPORTS_DIR = os.path.join("reports", "arima")
PROPHET_SUMMARY_FILE = os.path.join("reports", "prophet", "prophet_baseline_summary.csv")

TEST_DAYS = 14
TRAIN_WINDOW_DAYS = 45  # recent training window used for ARIMA (see note above)
SEASONAL_PERIOD = 24     # 24 hours = 1 day, matches our confirmed daily rush-hour cycle 

# A small, sensible set of candidate (order, seasonal_order) combinations.
# Covers a light AR-heavy option, a differencing-heavy option, and a
# MA-heavy option - a reasonable, explainable spread rather than an
# exhaustive (and much slower) search.
CANDIDATE_MODELS = [
    ((1, 0, 1), (1, 1, 1, SEASONAL_PERIOD)),
    ((2, 1, 1), (1, 1, 0, SEASONAL_PERIOD)),
    ((1, 1, 2), (0, 1, 1, SEASONAL_PERIOD)),
]


def load_data():
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows across {df['route_id'].nunique()} routes")
    return df


def build_hourly_series(df, route_id):
    """Builds a clean, regularly-spaced hourly time series for one route (no gaps)."""
    route_df = df[df["route_id"] == route_id].sort_values("timestamp")
    series = route_df.set_index("timestamp")["congestion_index"]

    full_range = pd.date_range(start=series.index.min(), end=series.index.max(), freq="h")
    series = series.reindex(full_range)
    series = series.interpolate(method="linear", limit_direction="both")
    series.index.name = "timestamp"
    return series


def check_stationarity(series):
    """Augmented Dickey-Fuller test: low p-value (<0.05) suggests the series is stationary."""
    result = adfuller(series.dropna())
    p_value = result[1]
    is_stationary = p_value < 0.05
    print(f"  ADF test p-value: {p_value:.4f} -> "
          f"{'Stationary' if is_stationary else 'Non-stationary (SARIMA differencing handles this)'}")
    return is_stationary


def split_train_test(series):
    cutoff = series.index.max() - pd.Timedelta(days=TEST_DAYS)
    train_full = series[series.index <= cutoff]
    train_recent = train_full.tail(24 * TRAIN_WINDOW_DAYS)  # only last 45 days used for fitting
    test = series[series.index > cutoff]
    return train_recent, test


def fit_best_candidate(train_series):
    """Fits each candidate SARIMA model and keeps the one with the lowest AIC."""
    best_aic = np.inf
    best_fit = None
    best_params = None

    for order, seasonal_order in CANDIDATE_MODELS:
        try:
            model = SARIMAX(
                train_series, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            print(f"    order={order} seasonal_order={seasonal_order} -> AIC={fit.aic:.1f}")
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
                best_params = (order, seasonal_order)
        except Exception as e:
            print(f"    order={order} seasonal_order={seasonal_order} -> FAILED ({e})")

    return best_fit, best_params, best_aic


def calculate_quick_mae(actual, predicted):
    return round(float(np.mean(np.abs(actual - predicted))), 4)


def plot_actual_vs_predicted(route_id, route_name, train, test, predicted_mean, conf_int):
    plt.figure(figsize=(14, 5))

    plt.plot(train.index, train.values, label="Actual (training history, last 45 days)", color="gray", alpha=0.5)
    plt.plot(test.index, test.values, label="Actual (test period)", color="#4C72B0", linewidth=2)
    plt.plot(test.index, predicted_mean.values, label="Predicted (ARIMA)", color="#55A868", linewidth=2)
    plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                      color="#55A868", alpha=0.15, label="Confidence interval")

    plt.axvline(test.index.min(), color="black", linestyle="--", alpha=0.5, label="Train/Test split")
    plt.xlabel("Date")
    plt.ylabel("Congestion Index")
    plt.title(f"Route {route_id} ({route_name}) - Actual vs Predicted (ARIMA)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    filepath = os.path.join(REPORTS_DIR, f"route_{route_id}_actual_vs_predicted.png")
    plt.savefig(filepath, dpi=100)
    plt.close()
    return filepath


def run_arima_for_all_routes():
    df = load_data()
    os.makedirs(MODELS_DIR, exist_ok=True)

    route_info = df[["route_id", "route_name"]].drop_duplicates().sort_values("route_id")
    results = []

    for _, row in route_info.iterrows():
        route_id, route_name = row["route_id"], row["route_name"]
        print(f"\nProcessing Route {route_id}: {route_name}...")

        series = build_hourly_series(df, route_id)

        if len(series) < 24 * (TRAIN_WINDOW_DAYS + TEST_DAYS):
            print(f"  Skipping - not enough data yet")
            continue

        train, test = split_train_test(series)
        check_stationarity(train)

        print("  Fitting candidate SARIMA models...")
        best_fit, best_params, best_aic = fit_best_candidate(train)

        if best_fit is None:
            print(f"  All candidates failed for this route - skipping")
            continue

        order, seasonal_order = best_params
        print(f"  -> Best model: order={order}, seasonal_order={seasonal_order}, AIC={best_aic:.1f}")

        forecast = best_fit.get_forecast(steps=len(test))
        predicted_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        mae = calculate_quick_mae(test.values, predicted_mean.values)
        print(f"  -> Quick MAE on test period: {mae}")

        plot_path = plot_actual_vs_predicted(route_id, route_name, train, test, predicted_mean, conf_int)
        print(f"  -> Saved plot: {plot_path}")

        model_path = os.path.join(MODELS_DIR, f"route_{route_id}_arima.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_fit, f)
        print(f"  -> Saved model: {model_path}")

        results.append({
            "route_id": route_id,
            "route_name": route_name,
            "arima_order": str(order),
            "seasonal_order": str(seasonal_order),
            "arima_quick_mae": mae,
        })

    arima_summary = pd.DataFrame(results)
    print("\n=== Summary: ARIMA parameters + Quick MAE per route ===")
    print(arima_summary.to_string(index=False))

    arima_summary_path = os.path.join(REPORTS_DIR, "arima_baseline_summary.csv")
    arima_summary.to_csv(arima_summary_path, index=False)
    print(f"\nSaved ARIMA summary to {arima_summary_path}")

    combine_with_prophet(arima_summary)


def combine_with_prophet(arima_summary):
    """Merges ARIMA results with Week 4's Prophet summary - the Comparative Model Benchmark deliverable."""
    if not os.path.isfile(PROPHET_SUMMARY_FILE):
        print(f"\nNote: Prophet summary not found at {PROPHET_SUMMARY_FILE} - "
              f"run prophet_modeling.py first to get the full comparison table.")
        return

    prophet_summary = pd.read_csv(PROPHET_SUMMARY_FILE)
    prophet_summary = prophet_summary.rename(columns={"quick_mae": "prophet_quick_mae"})

    comparison = prophet_summary.merge(
        arima_summary[["route_id", "arima_order", "seasonal_order", "arima_quick_mae"]],
        on="route_id", how="outer"
    )
    comparison["better_model"] = np.where(
        comparison["prophet_quick_mae"] < comparison["arima_quick_mae"], "Prophet", "ARIMA"
    )

    print("\n=== Comparative Model Benchmark: Prophet vs ARIMA ===")
    print(comparison.to_string(index=False))

    comparison_path = os.path.join("reports", "model_comparison_week5.csv")
    comparison.to_csv(comparison_path, index=False)
    print(f"\nSaved full comparison to {comparison_path}")


if __name__ == "__main__":
    run_arima_for_all_routes()
