"""
Karachi Traffic Congestion Forecasting - Formal Model Evaluation (Week 6)

What this script does, step by step:
1. Loads the already-trained Prophet models (models/prophet/) and
   ARIMA models (models/arima/) from Weeks 4 and 5 - no retraining needed
2. Re-generates predictions for each route's held-out 14-day test period
   (same test period used during training, so this is a fair comparison)
3. Calculates THREE formal error metrics per route, per model:
   - MAE  (Mean Absolute Error): average size of mistake, in congestion_index units
   - RMSE (Root Mean Squared Error): like MAE, but punishes big mistakes harder
   - MAPE (Mean Absolute Percentage Error): average mistake as a % - this is
     the metric your project plan specifically names as a success criterion
4. Selects the BEST model per route (lowest MAPE - ties broken by RMSE)
5. Saves one final comparison table: reports/model_evaluation_final.csv
   This is your Week 6 "Model Selection Report" deliverable.

Run it with:
    python src/evaluate_models.py
(assuming you're in the project's root folder, with venv activated,
and prophet_modeling.py + arima_modeling.py have already been run once)
"""

import os
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE = os.path.join("data", "processed", "traffic_data_features.csv")
PROPHET_MODELS_DIR = os.path.join("models", "prophet")
ARIMA_MODELS_DIR = os.path.join("models", "arima")
REPORTS_DIR = "reports"

TEST_DAYS = 14


def load_data():
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    return df


def get_prophet_test_predictions(route_id, df):
    """Loads a route's trained Prophet model and predicts over its 14-day test period."""
    model_path = os.path.join(PROPHET_MODELS_DIR, f"route_{route_id}_prophet.pkl")
    if not os.path.isfile(model_path):
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    route_df = df[df["route_id"] == route_id].sort_values("timestamp").copy()
    route_df = route_df.rename(columns={"timestamp": "ds", "congestion_index": "y"})
    route_df = route_df[["ds", "y", "precipitation_mm"]].dropna(subset=["y"])

    cutoff = route_df["ds"].max() - pd.Timedelta(days=TEST_DAYS)
    test_df = route_df[route_df["ds"] > cutoff]

    future = test_df[["ds", "precipitation_mm"]].copy()
    forecast = model.predict(future)

    return test_df["y"].values, forecast["yhat"].values


def get_arima_test_predictions(route_id, df):
    """Loads a route's trained ARIMA model and predicts over its 14-day test period."""
    model_path = os.path.join(ARIMA_MODELS_DIR, f"route_{route_id}_arima.pkl")
    if not os.path.isfile(model_path):
        return None

    with open(model_path, "rb") as f:
        fitted_model = pickle.load(f)

    route_df = df[df["route_id"] == route_id].sort_values("timestamp")
    series = route_df.set_index("timestamp")["congestion_index"]
    full_range = pd.date_range(start=series.index.min(), end=series.index.max(), freq="h")
    series = series.reindex(full_range).interpolate(method="linear", limit_direction="both")

    cutoff = series.index.max() - pd.Timedelta(days=TEST_DAYS)
    test = series[series.index > cutoff]

    forecast = fitted_model.get_forecast(steps=len(test))
    predicted_mean = forecast.predicted_mean.values

    return test.values, predicted_mean


def calculate_metrics(actual, predicted):
    """Calculates MAE, RMSE, and MAPE - the three standard forecasting error metrics."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100  # as a percentage

    return round(mae, 4), round(rmse, 4), round(mape, 2)


def run_evaluation():
    df = load_data()
    route_info = df[["route_id", "route_name"]].drop_duplicates().sort_values("route_id")

    results = []

    for _, row in route_info.iterrows():
        route_id, route_name = row["route_id"], row["route_name"]
        print(f"Evaluating Route {route_id}: {route_name}...")

        prophet_result = get_prophet_test_predictions(route_id, df)
        arima_result = get_arima_test_predictions(route_id, df)

        row_result = {"route_id": route_id, "route_name": route_name}

        if prophet_result is not None:
            actual, predicted = prophet_result
            mae, rmse, mape = calculate_metrics(actual, predicted)
            row_result.update({
                "prophet_mae": mae, "prophet_rmse": rmse, "prophet_mape": mape,
            })
            print(f"  Prophet -> MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%")
        else:
            row_result.update({"prophet_mae": None, "prophet_rmse": None, "prophet_mape": None})
            print("  Prophet model not found - skipping")

        if arima_result is not None:
            actual, predicted = arima_result
            mae, rmse, mape = calculate_metrics(actual, predicted)
            row_result.update({
                "arima_mae": mae, "arima_rmse": rmse, "arima_mape": mape,
            })
            print(f"  ARIMA   -> MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%")
        else:
            row_result.update({"arima_mae": None, "arima_rmse": None, "arima_mape": None})
            print("  ARIMA model not found - skipping")

        # Select the best model per route using MAPE as the primary metric
        # (MAPE is the metric your project plan names as a success criterion),
        # with RMSE as a tie-breaker if MAPE values are very close.
        if row_result["prophet_mape"] is not None and row_result["arima_mape"] is not None:
            if row_result["prophet_mape"] < row_result["arima_mape"]:
                row_result["best_model"] = "Prophet"
            elif row_result["arima_mape"] < row_result["prophet_mape"]:
                row_result["best_model"] = "ARIMA"
            else:
                row_result["best_model"] = "Prophet" if row_result["prophet_rmse"] <= row_result["arima_rmse"] else "ARIMA"
        else:
            row_result["best_model"] = "N/A - missing model(s)"

        print(f"  -> Best model for this route: {row_result['best_model']}\n")
        results.append(row_result)

    results_df = pd.DataFrame(results)

    print("=" * 70)
    print("FINAL MODEL EVALUATION - Week 6 Deliverable")
    print("=" * 70)
    print(results_df.to_string(index=False))

    prophet_wins = (results_df["best_model"] == "Prophet").sum()
    arima_wins = (results_df["best_model"] == "ARIMA").sum()
    print(f"\nProphet selected as best model for {prophet_wins} route(s)")
    print(f"ARIMA selected as best model for {arima_wins} route(s)")

    avg_prophet_mape = results_df["prophet_mape"].mean()
    avg_arima_mape = results_df["arima_mape"].mean()
    print(f"\nAverage MAPE across all routes - Prophet: {avg_prophet_mape:.2f}% | ARIMA: {avg_arima_mape:.2f}%")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    output_path = os.path.join(REPORTS_DIR, "model_evaluation_final.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved final evaluation report to {output_path}")


if __name__ == "__main__":
    run_evaluation()
