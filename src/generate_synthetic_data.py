"""
Karachi Traffic Congestion Forecasting - Synthetic Data Generator

WHY THIS EXISTS:
Your project plan's own risk register says: "supplement with synthetic/sample
data for initial development only" in case real historical data isn't ready
yet. This script creates a REALISTIC 6-month hourly dataset so you can start
Week 3 (EDA + feature engineering) and Weeks 4-6 (Prophet/ARIMA modeling)
immediately, without waiting for real data to accumulate.

IMPORTANT: every row is tagged is_synthetic=True. When your real Task
Scheduler data is ready, you can filter this out (or blend it) -  never
present synthetic rows as real collected data in your final report.

WHAT PATTERNS ARE SIMULATED (so it behaves like real Karachi traffic):
1. Daily rush hour peaks (8-10am, 5-8pm) - matches literature review point 1
2. Weekday vs weekend differences, and a Friday midday dip (Jummah prayers)
3. Rain increases congestion (weather -> traffic link from literature review point 3)
4. Random daily "incidents" (accidents, breakdowns) causing occasional spikes
5. Small random noise so it's not perfectly predictable (real traffic never is)
6. A few % of rows randomly marked as missing, to mimic real API failures -
   this lets you test your preprocessing pipeline's missing-data handling
   on something realistic before real gaps show up.

Run it with:
    python src/generate_synthetic_data.py
Output: data/raw/traffic_data_synthetic.csv
"""

import os
import numpy as np
import pandas as pd

ROUTES_FILE = os.path.join("data", "raw", "routes.csv")
OUTPUT_FILE = os.path.join("data", "raw", "traffic_data_synthetic.csv")

# 6 months of hourly data, ending "today" - adjust START_DATE if you want a
# different window. Using Feb-Jul 2026 so the last month (July) naturally
# lands in Karachi's monsoon season, giving us real rain-vs-congestion signal.
START_DATE = "2026-02-01"
END_DATE = "2026-07-31 23:00:00"

# Assumed free-flow driving speed (km/h) used to calculate each route's
# baseline free-flow travel time from its distance_km column.
FREE_FLOW_SPEED_KMPH = 28

np.random.seed(42)  # fixed seed = reproducible dataset every time you run this


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculates straight-line ("as the crow flies") distance in km between
    two lat/lon points using the Haversine formula. Real driving distance
    will be longer than this (roads aren't straight lines), so we apply a
    simple road-detour correction factor afterwards.
    """
    R = 6371.0  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# Straight-line distance underestimates real road distance (roads curve,
# have intersections, etc). 1.3x is a common rough correction factor for
# urban road networks.
ROAD_DETOUR_FACTOR = 1.3


def load_routes():
    df = pd.read_csv(ROUTES_FILE)

    # distance_km isn't in the CSV - calculate it from the lat/lon columns
    df["distance_km"] = haversine_km(
        df["origin_lat"], df["origin_lon"],
        df["destination_lat"], df["destination_lon"]
    ) * ROAD_DETOUR_FACTOR

    # corridor also isn't in the CSV - just reuse route_name as a simple
    # grouping label for now. Replace with real corridor groupings later
    # if multiple routes share the same corridor.
    if "corridor" not in df.columns:
        df["corridor"] = df["route_name"]

    return df


def daily_congestion_curve(hour, is_friday, is_weekend):
    """
    Returns a congestion multiplier addition (0 = no extra congestion,
    higher = more congestion) based on hour of day and day type.
    This encodes the "rush hour" pattern from our literature review.
    """
    if is_weekend:
        # Weekends: much lighter traffic, mild afternoon bump (shopping/leisure)
        base = 0.05
        if 16 <= hour <= 20:
            base += 0.15
        return base

    # Weekday morning rush (8-10am)
    """
   Weekday Commute (The Bell Curve Math)On normal weekdays,
   traffic builds up, peaks, and clears using Gaussian Bell Curves. 
   Instead of traffic instantly jumping from low to high, the function uses a smooth mathematical curve:
   $$\text{Spike Intensity} = A \cdot \exp\left(-\frac{(\text{hour} - \mu)^2}{2\sigma^2}\right)$$$\mu$ (Center):
   
   The hour when traffic peaks.$\sigma$ (Spread): 
   How wide/long the rush hour lasts.$A$ (Peak Height): 
   The maximum traffic added at the peak.
   The code combines three layers:
   Baseline (0.05): Standard late-night/midday light traffic.
   Morning Rush (morning): Peaks at 9:00 AM ($\mu = 9$) with an extra weight of 0.55.
   Evening Rush (evening): Peaks at 6:30 PM ($\mu = 18.5$) with a heavier weight of 0.75 (evening traffic is modeled as wider and worse than morning).
    """
    morning = np.exp(-((hour - 9) ** 2) / (2 * 1.2 ** 2)) * 0.55
    # Weekday evening rush (5-8pm), generally the worst congestion of the day
    evening = np.exp(-((hour - 18.5) ** 2) / (2 * 1.5 ** 2)) * 0.75
    # Quiet overnight baseline
    base = 0.05

    total = base + morning + evening

    if is_friday and 12 <= hour <= 14:
        # Jummah prayer effect: sharp midday dip as many people are at mosque,
        # roads briefly quieter than a normal weekday lunch hour
        total *= 0.5

    return total


def generate_weather(timestamps):
    """
    Generates a plausible daily weather series for Karachi:
    - Hot Feb-Jun, monsoon rain concentrated in Jul (with some Jun rain)
    - Temperature has a realistic seasonal curve + daily noise
    """
    weather_rows = []
    for ts in timestamps:
        month = ts.month
        # Rough seasonal temperature baseline by month (Karachi climate)
        seasonal_temp = {2: 24, 3: 28, 4: 32, 5: 34, 6: 33, 7: 30}.get(month, 28)
        temp = seasonal_temp + np.random.normal(0, 2.5)

        # Monsoon-heavy months: July (high chance), June (some chance)
        rain_chance = {2: 0.01, 3: 0.02, 4: 0.02, 5: 0.03, 6: 0.10, 7: 0.30}.get(month, 0.02)
        is_raining = np.random.random() < rain_chance
        precip = round(np.random.exponential(4.0), 1) if is_raining else 0.0
        weather_code = 61 if precip > 0 else 1  # 61=rain-ish, 1=mainly clear (Open-Meteo style codes)

        weather_rows.append({
            "timestamp": ts,
            "temperature_c": round(temp, 1),
            "precipitation_mm": precip,
            "weather_code": weather_code,
        })
    return pd.DataFrame(weather_rows)


def generate_dataset():
    routes = load_routes()
    timestamps = pd.date_range(start=START_DATE, end=END_DATE, freq="h")
    print(f"Generating data for {len(routes)} routes x {len(timestamps)} hours "
          f"= {len(routes) * len(timestamps)} rows...")

    weather_df = generate_weather(timestamps)
    weather_lookup = weather_df.set_index("timestamp")

    all_rows = []

    for _, route in routes.iterrows():
        distance_km = float(route["distance_km"])
        free_flow_sec = int((distance_km / FREE_FLOW_SPEED_KMPH) * 3600)

        for ts in timestamps:
            hour = ts.hour
            is_weekend = ts.dayofweek >= 5  # Saturday=5, Sunday=6
            is_friday = ts.dayofweek == 4

            congestion_add = daily_congestion_curve(hour, is_friday, is_weekend)

            # Weather effect: rain increases congestion further
            precip = weather_lookup.loc[ts, "precipitation_mm"]
            weather_add = min(precip, 20) / 20 * 0.35

            # Random noise: real traffic is never perfectly smooth
            noise = np.random.normal(0, 0.05)

            # Rare random incident (e.g. accident, breakdown): ~1% chance per hour per route
            incident_add = np.random.choice([0, np.random.uniform(0.4, 1.0)], p=[0.99, 0.01])

            congestion_index = max(1.0, 1.0 + congestion_add + weather_add + noise + incident_add)
            traffic_sec = int(free_flow_sec * congestion_index)

            all_rows.append({
                "timestamp": ts,
                "route_id": route["route_id"],
                "route_name": route["route_name"],
                "corridor": route["corridor"],
                "travel_time_with_traffic_sec": traffic_sec,
                "free_flow_time_sec": free_flow_sec,
                "congestion_index": round(congestion_index, 3),
                "temperature_c": weather_lookup.loc[ts, "temperature_c"],
                "precipitation_mm": precip,
                "weather_code": int(weather_lookup.loc[ts, "weather_code"]),
                "is_synthetic": True,
            })

    df = pd.DataFrame(all_rows)

    # Simulate ~2% missing data, like real API failures would cause
    missing_mask = np.random.random(len(df)) < 0.02
    cols_to_blank = ["travel_time_with_traffic_sec", "free_flow_time_sec", "congestion_index"]
    df.loc[missing_mask, cols_to_blank] = np.nan

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
    print(f"  ({missing_mask.sum()} rows have simulated missing values, ~2%)")


if __name__ == "__main__":
    generate_dataset()