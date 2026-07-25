"""
Karachi Traffic Congestion Forecasting - Data Collection Script (Week 1)

What this script does, step by step:
1. Reads the list of routes from data/raw/routes.csv
2. For each route, calls the TomTom Routing API twice:
   - once with live traffic (traffic=true)
   - once without traffic, i.e. free-flow (traffic=false)
3. Calculates a congestion_index = travel_time_with_traffic / free_flow_time
4. Saves one row per route into data/raw/traffic_data.csv, with a timestamp

Run it with:
    python src/collect_traffic_data.py
(assuming you're in the project's root folder, with venv activated)
"""

import os
import csv
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv

# Step 1: Set up logging - this writes messages to BOTH the console (so you see
# it live if running manually) AND a log file (so Task Scheduler runs, which
# happen silently in the background, still leave a record you can check later).
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "collection_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),  # this is what prints to the console
    ],
)
logger = logging.getLogger(__name__)

# Step 2: Load your TomTom API key from the .env file
load_dotenv()
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

if not TOMTOM_API_KEY:
    raise ValueError(
        "TOMTOM_API_KEY not found. Make sure your .env file has a line like:\n"
        "TOMTOM_API_KEY=your_actual_key_here"
    )

# File paths (adjust if your folder names differ)
ROUTES_FILE = os.path.join("data", "raw", "routes.csv")
OUTPUT_FILE = os.path.join("data", "raw", "traffic_data.csv")

# One central Karachi coordinate, used for the city-wide weather reading
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011


def get_current_weather():
    """
    Calls Open-Meteo (no API key needed) to get current weather for Karachi.
    Returns a dictionary with temperature, rain, and weather_code,
    or a dictionary of None values if the call fails.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "current": "temperature_2m,precipitation,weather_code",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data["current"]
        return {
            "temperature_c": current["temperature_2m"],
            "precipitation_mm": current["precipitation"],
            "weather_code": current["weather_code"],
        }
    except Exception as e:
        logger.error(f"Failed to fetch weather from Open-Meteo: {e}")
        return {
            "temperature_c": None,
            "precipitation_mm": None,
            "weather_code": None,
        }


def get_travel_time(origin_lat, origin_lon, dest_lat, dest_lon, use_traffic):
    """
    Calls the TomTom Routing API for one origin -> destination pair.
    use_traffic=True  -> returns current travel time with live traffic
    use_traffic=False -> returns free-flow travel time (no traffic)
    Returns travel time in seconds, or None if the call failed.
    """
    url = (
        f"https://api.tomtom.com/routing/1/calculateRoute/"
        f"{origin_lat},{origin_lon}:{dest_lat},{dest_lon}/json"
    )
    params = {
        "key": TOMTOM_API_KEY,
        "traffic": "true" if use_traffic else "false",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # raises an error if the request failed
        data = response.json()

        # The travel time (in seconds) is nested inside the response like this:
        travel_time_seconds = data["routes"][0]["summary"]["travelTimeInSeconds"]
        return travel_time_seconds
    except Exception as e:
        logger.error(f"Failed to fetch travel time (traffic={use_traffic}) "
                     f"for {origin_lat},{origin_lon} -> {dest_lat},{dest_lon}: {e}")
        return None


def load_routes():
    """Reads routes.csv and returns a list of route dictionaries."""
    routes = []
    try:
        with open(ROUTES_FILE, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                routes.append(row)
    except Exception as e:
        logger.error(f"Could not read routes file at {ROUTES_FILE}: {e}")
        raise
    return routes


def collect_data_for_all_routes():
    """Main function: loops through every route, collects data, saves results."""
    logger.info("=== Starting data collection run ===")
    routes = load_routes()
    timestamp = datetime.now().isoformat()

    logger.info("Fetching current Karachi weather...")
    weather = get_current_weather()
    logger.info(f"Weather -> temperature: {weather['temperature_c']}C | "
                f"precipitation: {weather['precipitation_mm']}mm | "
                f"weather_code: {weather['weather_code']}")

    results = []

    for route in routes:
        try:
            logger.info(f"Processing Route {route['route_id']}: {route['route_name']}...")

            traffic_time = get_travel_time(
                route["origin_lat"], route["origin_lon"],
                route["destination_lat"], route["destination_lon"],
                use_traffic=True,
            )

            free_flow_time = get_travel_time(
                route["origin_lat"], route["origin_lon"],
                route["destination_lat"], route["destination_lon"],
                use_traffic=False,
            )

            # Only calculate congestion_index if both calls succeeded
            if traffic_time is not None and free_flow_time is not None and free_flow_time > 0:
                congestion_index = round(traffic_time / free_flow_time, 3)
            else:
                congestion_index = None

            results.append({
                "timestamp": timestamp,
                "route_id": route["route_id"],
                "route_name": route["route_name"],
                "travel_time_with_traffic_sec": traffic_time,
                "free_flow_time_sec": free_flow_time,
                "congestion_index": congestion_index,
                "temperature_c": weather["temperature_c"],
                "precipitation_mm": weather["precipitation_mm"],
                "weather_code": weather["weather_code"],
            })

            logger.info(f"  -> traffic time: {traffic_time}s | free-flow: {free_flow_time}s | "
                        f"congestion index: {congestion_index}")

        except Exception as e:
            # If ANYTHING unexpected goes wrong for this one route, log it and
            # move on to the next route instead of crashing the whole run.
            logger.error(f"Unexpected error processing Route {route.get('route_id')}: {e}")
            continue

    if results:
        save_results(results)
    else:
        logger.warning("No results collected this run - nothing saved.")

    logger.info("=== Data collection run finished ===\n")


def save_results(results):
    """Appends the results to traffic_data.csv (creates the file with headers if it doesn't exist yet)."""
    file_exists = os.path.isfile(OUTPUT_FILE)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "timestamp", "route_id", "route_name",
            "travel_time_with_traffic_sec", "free_flow_time_sec", "congestion_index",
            "temperature_c", "precipitation_mm", "weather_code",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # only write the header row once

        writer.writerows(results)

    logger.info(f"Saved {len(results)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        collect_data_for_all_routes()
    except Exception as e:
        # Last-resort safety net: if something crashes the entire run
        # (e.g. routes.csv missing), log it clearly instead of failing silently
        # when run automatically by Task Scheduler.
        logger.critical(f"Data collection run FAILED completely: {e}")