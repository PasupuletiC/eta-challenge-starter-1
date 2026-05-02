#!/usr/bin/env python
"""Baseline: gradient-boosted trees on six simple features.

Trains in ~5 minutes on a laptop CPU. Produces `model.pkl` which `predict.py`
loads at inference.

Prerequisites:
    python data/download_data.py   # one-time, ~500 MB download

Run:
    python baseline.py             # trains and saves model.pkl

Your job is to replace this file with something better. The grader only cares
about `predict.py` — this file just needs to produce a `model.pkl` that
`predict.py` can load.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"

COORDS_PATH = DATA_DIR / "taxi_zone_coords.csv"
if COORDS_PATH.exists():
    coords_df = pd.read_csv(COORDS_PATH).set_index("LocationID")
    ZONE_LATS = coords_df["lat"].to_dict()
    ZONE_LONS = coords_df["lon"].to_dict()
else:
    ZONE_LATS = {}
    ZONE_LONS = {}

WEATHER_PATH = DATA_DIR / "weather_2023.csv"
if WEATHER_PATH.exists():
    weather_df = pd.read_csv(WEATHER_PATH)
    WEATHER_DICT = weather_df.set_index(["month", "day", "hour"])[["precipitation_mm", "snowfall_mm"]].to_dict("index")
else:
    WEATHER_DICT = {}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


FEATURES = [
    "pickup_zone", "dropoff_zone", "hour", "dow", "month", "passenger_count",
    "is_weekend", "is_rush_hour", "distance_km", "pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon",
    "precipitation_mm", "snowfall_mm"
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw request columns into the model features."""
    ts = pd.to_datetime(df["requested_at"])
    
    pickup_zone = df["pickup_zone"].astype("int32")
    dropoff_zone = df["dropoff_zone"].astype("int32")
    hour = ts.dt.hour.astype("int8")
    dow = ts.dt.dayofweek.astype("int8")
    month = ts.dt.month.astype("int8")
    day = ts.dt.day.astype("int8")
    passenger_count = df["passenger_count"].astype("int8")

    pickup_lat = pickup_zone.map(ZONE_LATS).fillna(0)
    pickup_lon = pickup_zone.map(ZONE_LONS).fillna(0)
    dropoff_lat = dropoff_zone.map(ZONE_LATS).fillna(0)
    dropoff_lon = dropoff_zone.map(ZONE_LONS).fillna(0)
    
    distance_km = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    
    is_weekend = (dow >= 5).astype("int8")
    is_rush_hour = (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))) & (dow < 5)
    is_rush_hour = is_rush_hour.astype("int8")

    precip_list = []
    snow_list = []
    for m, d, h in zip(month, day, hour):
        w = WEATHER_DICT.get((m, d, h), None)
        if w:
            precip_list.append(w["precipitation_mm"])
            snow_list.append(w["snowfall_mm"])
        else:
            precip_list.append(0.0)
            snow_list.append(0.0)
            
    precipitation_mm = np.array(precip_list, dtype=np.float32)
    snowfall_mm = np.array(snow_list, dtype=np.float32)

    return pd.DataFrame({
        "pickup_zone":     pickup_zone,
        "dropoff_zone":    dropoff_zone,
        "hour":            hour,
        "dow":             dow,
        "month":           month,
        "passenger_count": passenger_count,
        "is_weekend":      is_weekend,
        "is_rush_hour":    is_rush_hour,
        "distance_km":     distance_km.astype("float32"),
        "pickup_lat":      pickup_lat.astype("float32"),
        "pickup_lon":      pickup_lon.astype("float32"),
        "dropoff_lat":     dropoff_lat.astype("float32"),
        "dropoff_lon":     dropoff_lon.astype("float32"),
        "precipitation_mm": precipitation_mm,
        "snowfall_mm":      snowfall_mm,
    })[FEATURES]


def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.name}. Run `python data/download_data.py` first."
            )

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    X_train = engineer_features(train)
    y_train = train["duration_seconds"].to_numpy()
    X_dev = engineer_features(dev)
    y_dev = dev["duration_seconds"].to_numpy()

    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=12,
        learning_rate=0.0609,
        subsample=0.6838,
        colsample_bytree=0.6516,
        min_child_weight=7,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train, verbose=False)
    print(f"  trained in {time.time() - t0:.0f}s")

    preds = model.predict(X_dev)
    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"\nDev MAE: {mae:.1f} seconds")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "zone_lats": ZONE_LATS, "zone_lons": ZONE_LONS, "weather_dict": WEATHER_DICT}, f)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
