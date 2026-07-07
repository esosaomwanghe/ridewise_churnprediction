"""Rider-level feature engineering for the RideWise churn model.

Builds one row per rider by aggregating each rider's trip and session
history. This is the feature set the regularized Random Forest in
model.py was trained and validated on (see notebook/modeling.ipynb).
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

LOYALTY_ORDER = {"Bronze": 1, "Silver": 2, "Gold": 3, "Platinum": 4}
CITY_CATEGORIES = ["Cairo", "Lagos", "Nairobi"]  # Cairo is the dropped baseline

FEATURE_COLUMNS = [
    "age", "avg_rating_given", "loyalty_encoded", "tenure_days", "riders_referred",
    "city_Lagos", "city_Nairobi", "total_trips", "avg_fare", "avg_tip", "tip_rate",
    "avg_trip_duration", "days_since_last_trip", "days_active", "avg_trip_distance",
    "peak_hour_ratio", "trip_rate", "activity_decline",
]


def load_raw_data(data_dir: Path = DATA_DIR):
    riders = pd.read_csv(data_dir / "riders_cleaned.csv", parse_dates=["signup_date"])
    sessions = pd.read_csv(data_dir / "sessions_cleaned.csv", parse_dates=["session_time"])
    trips = pd.read_csv(data_dir / "trips_cleaned.csv", parse_dates=["pickup_time", "dropoff_time"])
    return riders, sessions, trips


def add_target(riders: pd.DataFrame) -> pd.DataFrame:
    riders = riders.copy()
    riders["churn"] = (riders["churn_prob"] >= 0.5).astype(int)
    return riders


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _rider_base_features(riders: pd.DataFrame) -> pd.DataFrame:
    riders = riders.copy()

    ref_date = pd.Timestamp.today()
    riders["tenure_days"] = (ref_date - riders["signup_date"]).dt.days
    riders["loyalty_encoded"] = riders["loyalty_status"].map(LOYALTY_ORDER)

    city = pd.Categorical(riders["city"], categories=CITY_CATEGORIES)
    city_dummies = pd.get_dummies(city, prefix="city", drop_first=True)
    city_dummies.index = riders.index

    referral_counts = (
        riders.groupby("referred_by")["user_id"].count()
        .reset_index()
        .rename(columns={"referred_by": "user_id", "user_id": "riders_referred"})
    )
    riders = riders.merge(referral_counts, on="user_id", how="left")
    riders["riders_referred"] = riders["riders_referred"].fillna(0)

    return pd.concat([
        riders[["user_id", "age", "avg_rating_given", "loyalty_encoded",
                "tenure_days", "riders_referred", "churn"]],
        city_dummies,
    ], axis=1)


def _session_features(sessions: pd.DataFrame) -> pd.DataFrame:
    return (
        sessions.groupby("rider_id")
        .agg(
            total_sessions=("session_id", "count"),
            conversion_rate=("converted", "mean"),
            avg_time_on_app=("time_on_app", "mean"),
        )
        .reset_index()
        .rename(columns={"rider_id": "user_id"})
    )


def _trip_features(trips: pd.DataFrame) -> pd.DataFrame:
    trips = trips.copy()

    trips["trip_duration_mins"] = (
        trips["dropoff_time"] - trips["pickup_time"]
    ).dt.total_seconds() / 60

    trips["trip_distance_km"] = haversine(
        trips["pickup_lat"], trips["pickup_lng"],
        trips["dropoff_lat"], trips["dropoff_lng"],
    )

    trips["hour"] = trips["pickup_time"].dt.hour
    trips["is_peak"] = (
        trips["hour"].between(7, 9) | trips["hour"].between(17, 19)
    ).astype(int)

    ref_trip_date = trips["pickup_time"].max()

    trip_agg = trips.groupby("user_id").agg(
        total_trips=("trip_id", "count"),
        avg_fare=("fare", "mean"),
        avg_tip=("tip", "mean"),
        tip_rate=("tip", lambda x: (x > 0).mean()),
        avg_trip_duration=("trip_duration_mins", "mean"),
        days_since_last_trip=("pickup_time", lambda x: (ref_trip_date - x.max()).days),
        days_active=("pickup_time", lambda x: (x.max() - x.min()).days),
        avg_trip_distance=("trip_distance_km", "mean"),
        peak_hour_ratio=("is_peak", "mean"),
    ).reset_index()

    trip_agg["trip_rate"] = trip_agg["total_trips"] / (trip_agg["days_active"] + 1)

    recent = trips[trips["pickup_time"] >= ref_trip_date - pd.Timedelta(days=60)]
    historic = trips[trips["pickup_time"] < ref_trip_date - pd.Timedelta(days=60)]

    recent_agg = recent.groupby("user_id").agg(recent_trips=("trip_id", "count")).reset_index()
    historic_agg = historic.groupby("user_id").agg(historic_trips=("trip_id", "count")).reset_index()

    activity = recent_agg.merge(historic_agg, on="user_id", how="outer").fillna(0)
    activity["activity_decline"] = activity["recent_trips"] / (activity["historic_trips"] + 1)

    return trip_agg.merge(activity[["user_id", "activity_decline"]], on="user_id", how="left")


def build_rider_features(riders: pd.DataFrame, sessions: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """One row per rider: demographic + session + trip aggregates, plus `churn`."""
    rider_features = _rider_base_features(riders)

    session_agg = _session_features(sessions)
    rider_features = rider_features.merge(session_agg, on="user_id", how="left")
    rider_features[["total_sessions", "conversion_rate", "avg_time_on_app"]] = (
        rider_features[["total_sessions", "conversion_rate", "avg_time_on_app"]].fillna(0)
    )

    trip_agg = _trip_features(trips)
    rider_features = rider_features.merge(trip_agg, on="user_id", how="left")

    return rider_features


def align_features(rider_features: pd.DataFrame) -> pd.DataFrame:
    """Reindex to the exact training-time column set/order, filling anything missing with 0.

    Needed at inference time since a scoring batch may not contain every
    city, or a rider may have no trips/sessions.
    """
    return rider_features.reindex(columns=FEATURE_COLUMNS, fill_value=0)


def get_X_y(rider_features: pd.DataFrame):
    X = align_features(rider_features)
    y = rider_features["churn"]
    return X, y


def build_training_data(data_dir: Path = DATA_DIR):
    riders, sessions, trips = load_raw_data(data_dir)
    riders = add_target(riders)
    rider_features = build_rider_features(riders, sessions, trips)
    return get_X_y(rider_features)
