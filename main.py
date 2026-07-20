from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing import LOYALTY_ORDER, align_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rf_churn_model.joblib"
model = joblib.load(MODEL_PATH)




# instantiate the FastAPI app
app = FastAPI(title="Ridewise Churn Prediction")


# define the input data model
class RiderFeatures(BaseModel):
    age: int = Field(example=29)
    avg_rating_given: float = Field(example=4.6)
    loyalty_status: Literal["Bronze", "Silver", "Gold", "Platinum"] = Field(example="Gold")
    tenure_days: int = Field(example=420)
    riders_referred: int = Field(example=2)
    city: Literal["Cairo", "Lagos", "Nairobi"] = Field(example="Lagos")
    total_trips: int = Field(example=87)
    avg_fare: float = Field(example=12.5)
    avg_tip: float = Field(example=1.2)
    tip_rate: float = Field(example=0.4)
    avg_trip_duration: float = Field(example=18.3)
    days_since_last_trip: int = Field(example=5)
    days_active: int = Field(example=300)
    avg_trip_distance: float = Field(example=6.7)
    peak_hour_ratio: float = Field(example=0.35)
    trip_rate: float = Field(example=0.29)
    activity_decline: float = Field(example=0.8)
    

def to_model_frame(rider: RiderFeatures) -> pd.DataFrame:
    """Map API-friendly fields (loyalty_status, city) to the encoded
    columns FEATURE_COLUMNS expects, then align to training-time schema."""
    row = rider.model_dump()

    row["loyalty_encoded"] = LOYALTY_ORDER[row.pop("loyalty_status")]

    city = row.pop("city")
    row["city_Lagos"] = int(city == "Lagos")
    row["city_Nairobi"] = int(city == "Nairobi")

    return align_features(pd.DataFrame([row]))

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(rider: RiderFeatures):
    try:
        X = to_model_frame(rider)
        churn_probability = float(model.predict_proba(X)[0, 1])
        prediction = int(churn_probability >= 0.5)

        return {
            "churn_prediction": prediction,
            "label": "Churn" if prediction == 1 else "Retained",
            "churn_probability": churn_probability,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

