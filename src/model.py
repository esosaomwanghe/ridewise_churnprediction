"""Churn model: regularized Random Forest on rider-level features.

This configuration was selected over several alternatives (see
notebook/modeling.ipynb) not because it scored highest on held-out
ROC-AUC, but because it was the only one whose train and test AUC
were close together. Every unconstrained variant (max_depth=None)
hit train AUC 1.0 by memorizing the training set while scoring at or
below random chance on held-out data. This dataset's `churn` label
has no strong learnable relationship to the available features,
regardless of grain, feature set, or model choice — this model is
the most honest artifact from that search, not a high-performing one.
"""

import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

if __package__ in (None, ""):
    # Allows `python src/model.py` in addition to `python -m src.model`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import align_features, build_training_data

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "rf_churn_model.joblib"

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1,
)


def train(test_size: float = 0.2, random_state: int = 42):
    """Fit the deployed model (regularized RF). Used by save/load below."""
    X, y = build_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    smote = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train_sm, y_train_sm)

    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_prob_train = model.predict_proba(X_train)[:, 1]

    metrics = {
        "roc_auc_train": roc_auc_score(y_train, y_prob_train),
        "roc_auc_test": roc_auc_score(y_test, y_prob_test),
        "report": classification_report(y_test, model.predict(X_test), target_names=["Retained", "Churned"]),
    }
    return model, metrics


def save_model(model, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict_proba(model, rider_features):
    """Churn probability for each row. `rider_features` may have extra columns;
    only FEATURE_COLUMNS are used, and any missing ones are filled with 0."""
    X = align_features(rider_features)
    return model.predict_proba(X)[:, 1]


def train_with_mlflow_tracking(test_size: float = 0.2, random_state: int = 42):
    """Fit Logistic Regression and Random Forest side by side, logging both
    to MLflow for comparison. Not used at inference time — run this file
    directly (or call this function) when you want a tracked experiment."""
    MODELS_DIR.mkdir(exist_ok=True)

    X, y = build_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    smote = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ride-wise-project")

    with mlflow.start_run(run_name="logistic-regression"):
        lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
        lr_model.fit(X_train_sm, y_train_sm)

        y_prob = lr_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("features", X.columns.tolist())
        mlflow.log_param("resampling", "SMOTE")
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(lr_model, artifact_path="model")
        joblib.dump(lr_model, MODELS_DIR / "logistic_regression.pkl")

        print(f"Logistic Regression AUC: {auc:.4f}")

    with mlflow.start_run(run_name="random-forest"):
        rf_model = RandomForestClassifier(**RF_PARAMS)
        rf_model.fit(X_train_sm, y_train_sm)

        y_prob = rf_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("features", X.columns.tolist())
        mlflow.log_param("resampling", "SMOTE")
        mlflow.log_params(RF_PARAMS)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(rf_model, artifact_path="model")

        feature_importance = (
            pd.Series(rf_model.feature_importances_, index=X.columns)
            .sort_values(ascending=False)
            .rename("importance")
            .rename_axis("feature")
            .reset_index()
        )
        fi_path = MODELS_DIR / "feature_importance_rf.csv"
        feature_importance.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))
        joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")

        print(f"Random Forest AUC: {auc:.4f}")

    return lr_model, rf_model


if __name__ == "__main__":
    model, metrics = train()
    print(metrics["report"])
    print(f"ROC-AUC train: {metrics['roc_auc_train']:.4f}")
    print(f"ROC-AUC test : {metrics['roc_auc_test']:.4f}")
    save_model(model)
    print(f"Saved model to {MODEL_PATH}")

    train_with_mlflow_tracking()
