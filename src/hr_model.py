from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/hr_model.joblib")


FEATURE_COLUMNS = [
    "season_hr_rate",
    "recent_hr_rate",
    "avg_exit_velocity",
    "hard_hit_rate",
    "barrel_rate",
    "avg_launch_angle",
    "pitcher_hr_rate",
    "expected_plate_appearances",
]


def model_exists():
    return MODEL_PATH.exists()


def load_hr_model():
    if not model_exists():
        raise FileNotFoundError(
            f"HR model not found at {MODEL_PATH}. "
            "Run train_hr_model.py first."
        )

    return joblib.load(MODEL_PATH)


def prepare_features(features):
    """
    Convert one hitter's feature dictionary into the exact
    dataframe format expected by the trained HR model.
    """

    row = {}

    for column in FEATURE_COLUMNS:
        value = features.get(column, 0)

        if value is None:
            value = 0

        row[column] = float(value)

    return pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS,
    )


def predict_hr_probability(features):
    """
    Returns the model's estimated probability that
    the hitter records at least one home run.
    """

    model = load_hr_model()

    X = prepare_features(features)

    probability = model.predict_proba(X)[0][1]

    return float(probability)


def predict_hr_percent(features):
    return predict_hr_probability(features) * 100
