from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
from pybaseball import statcast
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.hr_model import FEATURE_COLUMNS


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "hr_model.joblib"
TRAINING_DATA_PATH = MODEL_DIR / "hr_training_data.csv"

# First-pass training window.
DAYS_BACK = 120

MIN_BATTER_SAMPLE = 20
MIN_PITCHER_SAMPLE = 20


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    numerator = pd.to_numeric(numerator, errors="coerce")
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_statcast_history(start_dt: str, end_dt: str) -> pd.DataFrame:
    print(f"Downloading Statcast data: {start_dt} -> {end_dt}")

    df = statcast(start_dt=start_dt, end_dt=end_dt)

    if df is None or df.empty:
        raise RuntimeError("No Statcast data returned.")

    print(f"Downloaded {len(df):,} pitch rows")
    return df.copy()


def add_core_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["game_date"] = pd.to_datetime(
        df["game_date"],
        errors="coerce",
    ).dt.date

    df = df[df["game_date"].notna()].copy()

    # A non-null events value marks the pitch that ended a plate appearance.
    df["is_pa_end"] = df["events"].notna()

    # Target event.
    df["is_hr"] = (
        df["events"].astype("string").eq("home_run")
    ).fillna(False).astype(int)

    df["launch_speed_num"] = pd.to_numeric(
        df["launch_speed"],
        errors="coerce",
    )

    df["launch_angle_num"] = pd.to_numeric(
        df["launch_angle"],
        errors="coerce",
    )

    # Baseball Savant's launch_speed_angle category 6 = Barrel.
    # fillna(False) is important because Statcast columns can use pandas
    # nullable dtypes, which otherwise cannot be converted directly to int.
    launch_speed_angle = pd.to_numeric(
        df["launch_speed_angle"],
        errors="coerce",
    )

    df["is_barrel"] = (
        launch_speed_angle.eq(6)
        .fillna(False)
        .astype(int)
    )

    # Standard hard-hit threshold is 95+ mph exit velocity.
    df["is_hard_hit"] = (
        df["launch_speed_num"].ge(95.0)
        .fillna(False)
        .astype(int)
    )

    return df


def build_batter_game_rows(df: pd.DataFrame) -> pd.DataFrame:
    pa = df[df["is_pa_end"]].copy()

    if pa.empty:
        raise RuntimeError("No completed plate appearances found.")

    game_rows = (
        pa.groupby(
            ["game_date", "game_pk", "batter"],
            as_index=False,
            dropna=False,
        )
        .agg(
            plate_appearances=("events", "count"),
            homers=("is_hr", "sum"),
            pitcher=("pitcher", "first"),
            stand=("stand", "first"),
            p_throws=("p_throws", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
        )
    )

    game_rows["home_run"] = (
        game_rows["homers"] > 0
    ).astype(int)

    return game_rows


def add_prior_cumulative(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    output_col: str,
) -> pd.DataFrame:
    """
    Cumulative total strictly BEFORE the current row.

    Using cumsum() - current value avoids the subtle cross-player leakage
    that can happen with a plain global .shift() after groupby().cumsum().
    """
    frame[output_col] = (
        frame.groupby(group_col)[value_col].cumsum()
        - frame[value_col]
    )
    return frame


def build_batter_history_features(
    df: pd.DataFrame,
    game_rows: pd.DataFrame,
) -> pd.DataFrame:
    pa = df[df["is_pa_end"]].copy()

    batter_game = (
        pa.groupby(
            ["game_date", "game_pk", "batter"],
            as_index=False,
            dropna=False,
        )
        .agg(
            pa=("events", "count"),
            hr=("is_hr", "sum"),
        )
        .sort_values(
            ["batter", "game_date", "game_pk"]
        )
        .reset_index(drop=True)
    )

    batter_game = add_prior_cumulative(
        batter_game,
        "batter",
        "pa",
        "cum_pa_before",
    )

    batter_game = add_prior_cumulative(
        batter_game,
        "batter",
        "hr",
        "cum_hr_before",
    )

    batter_game["season_hr_rate"] = safe_rate(
        batter_game["cum_hr_before"],
        batter_game["cum_pa_before"],
    )

    batter_game["recent_pa_before"] = (
        batter_game.groupby("batter")["pa"]
        .transform(
            lambda s: s.shift(1).rolling(
                10,
                min_periods=1,
            ).sum()
        )
        .fillna(0.0)
    )

    batter_game["recent_hr_before"] = (
        batter_game.groupby("batter")["hr"]
        .transform(
            lambda s: s.shift(1).rolling(
                10,
                min_periods=1,
            ).sum()
        )
        .fillna(0.0)
    )

    batter_game["recent_hr_rate"] = safe_rate(
        batter_game["recent_hr_before"],
        batter_game["recent_pa_before"],
    )

    keep = batter_game[
        [
            "game_date",
            "game_pk",
            "batter",
            "season_hr_rate",
            "recent_hr_rate",
            "cum_pa_before",
        ]
    ]

    return game_rows.merge(
        keep,
        on=["game_date", "game_pk", "batter"],
        how="left",
    )


def build_contact_features(
    df: pd.DataFrame,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    contact = df[
        df["launch_speed_num"].notna()
    ].copy()

    if contact.empty:
        rows["avg_exit_velocity"] = 0.0
        rows["hard_hit_rate"] = 0.0
        rows["barrel_rate"] = 0.0
        rows["avg_launch_angle"] = 0.0
        rows["bbe_before"] = 0.0
        return rows

    batter_contact_game = (
        contact.groupby(
            ["game_date", "game_pk", "batter"],
            as_index=False,
            dropna=False,
        )
        .agg(
            bbe=("launch_speed_num", "count"),
            exit_velo_sum=("launch_speed_num", "sum"),
            hard_hits=("is_hard_hit", "sum"),
            barrels=("is_barrel", "sum"),
            launch_angle_sum=("launch_angle_num", "sum"),
            launch_angle_n=("launch_angle_num", "count"),
        )
        .sort_values(
            ["batter", "game_date", "game_pk"]
        )
        .reset_index(drop=True)
    )

    for col in [
        "bbe",
        "exit_velo_sum",
        "hard_hits",
        "barrels",
        "launch_angle_sum",
        "launch_angle_n",
    ]:
        batter_contact_game = add_prior_cumulative(
            batter_contact_game,
            "batter",
            col,
            f"{col}_before",
        )

    batter_contact_game["avg_exit_velocity"] = safe_rate(
        batter_contact_game["exit_velo_sum_before"],
        batter_contact_game["bbe_before"],
    )

    batter_contact_game["hard_hit_rate"] = safe_rate(
        batter_contact_game["hard_hits_before"],
        batter_contact_game["bbe_before"],
    )

    batter_contact_game["barrel_rate"] = safe_rate(
        batter_contact_game["barrels_before"],
        batter_contact_game["bbe_before"],
    )

    batter_contact_game["avg_launch_angle"] = safe_rate(
        batter_contact_game["launch_angle_sum_before"],
        batter_contact_game["launch_angle_n_before"],
    )

    keep = batter_contact_game[
        [
            "game_date",
            "game_pk",
            "batter",
            "avg_exit_velocity",
            "hard_hit_rate",
            "barrel_rate",
            "avg_launch_angle",
            "bbe_before",
        ]
    ]

    return rows.merge(
        keep,
        on=["game_date", "game_pk", "batter"],
        how="left",
    )


def build_pitcher_features(
    df: pd.DataFrame,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    pa = df[df["is_pa_end"]].copy()

    pitcher_game = (
        pa.groupby(
            ["game_date", "game_pk", "pitcher"],
            as_index=False,
            dropna=False,
        )
        .agg(
            batters_faced=("events", "count"),
            homers_allowed=("is_hr", "sum"),
        )
        .sort_values(
            ["pitcher", "game_date", "game_pk"]
        )
        .reset_index(drop=True)
    )

    pitcher_game = add_prior_cumulative(
        pitcher_game,
        "pitcher",
        "batters_faced",
        "bf_before",
    )

    pitcher_game = add_prior_cumulative(
        pitcher_game,
        "pitcher",
        "homers_allowed",
        "hr_allowed_before",
    )

    pitcher_game["pitcher_hr_rate"] = safe_rate(
        pitcher_game["hr_allowed_before"],
        pitcher_game["bf_before"],
    )

    keep = pitcher_game[
        [
            "game_date",
            "game_pk",
            "pitcher",
            "pitcher_hr_rate",
            "bf_before",
        ]
    ]

    return rows.merge(
        keep,
        on=["game_date", "game_pk", "pitcher"],
        how="left",
    )


def finalize_features(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()

    rows = rows.sort_values(
        ["batter", "game_date", "game_pk"]
    ).reset_index(drop=True)

    rows["expected_plate_appearances"] = (
        rows.groupby("batter")["plate_appearances"]
        .transform(
            lambda s: s.shift(1).rolling(
                10,
                min_periods=1,
            ).mean()
        )
    )

    defaults = {
        "season_hr_rate": 0.0,
        "recent_hr_rate": 0.0,
        "avg_exit_velocity": 88.0,
        "hard_hit_rate": 0.0,
        "barrel_rate": 0.0,
        "avg_launch_angle": 12.0,
        "pitcher_hr_rate": 0.0,
        "expected_plate_appearances": 4.0,
    }

    for col, default in defaults.items():
        rows[col] = pd.to_numeric(
            rows[col],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(default)

    if "cum_pa_before" in rows.columns:
        small_batter_sample = (
            pd.to_numeric(
                rows["cum_pa_before"],
                errors="coerce",
            ).fillna(0) < MIN_BATTER_SAMPLE
        )

        rows.loc[
            small_batter_sample,
            ["season_hr_rate", "recent_hr_rate"],
        ] *= 0.5

    if "bbe_before" in rows.columns:
        small_contact_sample = (
            pd.to_numeric(
                rows["bbe_before"],
                errors="coerce",
            ).fillna(0) < MIN_BATTER_SAMPLE
        )

        rows.loc[
            small_contact_sample,
            ["hard_hit_rate", "barrel_rate"],
        ] *= 0.5

    if "bf_before" in rows.columns:
        small_pitcher_sample = (
            pd.to_numeric(
                rows["bf_before"],
                errors="coerce",
            ).fillna(0) < MIN_PITCHER_SAMPLE
        )

        rows.loc[
            small_pitcher_sample,
            "pitcher_hr_rate",
        ] *= 0.5

    return rows


def build_training_table(df: pd.DataFrame) -> pd.DataFrame:
    print("Building one-row-per-hitter-game training table...")

    df = add_core_flags(df)

    rows = build_batter_game_rows(df)
    rows = build_batter_history_features(df, rows)
    rows = build_contact_features(df, rows)
    rows = build_pitcher_features(df, rows)
    rows = finalize_features(rows)
       snapshot_columns = ["batter"] + FEATURE_COLUMNS

    latest_snapshot = (
        rows.sort_values(["batter", "game_date", "game_pk"])
        .groupby("batter", as_index=False)
        .tail(1)[snapshot_columns]
        .copy()
    )

    league_pitcher_hr_rate = float(
        pd.to_numeric(
            rows["pitcher_hr_rate"],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna().median()
    )

    latest_snapshot["pitcher_hr_rate"] = league_pitcher_hr_rate

    snapshot_path = MODEL_DIR / "hr_batter_features.csv"

    latest_snapshot.to_csv(
        snapshot_path,
        index=False,
    )

    print(
        f"Saved live batter feature snapshot: "
        f"{snapshot_path} "
        f"({len(latest_snapshot):,} batters)"
    )

    training = rows[
        FEATURE_COLUMNS + ["home_run"]
    ].copy()

    training = training.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    print(
        f"Training rows: {len(training):,} | "
        f"HR rate: {training['home_run'].mean():.3%}"
    )

    return training

def train_model(training: pd.DataFrame):
    X = training[FEATURE_COLUMNS]
    y = training["home_run"].astype(int)

    if y.nunique() < 2:
        raise RuntimeError(
            "Training data needs both HR and non-HR examples."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    base_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )

    model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=3,
    )

    print("Training HR model...")
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    print("\n===== HOLDOUT RESULTS =====")

    try:
        print(
            f"ROC AUC: "
            f"{roc_auc_score(y_test, probs):.4f}"
        )
    except Exception:
        print("ROC AUC: unavailable")

    print(
        f"Brier score: "
        f"{brier_score_loss(y_test, probs):.5f}"
    )

    print(
        f"Log loss: "
        f"{log_loss(y_test, probs):.5f}"
    )

    calibration = pd.DataFrame({
        "predicted": probs,
        "actual": y_test.to_numpy(),
    })

    calibration["bucket"] = pd.cut(
        calibration["predicted"],
        bins=[
            0.00,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.40,
            1.00,
        ],
        include_lowest=True,
    )

    print("\n===== CALIBRATION BUCKETS =====")

    bucket_table = (
        calibration.groupby(
            "bucket",
            observed=False,
        )
        .agg(
            predictions=("actual", "size"),
            avg_model_probability=("predicted", "mean"),
            actual_hr_rate=("actual", "mean"),
        )
    )

    print(bucket_table.to_string())

    return model


def main() -> None:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=DAYS_BACK)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_statcast_history(
        start.isoformat(),
        end.isoformat(),
    )

    training = build_training_table(df)

    training.to_csv(
        TRAINING_DATA_PATH,
        index=False,
    )

    print(
        f"Saved training data: "
        f"{TRAINING_DATA_PATH}"
    )

    model = train_model(training)

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"\nSUCCESS: saved HR model to "
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
