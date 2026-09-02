from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from pybaseball import playerid_lookup

from src.hr_model import FEATURE_COLUMNS, predict_hr_probability


FEATURE_SNAPSHOT_PATH = Path("models/hr_batter_features.csv")


def _normalize_name(name: str) -> str:
    return " ".join(str(name).replace(".", "").replace("'", "").lower().split())


def _split_name(full_name: str) -> tuple[str, str]:
    parts = str(full_name).strip().split()
    if len(parts) < 2:
        raise ValueError(f"Cannot split player name: {full_name}")
    return parts[0], " ".join(parts[1:])


@lru_cache(maxsize=512)
def resolve_mlbam_id(player_name: str) -> int | None:
    """
    Resolve sportsbook player name to MLBAM id using pybaseball's player registry.
    """
    first, last = _split_name(player_name)

    try:
        result = playerid_lookup(last, first)
    except Exception as exc:
        print(f"PLAYER LOOKUP WARNING: {player_name}: {exc}")
        return None

    if result is None or result.empty or "key_mlbam" not in result.columns:
        return None

    result = result[result["key_mlbam"].notna()].copy()
    if result.empty:
        return None

    # Prefer the most recent MLB record when several matches exist.
    if "mlb_played_last" in result.columns:
        result = result.sort_values("mlb_played_last", ascending=False)

    try:
        return int(result.iloc[0]["key_mlbam"])
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_feature_snapshot() -> pd.DataFrame:
    if not FEATURE_SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            f"Live HR feature snapshot not found at {FEATURE_SNAPSHOT_PATH}. "
            "Re-run Train HR Model after updating train_hr_model.py."
        )

    df = pd.read_csv(FEATURE_SNAPSHOT_PATH)

    if "batter" not in df.columns:
        raise RuntimeError("HR feature snapshot is missing batter ids.")

    df["batter"] = pd.to_numeric(df["batter"], errors="coerce")
    df = df[df["batter"].notna()].copy()
    df["batter"] = df["batter"].astype(int)

    return df.set_index("batter", drop=False)


def features_for_player(player_name: str) -> dict[str, float] | None:
    batter_id = resolve_mlbam_id(player_name)
    if batter_id is None:
        print(f"HR MODEL: could not resolve MLB id for {player_name}")
        return None

    snapshot = load_feature_snapshot()

    if batter_id not in snapshot.index:
        print(f"HR MODEL: no feature snapshot for {player_name} ({batter_id})")
        return None

    row = snapshot.loc[batter_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    features: dict[str, float] = {}

    for col in FEATURE_COLUMNS:
        value = row.get(col, 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        features[col] = value

    return features


def model_probability_for_player(player_name: str) -> float | None:
    features = features_for_player(player_name)
    if features is None:
        return None

    try:
        return float(predict_hr_probability(features))
    except Exception as exc:
        print(f"HR MODEL ERROR: {player_name}: {exc}")
        return None
