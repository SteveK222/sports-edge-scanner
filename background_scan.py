from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any

from src.config import (
    ODDS_API_KEY,
    SUPABASE_URL,
    SUPABASE_KEY,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    SPORT_KEYS,
    REGIONS,
    MARKETS,
    MIN_EDGE_PCT,
    MIN_EV_PCT,
    PAPER_STAKE_UNITS,
)

from src.odds_client import get_odds, get_events, get_event_odds
from src.storage import get_client, save_pick
from src.telegram import send_message, format_pick
from src.hr_live_features import model_probability_for_player


TOP_NEAR_MISSES = 5

# Paper-bet thresholds for the independent HR model.
HR_MIN_EDGE_PCT = 4.0
HR_MIN_EV_PCT = 8.0
HR_LOOKAHEAD_HOURS = 10

# Require at least two books for game-line consensus.
MIN_CONSENSUS_BOOKS = 2


def implied_probability(american_odds: float) -> float:
    odds = float(american_odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    raise ValueError("American odds cannot be 0.")


def decimal_odds(american_odds: float) -> float:
    odds = float(american_odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    if odds < 0:
        return 1.0 + 100.0 / (-odds)
    raise ValueError("American odds cannot be 0.")


def expected_value(probability: float, american_odds: float) -> float:
    return probability * decimal_odds(american_odds) - 1.0


def normalize_probabilities(raw_probs: dict[str, float]) -> dict[str, float]:
    total = sum(raw_probs.values())
    if total <= 0:
        return {}
    return {name: prob / total for name, prob in raw_probs.items()}


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_send_pick(pick: dict[str, Any]) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("TELEGRAM: skipped - token/chat id missing")
        return

    try:
        send_message(
            TELEGRAM_BOT_TOKEN,
            TELEGRAM_CHAT_ID,
            format_pick(pick),
        )
        print("TELEGRAM: alert sent")
    except Exception as exc:
        print(f"TELEGRAM WARNING: alert failed but scan will continue: {exc}")


def save_and_alert(db, pick: dict[str, Any]) -> None:
    pick = dict(pick)
    pick["paper_stake_units"] = PAPER_STAKE_UNITS
    pick["status"] = "open"

    try:
        save_pick(db, pick)
        print(
            "SAVED:",
            pick.get("sport_title"),
            "|",
            pick.get("away_team"),
            "@",
            pick.get("home_team"),
            "|",
            pick.get("selection"),
            "|",
            pick.get("bookmaker"),
            "|",
            pick.get("american_odds"),
        )
    except Exception as exc:
        print(f"SUPABASE ERROR: could not save pick: {exc}")
        return

    safe_send_pick(pick)


def print_top_candidates(candidates: list[dict[str, Any]], title: str) -> None:
    print(f"\n===== {title} =====")

    if not candidates:
        print("none")
        return

    ranked = sorted(
        candidates,
        key=lambda p: (p.get("ev_pct", -999.0), p.get("edge_pct", -999.0)),
        reverse=True,
    )

    for pick in ranked[:TOP_NEAR_MISSES]:
        odds = safe_float(pick.get("american_odds"))
        odds_text = f"{odds:+.0f}" if odds is not None else str(pick.get("american_odds"))

        print(
            f"{pick.get('sport_title')} | "
            f"{pick.get('away_team')} @ {pick.get('home_team')} | "
            f"{pick.get('selection')} | "
            f"{pick.get('bookmaker')} | "
            f"{odds_text} | "
            f"model={pick.get('model_probability', 0) * 100:.2f}% | "
            f"book={pick.get('market_probability', 0) * 100:.2f}% | "
            f"edge={pick.get('edge_pct', 0):.2f}% | "
            f"EV={pick.get('ev_pct', 0):.2f}%"
        )


# -----------------------------
# Regular moneyline scanner
# -----------------------------

def get_h2h_market(bookmaker: dict[str, Any]) -> dict[str, Any] | None:
    for market in bookmaker.get("markets", []):
        if market.get("key") == "h2h":
            return market
    return None


def build_game_candidates(event: dict[str, Any]) -> list[dict[str, Any]]:
    book_probabilities: dict[str, dict[str, float]] = {}
    book_prices: dict[str, dict[str, float]] = {}
    book_titles: dict[str, str] = {}

    for book in event.get("bookmakers", []):
        book_key = book.get("key") or book.get("title")
        if not book_key:
            continue

        market = get_h2h_market(book)
        if not market:
            continue

        raw_probs: dict[str, float] = {}
        prices: dict[str, float] = {}

        for outcome in market.get("outcomes", []):
            name = outcome.get("name")
            price = safe_float(outcome.get("price"))
            if not name or price is None:
                continue
            raw_probs[name] = implied_probability(price)
            prices[name] = price

        if len(raw_probs) < 2:
            continue

        no_vig = normalize_probabilities(raw_probs)
        if not no_vig:
            continue

        book_probabilities[book_key] = no_vig
        book_prices[book_key] = prices
        book_titles[book_key] = book.get("title") or str(book_key)

    if len(book_probabilities) < MIN_CONSENSUS_BOOKS:
        return []

    candidates: list[dict[str, Any]] = []

    for book_key, prices in book_prices.items():
        for selection, price in prices.items():
            comparison_probs = [
                probs[selection]
                for other_key, probs in book_probabilities.items()
                if other_key != book_key and selection in probs
            ]

            if not comparison_probs:
                comparison_probs = [
                    probs[selection]
                    for probs in book_probabilities.values()
                    if selection in probs
                ]

            if not comparison_probs:
                continue

            consensus_prob = median(comparison_probs)
            book_prob = implied_probability(price)
            edge = consensus_prob - book_prob
            ev = expected_value(consensus_prob, price)

            candidates.append({
                "event_id": event.get("id"),
                "sport_key": event.get("sport_key"),
                "sport_title": event.get("sport_title") or event.get("sport_key"),
                "commence_time": event.get("commence_time"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "market": "h2h",
                "selection": selection,
                "bookmaker": book_titles.get(book_key, str(book_key)),
                "american_odds": price,
                "model_probability": consensus_prob,
                "market_probability": book_prob,
                "edge_pct": edge * 100.0,
                "ev_pct": ev * 100.0,
            })

    return candidates


def regular_scan(db) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_candidates: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []

    for sport in SPORT_KEYS:
        try:
            events, quota = get_odds(
                ODDS_API_KEY,
                sport,
                REGIONS,
                MARKETS,
            )
        except Exception as exc:
            print(f"{sport}: ODDS API ERROR: {exc}")
            continue

        print(f"{sport}: {len(events)} events | quota={quota}")

        events_with_books = 0
        sport_candidates = 0

        for event in events:
            if event.get("bookmakers"):
                events_with_books += 1

            candidates = build_game_candidates(event)
            sport_candidates += len(candidates)
            all_candidates.extend(candidates)

            for pick in candidates:
                if (
                    pick["edge_pct"] >= MIN_EDGE_PCT
                    and pick["ev_pct"] >= MIN_EV_PCT
                ):
                    qualified.append(pick)
                    save_and_alert(db, pick)

        print(
            f"{sport}: events_with_bookmakers={events_with_books} | "
            f"raw_candidates={sport_candidates}"
        )

    print_top_candidates(all_candidates, "TOP 5 GAME EDGES")
    return qualified, all_candidates


# -----------------------------
# Independent MLB HR model
# -----------------------------

def extract_positive_hr_prices(event: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract each sportsbook's 1+ HR price.

    Supports common outcome naming:
    - Over 0.5
    - Yes
    """
    rows: list[dict[str, Any]] = []

    for book in event.get("bookmakers", []):
        book_name = book.get("title") or book.get("key")

        for market in book.get("markets", []):
            if market.get("key") != "batter_home_runs":
                continue

            for outcome in market.get("outcomes", []):
                player = outcome.get("description")
                side = str(outcome.get("name", "")).strip().lower()
                point = safe_float(outcome.get("point"))
                odds = safe_float(outcome.get("price"))

                if not player or odds is None:
                    continue

                positive = side in {"over", "yes"}

                if side == "over" and point is not None and abs(point - 0.5) > 1e-9:
                    positive = False

                if not positive:
                    continue

                rows.append({
                    "player": player,
                    "bookmaker": book_name,
                    "american_odds": odds,
                })

    return rows


def build_model_hr_candidates(event: dict[str, Any]) -> list[dict[str, Any]]:
    prices = extract_positive_hr_prices(event)

    if not prices:
        return []

    # Cache model probability once per player even if several books are present.
    player_probabilities: dict[str, float | None] = {}
    candidates: list[dict[str, Any]] = []

    for row in prices:
        player = row["player"]

        if player not in player_probabilities:
            player_probabilities[player] = model_probability_for_player(player)

        model_prob = player_probabilities[player]
        if model_prob is None:
            continue

        odds = row["american_odds"]
        book_prob = implied_probability(odds)
        edge = model_prob - book_prob
        ev = expected_value(model_prob, odds)

        candidates.append({
            "event_id": event.get("id"),
            "sport_key": event.get("sport_key") or "baseball_mlb",
            "sport_title": event.get("sport_title") or "MLB",
            "commence_time": event.get("commence_time"),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
            "market": "batter_home_runs",
            "selection": f"{player} 1+ HR",
            "bookmaker": row["bookmaker"],
            "american_odds": odds,
            "model_probability": model_prob,
            "market_probability": book_prob,
            "edge_pct": edge * 100.0,
            "ev_pct": ev * 100.0,
        })

    return candidates


def hr_scan(db) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sport = "baseball_mlb"

    try:
        events = get_events(ODDS_API_KEY, sport)
    except Exception as exc:
        print(f"MLB HR events error: {exc}")
        return [], []

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=HR_LOOKAHEAD_HOURS)

    eligible: list[dict[str, Any]] = []

    for event in events:
        commence = event.get("commence_time")
        if not commence:
            continue

        try:
            start = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        except Exception:
            continue

        if now <= start <= cutoff:
            eligible.append(event)

    print(f"\nMLB HR games eligible: {len(eligible)}")

    all_candidates: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []

    for event in eligible:
        away = event.get("away_team")
        home = event.get("home_team")

        try:
            event_odds, quota = get_event_odds(
                ODDS_API_KEY,
                sport,
                event["id"],
                REGIONS,
                "batter_home_runs",
            )
        except Exception as exc:
            print(f"HR API ERROR: {away} @ {home} | {exc}")
            continue

        bookmakers = event_odds.get("bookmakers", [])
        raw_outcomes = sum(
            len(market.get("outcomes", []))
            for book in bookmakers
            for market in book.get("markets", [])
            if market.get("key") == "batter_home_runs"
        )

        candidates = build_model_hr_candidates(event_odds)
        all_candidates.extend(candidates)

        game_qualified = [
            pick
            for pick in candidates
            if (
                pick["edge_pct"] >= HR_MIN_EDGE_PCT
                and pick["ev_pct"] >= HR_MIN_EV_PCT
            )
        ]

        print(
            f"{away} @ {home} | "
            f"bookmakers={len(bookmakers)} | "
            f"raw_hr_outcomes={raw_outcomes} | "
            f"model_candidates={len(candidates)} | "
            f"qualified={len(game_qualified)} | "
            f"quota={quota}"
        )

        for pick in game_qualified:
            qualified.append(pick)
            save_and_alert(db, pick)

    print_top_candidates(all_candidates, "TOP 5 HR MODEL EDGES")
    return qualified, all_candidates


def main() -> None:
    print("Sports Edge Scanner starting...")
    print(
        f"Regular thresholds: edge >= {MIN_EDGE_PCT}% | "
        f"EV >= {MIN_EV_PCT}%"
    )
    print(
        f"HR MODEL thresholds: edge >= {HR_MIN_EDGE_PCT}% | "
        f"EV >= {HR_MIN_EV_PCT}%"
    )

    db = get_client(SUPABASE_URL, SUPABASE_KEY)

    regular_picks, regular_candidates = regular_scan(db)
    hr_picks, hr_candidates = hr_scan(db)

    print("\n===== SCAN SUMMARY =====")
    print(f"raw game candidates: {len(regular_candidates)}")
    print(f"qualified game picks: {len(regular_picks)}")
    print(f"raw HR model candidates: {len(hr_candidates)}")
    print(f"qualified HR model picks: {len(hr_picks)}")

    if not regular_picks and not hr_picks:
        print("No bets met the current alert thresholds.")


if __name__ == "__main__":
    main()
