from __future__ import annotations

from collections import defaultdict
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

TOP_NEAR_MISSES = 5
HR_MIN_EDGE_PCT = 2.0
HR_MIN_EV_PCT = 3.0
HR_LOOKAHEAD_HOURS = 10
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
            f"consensus={pick.get('model_probability', 0) * 100:.2f}% | "
            f"book={pick.get('market_probability', 0) * 100:.2f}% | "
            f"edge={pick.get('edge_pct', 0):.2f}% | "
            f"EV={pick.get('ev_pct', 0):.2f}%"
        )


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


POSITIVE_HR_NAMES = {"over", "yes"}
NEGATIVE_HR_NAMES = {"under", "no"}


def parse_hr_book(book: dict[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}

    for market in book.get("markets", []):
        if market.get("key") != "batter_home_runs":
            continue

        grouped: dict[str, dict[str, float]] = defaultdict(dict)

        for outcome in market.get("outcomes", []):
            player = outcome.get("description")
            name = str(outcome.get("name", "")).strip().lower()
            point = outcome.get("point")
            price = safe_float(outcome.get("price"))

            if not player or price is None:
                continue

            if point is not None:
                point_value = safe_float(point)
                if point_value is not None and abs(point_value - 0.5) > 1e-9:
                    continue

            if name in POSITIVE_HR_NAMES:
                grouped[player]["positive"] = price
            elif name in NEGATIVE_HR_NAMES:
                grouped[player]["negative"] = price

        for player, sides in grouped.items():
            positive_price = sides.get("positive")
            negative_price = sides.get("negative")

            if positive_price is None:
                continue

            positive_raw = implied_probability(positive_price)

            if negative_price is not None:
                negative_raw = implied_probability(negative_price)
                no_vig = normalize_probabilities({
                    "positive": positive_raw,
                    "negative": negative_raw,
                })
                probability = no_vig.get("positive")
            else:
                probability = positive_raw

            if probability is None:
                continue

            result[player] = {
                "positive_price": positive_price,
                "probability": probability,
            }

    return result


def build_hr_candidates(event: dict[str, Any]) -> list[dict[str, Any]]:
    parsed_books: dict[str, dict[str, dict[str, float]]] = {}
    book_titles: dict[str, str] = {}

    for book in event.get("bookmakers", []):
        book_key = book.get("key") or book.get("title")
        if not book_key:
            continue

        parsed = parse_hr_book(book)

        if parsed:
            parsed_books[book_key] = parsed
            book_titles[book_key] = book.get("title") or str(book_key)

    if len(parsed_books) < MIN_CONSENSUS_BOOKS:
        return []

    candidates: list[dict[str, Any]] = []

    for book_key, players in parsed_books.items():
        for player, info in players.items():
            positive_price = info["positive_price"]

            comparison_probs = [
                other_players[player]["probability"]
                for other_key, other_players in parsed_books.items()
                if other_key != book_key and player in other_players
            ]

            if not comparison_probs:
                comparison_probs = [
                    players_map[player]["probability"]
                    for players_map in parsed_books.values()
                    if player in players_map
                ]

            if not comparison_probs:
                continue

            consensus_prob = median(comparison_probs)
            book_prob = implied_probability(positive_price)
            edge = consensus_prob - book_prob
            ev = expected_value(consensus_prob, positive_price)

            candidates.append({
                "event_id": event.get("id"),
                "sport_key": event.get("sport_key") or "baseball_mlb",
                "sport_title": event.get("sport_title") or "MLB",
                "commence_time": event.get("commence_time"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "market": "batter_home_runs",
                "selection": f"{player} 1+ HR",
                "bookmaker": book_titles.get(book_key, str(book_key)),
                "american_odds": positive_price,
                "model_probability": consensus_prob,
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

        market_keys = sorted({
            market.get("key")
            for book in bookmakers
            for market in book.get("markets", [])
            if market.get("key")
        })

        raw_outcomes = sum(
            len(market.get("outcomes", []))
            for book in bookmakers
            for market in book.get("markets", [])
            if market.get("key") == "batter_home_runs"
        )

        print(
            f"DEBUG HR: {away} @ {home} | "
            f"bookmakers={len(bookmakers)} | "
            f"markets={market_keys} | "
            f"raw_hr_outcomes={raw_outcomes}"
        )

        candidates = build_hr_candidates(event_odds)
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
            f"HR candidates={len(candidates)} | "
            f"qualified={len(game_qualified)} | "
            f"quota={quota}"
        )

        for pick in game_qualified:
            qualified.append(pick)
            save_and_alert(db, pick)

    print_top_candidates(all_candidates, "TOP 5 HR EDGES")
    return qualified, all_candidates


def main() -> None:
    print("Sports Edge Scanner starting...")
    print(
        f"Regular thresholds: edge >= {MIN_EDGE_PCT}% | "
        f"EV >= {MIN_EV_PCT}%"
    )
    print(
        f"HR thresholds: edge >= {HR_MIN_EDGE_PCT}% | "
        f"EV >= {HR_MIN_EV_PCT}%"
    )

    db = get_client(SUPABASE_URL, SUPABASE_KEY)

    regular_picks, regular_candidates = regular_scan(db)
    hr_picks, hr_candidates = hr_scan(db)

    print("\n===== SCAN SUMMARY =====")
    print(f"raw game candidates: {len(regular_candidates)}")
    print(f"qualified game picks: {len(regular_picks)}")
    print(f"raw HR candidates: {len(hr_candidates)}")
    print(f"qualified HR picks: {len(hr_picks)}")

    if not regular_picks and not hr_picks:
        print("No bets met the current alert thresholds.")


if __name__ == "__main__":
    main()
