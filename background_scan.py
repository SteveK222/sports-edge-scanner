from datetime import datetime, timezone, timedelta

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

from src.odds_client import (
    get_odds,
    get_events,
    get_event_odds,
)

from src.scanner import scan_event
from src.hr_scanner import get_hr_candidates

from src.storage import get_client, save_pick
from src.telegram import send_message, format_pick


HR_MIN_EDGE_PCT = 2.0
HR_MIN_EV_PCT = 3.0

HR_LOOKAHEAD_HOURS = 10

TOP_NEAR_MISSES = 5


def save_and_alert(db, pick):

    pick["paper_stake_units"] = PAPER_STAKE_UNITS
    pick["status"] = "open"

    save_pick(db, pick)

    send_message(
        TELEGRAM_BOT_TOKEN,
        TELEGRAM_CHAT_ID,
        format_pick(pick),
    )


def print_near_misses(candidates, title):

    if not candidates:
        print(f"\n{title}: none")
        return

    candidates = sorted(
        candidates,
        key=lambda x: x.get("ev_pct", -999),
        reverse=True,
    )

    print(f"\n===== {title} =====")

    for pick in candidates[:TOP_NEAR_MISSES]:

        print(
            f"{pick.get('away_team')} @ "
            f"{pick.get('home_team')} | "
            f"{pick.get('selection')} | "
            f"{pick.get('bookmaker')} | "
            f"{pick.get('american_odds'):+.0f} | "
            f"edge={pick.get('edge_pct', 0):.2f}% | "
            f"EV={pick.get('ev_pct', 0):.2f}%"
        )


def regular_scan(db):

    qualified = []
    near_misses = []

    for sport in SPORT_KEYS:

        events, quota = get_odds(
            ODDS_API_KEY,
            sport,
            REGIONS,
            MARKETS,
        )

        print(
            f"{sport}: {len(events)} events | "
            f"quota={quota}"
        )

        for event in events:

            picks = scan_event(
                event,
                MIN_EDGE_PCT,
                MIN_EV_PCT,
            )

            for pick in picks:
                qualified.append(pick)

                save_and_alert(
                    db,
                    pick,
                )

    return qualified, near_misses


def hr_scan(db):

    sport = "baseball_mlb"

    events = get_events(
        ODDS_API_KEY,
        sport,
    )

    now = datetime.now(timezone.utc)

    cutoff = now + timedelta(
        hours=HR_LOOKAHEAD_HOURS
    )

    eligible = []

    for event in events:

        start = datetime.fromisoformat(
            event["commence_time"].replace(
                "Z",
                "+00:00"
            )
        )

        if now <= start <= cutoff:
            eligible.append(event)

    print(
        f"\nMLB HR games eligible: "
        f"{len(eligible)}"
    )

    all_candidates = []
    qualified = []

    for event in eligible:

        try:

            event_odds, quota = get_event_odds(
                ODDS_API_KEY,
                sport,
                event["id"],
                REGIONS,
                "batter_home_runs",
            )
                       print(
            "DEBUG HR RESPONSE:",
            event["away_team"],
            "@",
            event["home_team"],
            "| bookmakers:",
            len(event_odds.get("bookmakers", []))
        )

        for book in event_odds.get("bookmakers", []):
            print(
                "  BOOK:",
                book.get("title"),
                "| MARKETS:",
                [m.get("key") for m in book.get("markets", [])]
            )

            for market in book.get("markets", []):
                if market.get("key") == "batter_home_runs":
                    print(
                        "    HR OUTCOMES SAMPLE:",
                        market.get("outcomes", [])[:4]
                    )

        candidates = get_hr_candidates(
            event_odds
        )

        all_candidates.extend(
            candidates
        )

        game_qualified = [
            p for p in candidates
            if p["edge_pct"] >= HR_MIN_EDGE_PCT
            and p["ev_pct"] >= HR_MIN_EV_PCT
        ]

        print(
            f"{event['away_team']} @ "
            f"{event['home_team']} | "
            f"HR candidates={len(candidates)} | "
            f"qualified={len(game_qualified)} | "
            f"quota={quota}"
        )

        for pick in game_qualified:
            save_and_alert(
                db,
                pick,
            )

            qualified.append(
                pick
            )
                    pick
                )

        except Exception as exc:

            print(
                f"HR error: "
                f"{event.get('away_team')} @ "
                f"{event.get('home_team')} | "
                f"{exc}"
            )

    print_near_misses(
        all_candidates,
        "TOP 5 HR EDGES"
    )

    return qualified


def main():

    db = get_client(
        SUPABASE_URL,
        SUPABASE_KEY,
    )

    regular_picks, regular_near = regular_scan(
        db
    )

    hr_picks = hr_scan(
        db
    )

    print(
        f"\nqualified game picks: "
        f"{len(regular_picks)}"
    )

    print(
        f"qualified HR picks: "
        f"{len(hr_picks)}"
    )


if __name__ == "__main__":
    main()
