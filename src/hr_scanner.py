from collections import defaultdict

from .math_utils import implied_probability, expected_value, no_vig_two_way


def get_hr_candidates(event):
    """
    Finds sportsbook consensus probability for a player
    to hit 1+ HR, then compares each book's price to consensus.
    """

    player_probs = defaultdict(list)

    # Build no-vig consensus probabilities
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "batter_home_runs":
                continue

            players = defaultdict(dict)

            for outcome in market.get("outcomes", []):
                player = outcome.get("description")
                side = outcome.get("name")
                point = outcome.get("point")

                if not player or point != 0.5:
                    continue

                players[player][side] = float(outcome["price"])

            for player, prices in players.items():
                if "Over" not in prices or "Under" not in prices:
                    continue

                over_prob = implied_probability(prices["Over"])
                under_prob = implied_probability(prices["Under"])

                over_nv, _ = no_vig_two_way(
                    over_prob,
                    under_prob
                )

                player_probs[player].append(over_nv)

    consensus = {
        player: sum(probs) / len(probs)
        for player, probs in player_probs.items()
        if len(probs) >= 2
    }

    candidates = []

    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):

            if market.get("key") != "batter_home_runs":
                continue

            for outcome in market.get("outcomes", []):

                if outcome.get("name") != "Over":
                    continue

                if outcome.get("point") != 0.5:
                    continue

                player = outcome.get("description")

                if player not in consensus:
                    continue

                odds = float(outcome["price"])

                model_prob = consensus[player]
                book_prob = implied_probability(odds)

                edge = model_prob - book_prob
                ev = expected_value(model_prob, odds)

                candidates.append({
                    "event_id": event["id"],
                    "sport_key": "baseball_mlb",
                    "sport_title": "MLB",
                    "commence_time": event["commence_time"],
                    "home_team": event["home_team"],
                    "away_team": event["away_team"],
                    "market": "batter_home_runs",
                    "selection": f"{player} 1+ HR",
                    "bookmaker": book.get(
                        "title",
                        book.get("key")
                    ),
                    "american_odds": odds,
                    "model_probability": model_prob,
                    "market_probability": book_prob,
                    "edge_pct": edge * 100,
                    "ev_pct": ev * 100,
                })

    return sorted(
        candidates,
        key=lambda x: x["ev_pct"],
        reverse=True,
    )
