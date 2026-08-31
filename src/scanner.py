from .math_utils import implied_probability, expected_value
from .model import market_consensus_probability


def scan_event(event: dict, min_edge_pct: float, min_ev_pct: float):
    picks = []
    consensus = market_consensus_probability(event, 'h2h')

    for book in event.get('bookmakers', []):
        for market in book.get('markets', []):
            if market.get('key') != 'h2h':
                continue
            for outcome in market.get('outcomes', []):
                name = outcome['name']
                odds = float(outcome['price'])
                model_prob = consensus.get(name)
                if model_prob is None:
                    continue
                market_prob = implied_probability(odds)
                edge = model_prob - market_prob
                ev = expected_value(model_prob, odds)
                if edge * 100 >= min_edge_pct and ev * 100 >= min_ev_pct:
                    picks.append({
                        'event_id': event['id'],
                        'sport_key': event['sport_key'],
                        'sport_title': event.get('sport_title'),
                        'commence_time': event['commence_time'],
                        'home_team': event['home_team'],
                        'away_team': event['away_team'],
                        'market': 'h2h',
                        'selection': name,
                        'bookmaker': book.get('title', book.get('key')),
                        'american_odds': odds,
                        'model_probability': model_prob,
                        'market_probability': market_prob,
                        'edge_pct': edge * 100,
                        'ev_pct': ev * 100,
                    })
    return picks
