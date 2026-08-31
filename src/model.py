from collections import defaultdict
from .math_utils import implied_probability, no_vig_two_way


def market_consensus_probability(event: dict, market_key: str):
    """V1 baseline model: no-vig consensus across available books.

    This is intentionally conservative. Later versions can blend team/player
    features with market information using logistic regression / gradient boosting.
    """
    prices = defaultdict(list)

    for book in event.get('bookmakers', []):
        for market in book.get('markets', []):
            if market.get('key') != market_key:
                continue
            outcomes = market.get('outcomes', [])
            if market_key == 'h2h' and len(outcomes) == 2:
                a, b = outcomes
                p1 = implied_probability(float(a['price']))
                p2 = implied_probability(float(b['price']))
                nv1, nv2 = no_vig_two_way(p1, p2)
                prices[a['name']].append(nv1)
                prices[b['name']].append(nv2)

    return {name: sum(vals)/len(vals) for name, vals in prices.items() if vals}
