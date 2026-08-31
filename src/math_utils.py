def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1 + odds / 100.0
    return 1 + 100.0 / abs(odds)


def implied_probability(odds: float) -> float:
    return 1.0 / american_to_decimal(odds)


def no_vig_two_way(p1: float, p2: float):
    total = p1 + p2
    if total <= 0:
        return 0.5, 0.5
    return p1 / total, p2 / total


def expected_value(prob: float, american_odds: float) -> float:
    dec = american_to_decimal(american_odds)
    return prob * (dec - 1) - (1 - prob)
