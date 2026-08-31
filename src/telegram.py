import requests


def send_message(token: str, chat_id: str, text: str):
    if not token or not chat_id:
        return None
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    r = requests.post(url, json={'chat_id': chat_id, 'text': text}, timeout=20)
    r.raise_for_status()
    return r.json()


def format_pick(p: dict):
    fire = '🔥🔥🔥' if p['edge_pct'] >= 8 else ('🔥🔥' if p['edge_pct'] >= 5 else '🔥')
    return (
        f"{fire} SPORTS EDGE ALERT\n"
        f"{p['away_team']} @ {p['home_team']}\n"
        f"Pick: {p['selection']} {p['american_odds']:+.0f}\n"
        f"Book: {p['bookmaker']}\n"
        f"Model probability: {p['model_probability']:.1%}\n"
        f"Book implied: {p['market_probability']:.1%}\n"
        f"Edge: {p['edge_pct']:.1f}%\n"
        f"EV: {p['ev_pct']:.1f}%\n"
        f"Mode: PAPER BET"
    )
