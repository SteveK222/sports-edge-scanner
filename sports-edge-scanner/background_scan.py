from src.config import (
    ODDS_API_KEY, SUPABASE_URL, SUPABASE_KEY, TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID, SPORT_KEYS, REGIONS, MARKETS, MIN_EDGE_PCT,
    MIN_EV_PCT, PAPER_STAKE_UNITS,
)
from src.odds_client import get_odds
from src.scanner import scan_event
from src.storage import get_client, save_pick
from src.telegram import send_message, format_pick


def main():
    db = get_client(SUPABASE_URL, SUPABASE_KEY)
    total = 0
    for sport in SPORT_KEYS:
        events, quota = get_odds(ODDS_API_KEY, sport, REGIONS, MARKETS)
        print(f'{sport}: {len(events)} events | quota={quota}')
        for event in events:
            picks = scan_event(event, MIN_EDGE_PCT, MIN_EV_PCT)
            for pick in picks:
                pick['paper_stake_units'] = PAPER_STAKE_UNITS
                pick['status'] = 'open'
                save_pick(db, pick)
                send_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, format_pick(pick))
                total += 1
    print(f'qualified picks: {total}')


if __name__ == '__main__':
    main()
