import os
from dotenv import load_dotenv

load_dotenv()

def csv_env(name: str, default: str):
    return [x.strip() for x in os.getenv(name, default).split(',') if x.strip()]

ODDS_API_KEY = os.getenv('ODDS_API_KEY', '')
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
SPORT_KEYS = csv_env('SPORT_KEYS', 'americanfootball_nfl,basketball_nba')
REGIONS = os.getenv('REGIONS', 'us')
MARKETS = os.getenv('MARKETS', 'h2h,spreads,totals')
MIN_EDGE_PCT = float(os.getenv('MIN_EDGE_PCT', '4.0'))
MIN_EV_PCT = float(os.getenv('MIN_EV_PCT', '3.0'))
PAPER_STAKE_UNITS = float(os.getenv('PAPER_STAKE_UNITS', '1.0'))
