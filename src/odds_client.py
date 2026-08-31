import requests

BASE = 'https://api.the-odds-api.com/v4'


def get_odds(api_key: str, sport: str, regions: str, markets: str):
    if not api_key:
        raise RuntimeError('ODDS_API_KEY is missing')
    r = requests.get(
        f'{BASE}/sports/{sport}/odds/',
        params={
            'apiKey': api_key,
            'regions': regions,
            'markets': markets,
            'oddsFormat': 'american',
            'dateFormat': 'iso',
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json(), {
        'remaining': r.headers.get('x-requests-remaining'),
        'used': r.headers.get('x-requests-used'),
        'last': r.headers.get('x-requests-last'),
    }
