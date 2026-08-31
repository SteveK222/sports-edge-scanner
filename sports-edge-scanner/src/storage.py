from supabase import create_client


def get_client(url: str, key: str):
    if not url or not key:
        return None
    return create_client(url, key)


def save_pick(client, pick: dict):
    if client is None:
        return None
    return client.table('sports_picks').insert(pick).execute()
