import os
import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title='Sports Edge Scanner', layout='wide')
st.title('🏈🏀 Sports Edge Scanner')
st.caption('Paper-betting mode — model edge, EV, and performance tracking')

url = os.getenv('SUPABASE_URL', '')
key = os.getenv('SUPABASE_KEY', '')
if not url or not key:
    st.warning('Add SUPABASE_URL and SUPABASE_KEY to Streamlit secrets/environment variables.')
    st.stop()

client = create_client(url, key)
rows = client.table('sports_picks').select('*').order('created_at', desc=True).limit(500).execute().data
if not rows:
    st.info('No picks recorded yet.')
    st.stop()

df = pd.DataFrame(rows)

c1, c2, c3, c4 = st.columns(4)
c1.metric('Recorded picks', len(df))
c2.metric('Avg edge', f"{df['edge_pct'].mean():.1f}%" if 'edge_pct' in df else '—')
c3.metric('Avg EV', f"{df['ev_pct'].mean():.1f}%" if 'ev_pct' in df else '—')
if 'profit_units' in df:
    c4.metric('Paper P/L', f"{df['profit_units'].fillna(0).sum():+.2f}u")
else:
    c4.metric('Paper P/L', 'Pending')

st.dataframe(df, use_container_width=True)
