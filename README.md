# Sports Edge Scanner

A paper-betting sports market scanner built around Python, The Odds API, Supabase, Streamlit, GitHub Actions, and Telegram.

## What V1 does

- Pulls current NFL/NBA bookmaker odds.
- Removes two-way moneyline vig to build a conservative cross-book consensus probability.
- Finds individual sportsbook prices that are meaningfully better than consensus.
- Computes probability edge and expected value (EV).
- Stores qualified paper picks in Supabase.
- Sends Telegram alerts.
- Displays recorded picks in Streamlit.

> Important: V1 is a market-pricing baseline, not yet a team-strength predictive model. That is deliberate: it gives us a clean data collection and validation layer before adding ML.

## Setup

1. Create an account/API key at The Odds API.
2. Run `schema.sql` in your Supabase SQL editor.
3. Add these GitHub repository secrets:
   - `ODDS_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
4. Push this repo to GitHub.
5. Open Actions > Sports Edge Scan > Run workflow.
6. Add the same Supabase values to Streamlit secrets/environment variables and deploy `app.py`.

## Why paper mode first

We need forward-looking data to measure calibration, ROI, closing-line value, hit rate by edge bucket, and whether the model survives out-of-sample testing. Do not treat alerts as guaranteed winners.

## Planned V2

- Results settlement and unit P/L
- Closing line capture / CLV
- Team-strength and player-feature models
- Injuries/rest/travel/weather features
- Spread and total modeling
- Edge calibration by sport/market
- Duplicate-alert suppression
- Bankroll/Kelly simulation (paper only by default)
- Model versioning and retraining pipeline
