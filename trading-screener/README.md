# Momentum & Options Screener (Streamlit)

Screens liquid momentum names (price >$5, vol >1M, mcap >$1B, RSI 35-60, breakout/support bounce, earnings/short-squeeze catalysts),
ranks 3-21 DTE option contracts (vol/OI >1000, spread <10%, delta .30-.50, R:R >= 3:1), and scans pre/post-market penny movers
with news keyword + red-flag checks. Alerts go to Discord and/or Telegram.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy free on Streamlit Community Cloud
1. Push this repo to GitHub.
2. share.streamlit.io -> New app -> pick repo, branch `main`, file `app.py`.
3. App settings -> Secrets: paste the contents of `.streamlit/secrets.toml.example` with your values.

## 24/7 alerts without a browser (GitHub Actions)
Repo Settings -> Secrets and variables -> Actions:
- Secrets: `DISCORD_WEBHOOK` and/or `TELEGRAM_TOKEN` + `TELEGRAM_CHAT_ID`
- Variables (optional): `WATCHLIST`, `PENNY_WATCHLIST`, `AH_MOVE`
Workflow `alerts.yml` runs every 5 min on weekdays. GitHub cron is best-effort (can lag several minutes), so this is not tick-level real time.

## Limits
Yahoo data is delayed/unofficial and rate-limited; replace `core.py` fetchers with Polygon/Alpaca/Tradier for true real time.
POP is a risk-neutral lognormal estimate. Not financial advice.
