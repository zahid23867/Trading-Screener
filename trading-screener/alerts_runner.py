"""Headless alert run for GitHub Actions. Env: DISCORD_WEBHOOK and/or TELEGRAM_TOKEN + TELEGRAM_CHAT_ID."""
import datetime as dt, json, os
from zoneinfo import ZoneInfo

import core

ET = ZoneInfo("America/New_York")
SEEN_FILE = ".seen_alerts.json"  # persisted via actions/cache to avoid duplicate alerts


def load_seen():
    try:
        d = json.load(open(SEEN_FILE))
        return set(d.get("keys", [])) if d.get("day") == str(dt.date.today()) else set()
    except Exception:
        return set()


def save_seen(keys):
    json.dump({"day": str(dt.date.today()), "keys": sorted(keys)}, open(SEEN_FILE, "w"))


def main():
    now = dt.datetime.now(ET)
    mins = now.hour * 60 + now.minute
    if now.weekday() >= 5:
        print("weekend, skipping"); return
    watch = core.parse_tickers(os.getenv("WATCHLIST", "NVDA,TSLA,AMD,PLTR,COIN,SMCI,MSTR,AAPL,META,AMZN"))
    penny = core.parse_tickers(os.getenv("PENNY_WATCHLIST", ""))
    seen = load_seen()
    if 570 <= mins < 960:
        und = core.scan_underlyings(watch)
        print(und.to_string() if not und.empty else "no underlyings")
        if not und.empty:
            for r in core.top_contracts(und).to_dict("records"):
                k = f"{r['ticker']}{r['strike']}{r['exp']}"
                if k not in seen:
                    seen.add(k); print(core.send_alert("📈 " + core.contract_msg(r)))
    elif 240 <= mins < 570 or 960 <= mins < 1200:
        ah = core.scan_extended_hours(extra=penny, min_move=float(os.getenv("AH_MOVE", "15")))
        for r in ah.to_dict("records"):
            k = f"AH{r['ticker']}"
            if k not in seen:
                seen.add(k)
                core.send_alert(f"🚨 AH {r['ticker']} ${r['ext_price']} {r['move_pct']:+}% | +{r['pos_keywords']} | flags {r['red_flags']} | {r['headline']}")
    save_seen(seen)


if __name__ == "__main__":
    main()
