"""Screening engine: no UI, no printing. Used by app.py (Streamlit) and alerts_runner.py (GitHub Actions)."""
import math, os, datetime as dt
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

R = 0.04
POS_WORDS = ["fda", "approval", "approved", "contract", "beats", "raises", "guidance",
             "partnership", "acquire", "acquisition", "buyout", "upgrade", "breakthrough",
             "record", "wins", "award", "phase 3", "merger", "short squeeze", "clearance"]
NEG_WORDS = ["offering", "dilution", "reverse split", "delist", "halt", "going concern",
             "bankruptcy", "investigation", "downgrade", "misses", "lawsuit"]


# ------------------------------------------------------------------ math
def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_delta(S, K, T, iv, is_call):
    if T <= 0 or iv <= 0:
        return 0.0
    d1 = (math.log(S / K) + (R + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    return ncdf(d1) if is_call else ncdf(d1) - 1


def prob_beyond(S, level, T, iv, above=True):
    if T <= 0 or iv <= 0:
        return 0.0
    d = (math.log(S / level) + (R - 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    return ncdf(d) if above else 1 - ncdf(d)


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ------------------------------------------------------------------ universe
def sp500():
    try:
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text
        return pd.read_html(StringIO(html))[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        return []


def parse_tickers(text):
    return sorted({t.strip().upper() for t in text.replace(",", "\n").splitlines() if t.strip()})


def earnings_days(tk):
    try:
        cal = tk.calendar
        ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if ed:
            d = ed[0] if isinstance(ed, (list, tuple)) else ed
            return (pd.Timestamp(d).date() - dt.date.today()).days
    except Exception:
        pass
    return None


# ------------------------------------------------------------------ step 1
def scan_underlyings(universe, rsi_lo=35, rsi_hi=60, min_rvol=1.5, min_price=5.0,
                     min_avg_vol=1_000_000, min_mcap_b=1.0, top=10, direction="long"):
    data = yf.download(universe, period="6mo", interval="1d", group_by="ticker",
                       threads=True, progress=False, auto_adjust=True)
    rows = []
    for t in universe:
        try:
            df = data[t].dropna() if len(universe) > 1 else data.dropna()
            if len(df) < 60:
                continue
            c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
            price, avgv = float(c.iloc[-1]), float(v.iloc[-21:-1].mean())
            if price <= min_price or avgv < min_avg_vol:
                continue
            rvol = float(v.iloc[-1] / avgv)
            r = float(rsi(c).iloc[-1])
            sma50 = float(c.rolling(50).mean().iloc[-1])
            if direction == "long":
                brk = price >= h.iloc[-21:-1].max() * 0.995
                bnc = abs(l.iloc[-1] / sma50 - 1) < 0.015 and price > sma50 and c.iloc[-1] > c.iloc[-2]
                setup = "breakout" if brk else "support bounce"
            else:
                brk = price <= l.iloc[-21:-1].min() * 1.005
                bnc = abs(h.iloc[-1] / sma50 - 1) < 0.015 and price < sma50 and c.iloc[-1] < c.iloc[-2]
                setup = "breakdown" if brk else "resistance reject"
            if not (brk or bnc) or rvol < min_rvol or not (rsi_lo <= r <= rsi_hi):
                continue
            rows.append(dict(ticker=t, price=round(price, 2), rvol=round(rvol, 2), rsi=round(r, 1),
                             chg_pct=round((c.iloc[-1] / c.iloc[-2] - 1) * 100, 2), setup=setup))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    cand = pd.DataFrame(rows).sort_values("rvol", ascending=False).head(25)
    out = []
    for row in cand.to_dict("records"):
        tk = yf.Ticker(row["ticker"])
        try:
            info = tk.info
        except Exception:
            info = {}
        mcap = info.get("marketCap") or 0
        if mcap < min_mcap_b * 1e9:
            continue
        sf = (info.get("shortPercentOfFloat") or 0) * 100
        ed = earnings_days(tk)
        cat = []
        if ed is not None and 0 <= ed <= 14:
            cat.append(f"earnings in {ed}d")
        if sf >= 15:
            cat.append(f"short float {sf:.0f}% (squeeze)")
        row.update(mcap_B=round(mcap / 1e9, 1), short_float=round(sf, 1),
                   catalyst=", ".join(cat) or "none found (technical only)",
                   score=row["rvol"] * (1.5 if cat else 1.0))
        out.append(row)
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values("score", ascending=False).head(top).reset_index(drop=True)


# ------------------------------------------------------------------ steps 2-3
def analyze_options(ticker, spot, bullish=True, dmin=3, dmax=21, dlo=0.30, dhi=0.50,
                    min_vol=1000, min_oi=1000, stop_pct=0.5, em_mult=0.8):
    tk = yf.Ticker(ticker)
    today = dt.date.today()
    rows = []
    for exp in tk.options:
        dte = (dt.date.fromisoformat(exp) - today).days
        if not dmin <= dte <= dmax:
            continue
        T = dte / 365
        ch = tk.option_chain(exp)
        calls, puts = ch.calls, ch.puts
        pcr = puts["volume"].fillna(0).sum() / max(calls["volume"].fillna(0).sum(), 1)
        atm = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
        atm_iv = float(atm["impliedVolatility"].iloc[0])
        em = spot * atm_iv * math.sqrt(T)
        side = calls if bullish else puts
        for _, o in side.iterrows():
            bid, ask, vol, oi, iv = o.bid, o.ask, o.volume, o.openInterest, o.impliedVolatility
            if not (bid > 0 and vol and oi and vol > min_vol and oi > min_oi and iv > 0):
                continue
            mid, spr = (bid + ask) / 2, ask - bid
            if not (spr / mid < 0.10 or spr < 0.10):
                continue
            delta = abs(bs_delta(spot, o.strike, T, iv, bullish))
            if not dlo <= delta <= dhi:
                continue
            be = o.strike + mid if bullish else o.strike - mid
            tgt_und = spot + em_mult * em if bullish else spot - em_mult * em
            tgt = max(tgt_und - o.strike, 0) if bullish else max(o.strike - tgt_und, 0)
            risk = mid * stop_pct
            rr = (tgt - mid) / risk if tgt > mid else 0
            pop = prob_beyond(spot, be, T, iv, above=bullish)
            rows.append(dict(ticker=ticker, type="CALL" if bullish else "PUT", strike=o.strike, exp=exp,
                             dte=dte, entry=round(mid, 2), breakeven=round(be, 2), target=round(tgt, 2),
                             stop=round(mid * (1 - stop_pct), 2), roi_pct=round((tgt / mid - 1) * 100),
                             max_loss_usd=round(mid * 100), rr=round(rr, 1), pop_pct=round(pop * 100),
                             delta=round(delta, 2), iv_pct=round(iv * 100), exp_move=round(em, 2),
                             pc_ratio=round(pcr, 2), spread=round(spr, 2), volume=int(vol), oi=int(oi),
                             score=rr * pop))
    return pd.DataFrame(rows)


def top_contracts(und, bullish=True, n=5, min_rr=3.0, **kw):
    frames = []
    for t, p in und.head(5)[["ticker", "price"]].itertuples(index=False):
        try:
            f = analyze_options(t, p, bullish=bullish, **kw)
            if not f.empty:
                frames.append(f)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    a = pd.concat(frames)
    a = a[a.rr >= min_rr]
    return a.sort_values("score", ascending=False).head(n).reset_index(drop=True)


def pnl_curve(row, n=41):
    """Option P&L at expiry across underlying prices (+/-1.5 expected moves) for one contract."""
    spot_guess = row["breakeven"] - row["entry"] if row["type"] == "CALL" else row["breakeven"] + row["entry"]
    em = row["exp_move"]
    xs = np.linspace(spot_guess - 1.5 * em, spot_guess + 1.5 * em, n)
    k = row["strike"]
    val = np.maximum(xs - k, 0) if row["type"] == "CALL" else np.maximum(k - xs, 0)
    return pd.DataFrame({"underlying": xs.round(2), "pnl_usd": ((val - row["entry"]) * 100).round(0)})


# ------------------------------------------------------------------ penny / after-hours
def penny_candidates(extra=()):
    c = list(extra)
    for name in ("day_gainers", "most_actives"):
        try:
            res = yf.screen(name, count=100)
            c += [q["symbol"] for q in res.get("quotes", []) if q.get("regularMarketPrice", 99) < 5]
        except Exception:
            pass
    return sorted(set(c))


def scan_extended_hours(extra=(), min_move=15, max_price=5.0, min_avg_vol=500_000):
    hits = []
    for t in penny_candidates(extra):
        try:
            tk = yf.Ticker(t)
            info = tk.info
            reg = info.get("regularMarketPrice")
            ext = info.get("postMarketPrice") or info.get("preMarketPrice")
            avgv = info.get("averageVolume") or 0
            if not reg or not ext or ext > max_price or avgv < min_avg_vol:
                continue
            move = (ext / reg - 1) * 100
            if move < min_move:
                continue
            news = [n.get("title") or n.get("content", {}).get("title", "") for n in (tk.news or [])[:5]]
            low = " ".join(x.lower() for x in news if x)
            pos = [w for w in POS_WORDS if w in low]
            neg = [w for w in NEG_WORDS if w in low]
            hits.append(dict(ticker=t, ext_price=round(ext, 3), move_pct=round(move, 1), avg_vol=int(avgv),
                             pos_keywords=", ".join(pos) or "-", red_flags=", ".join(neg) or "-",
                             headline=news[0] if news else ""))
        except Exception:
            continue
    return pd.DataFrame(hits).sort_values("move_pct", ascending=False).reset_index(drop=True) if hits else pd.DataFrame()


# ------------------------------------------------------------------ alerts
def send_alert(msg, discord=None, tg_token=None, tg_chat=None):
    discord = discord or os.getenv("DISCORD_WEBHOOK")
    tg_token = tg_token or os.getenv("TELEGRAM_TOKEN")
    tg_chat = tg_chat or os.getenv("TELEGRAM_CHAT_ID")
    sent = []
    try:
        if discord:
            requests.post(discord, json={"content": msg[:1900]}, timeout=10); sent.append("discord")
        if tg_token and tg_chat:
            requests.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                          data={"chat_id": tg_chat, "text": msg}, timeout=10); sent.append("telegram")
    except Exception:
        pass
    return sent


def contract_msg(r):
    return (f"{r['ticker']} {r['type']} {r['strike']} {r['exp']} | entry ${r['entry']} | tgt ${r['target']} "
            f"({r['roi_pct']}%) | stop ${r['stop']} | R:R {r['rr']} | POP~{r['pop_pct']}% | BE {r['breakeven']}")
