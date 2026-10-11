import time, datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="Momentum & Options Screener", page_icon="📈", layout="wide")
ET = ZoneInfo("America/New_York")


def secret(k):
    try:
        return st.secrets.get(k)
    except Exception:
        return None


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Filters")
    direction = st.radio("Bias", ["long", "short"], horizontal=True)
    c1, c2 = st.columns(2)
    rsi_lo = c1.number_input("RSI min", 0, 100, 35)
    rsi_hi = c2.number_input("RSI max", 0, 100, 60)
    min_rvol = st.slider("Min relative volume", 1.0, 5.0, 1.5, 0.1)
    min_price = st.number_input("Min price $", 1.0, 500.0, 5.0)
    min_mcap = st.number_input("Min mkt cap $B", 0.1, 100.0, 1.0)
    extra = st.text_area("Extra tickers (comma/line separated)", "NVDA, TSLA, AMD, PLTR, COIN, SMCI, MSTR")
    use_sp = st.checkbox("Include S&P 500 universe (slower)", False)
    st.subheader("Options")
    dte = st.slider("DTE range", 1, 45, (3, 21))
    d_lo, d_hi = st.slider("Delta range", 0.10, 0.80, (0.30, 0.50), 0.01)
    min_rr = st.slider("Min R:R", 1.0, 6.0, 3.0, 0.5)
    stop_pct = st.slider("Stop (% premium lost)", 0.2, 0.8, 0.5, 0.05)
    em_mult = st.slider("Target (× IV expected move)", 0.3, 1.5, 0.8, 0.1)
    st.subheader("After-hours")
    ah_move = st.slider("Min ext-hours move %", 5, 100, 15)
    ah_extra = st.text_area("Penny watchlist", "")
    st.subheader("Alerts")
    discord = st.text_input("Discord webhook", secret("DISCORD_WEBHOOK") or "", type="password")
    tg_token = st.text_input("Telegram bot token", secret("TELEGRAM_TOKEN") or "", type="password")
    tg_chat = st.text_input("Telegram chat id", secret("TELEGRAM_CHAT_ID") or "")

now = dt.datetime.now(ET)
mins = now.hour * 60 + now.minute
if now.weekday() >= 5:
    session = "CLOSED (weekend)"
elif 570 <= mins < 960:
    session = "REGULAR"
elif 240 <= mins < 570:
    session = "PRE-MARKET"
elif 960 <= mins < 1200:
    session = "AFTER-HOURS"
else:
    session = "CLOSED"

st.title("📈 Momentum & Options Screener")
st.caption(f"ET {now:%a %H:%M} · session: **{session}** · data: Yahoo (delayed, unofficial). "
           "Not financial advice. Options are high risk; max loss = premium paid.")

if "alerted" not in st.session_state:
    st.session_state.alerted = set()


@st.cache_data(ttl=300, show_spinner=False)
def run_underlyings(universe, *args):
    return core.scan_underlyings(list(universe), *args)


@st.cache_data(ttl=300, show_spinner=False)
def run_contracts(und_json, bullish, dmin, dmax, dlo, dhi, stop_pct, em_mult, min_rr):
    und = pd.read_json(und_json)
    return core.top_contracts(und, bullish=bullish, min_rr=min_rr, dmin=dmin, dmax=dmax,
                              dlo=dlo, dhi=dhi, stop_pct=stop_pct, em_mult=em_mult)


@st.cache_data(ttl=120, show_spinner=False)
def run_ah(extra, move):
    return core.scan_extended_hours(extra=list(extra), min_move=move)


def push(msg, key):
    if key in st.session_state.alerted:
        return
    st.session_state.alerted.add(key)
    sent = core.send_alert(msg, discord or None, tg_token or None, tg_chat or None)
    st.toast(("🔔 " + msg)[:200] + (f"  → {','.join(sent)}" if sent else ""))


tab1, tab2, tab3, tab4 = st.tabs(["1 · Underlyings", "2-4 · Top contracts", "After-hours pennies", "Live mode"])

universe = core.parse_tickers(extra)
if use_sp:
    universe = sorted(set(universe + core.sp500()))

with tab1:
    if st.button("Scan underlyings", type="primary") or "und" in st.session_state:
        if st.button("Force refresh"):
            st.cache_data.clear()
            st.session_state.pop("und", None)
            st.rerun()
        with st.spinner(f"Scanning {len(universe)} tickers..."):
            und = run_underlyings(tuple(universe), rsi_lo, rsi_hi, min_rvol, min_price, 1_000_000, min_mcap, 10, direction)
        st.session_state.und = und
        if und.empty:
            st.warning("Nothing passed. Loosen RVOL/RSI, or add tickers. A quiet scan is information, not a bug.")
        else:
            st.dataframe(und.drop(columns="score"), use_container_width=True, hide_index=True)

with tab2:
    und = st.session_state.get("und")
    if und is None or und.empty:
        st.info("Run the underlying scan first.")
    else:
        bullish = direction == "long"
        with st.spinner("Pulling option chains..."):
            top = run_contracts(und.to_json(), bullish, dte[0], dte[1], d_lo, d_hi, stop_pct, em_mult, min_rr)
        st.session_state.top = top
        if top.empty:
            st.warning("No contract passed volume/OI/spread/delta/R:R filters. Do not force a trade.")
        else:
            st.dataframe(top.drop(columns="score"), use_container_width=True, hide_index=True)
            pick = st.selectbox("P&L sketch at expiry", [f"{r.ticker} {r.type} {r.strike} {r.exp}" for r in top.itertuples()])
            row = top.iloc[[f"{r.ticker} {r.type} {r.strike} {r.exp}" for r in top.itertuples()].index(pick)].to_dict()
            st.line_chart(core.pnl_curve(row).set_index("underlying"))
            st.caption(f"Max loss ${row['max_loss_usd']}/contract · breakeven {row['breakeven']} · "
                       f"stop at ${row['stop']} · target ${row['target']} (~{row['roi_pct']}%).")
            if st.button("Send these to alerts"):
                for r in top.to_dict("records"):
                    push(core.contract_msg(r), f"{r['ticker']}{r['strike']}{r['exp']}{dt.date.today()}")

with tab3:
    st.write("Pre/post-market movers under $5 with news keyword check. Red flags (offering, halt, reverse split) "
             "mean read the filing before touching it.")
    if st.button("Scan extended hours"):
        ah = run_ah(tuple(core.parse_tickers(ah_extra)), ah_move)
        if ah.empty:
            st.info("No movers right now (extended-hours quotes only exist 4:00-9:30 and 16:00-20:00 ET).")
        else:
            st.dataframe(ah, use_container_width=True, hide_index=True)
            for r in ah.to_dict("records"):
                push(f"AH {r['ticker']} ${r['ext_price']} {r['move_pct']:+}% | +{r['pos_keywords']} | flags {r['red_flags']} | {r['headline']}",
                     f"AH{r['ticker']}{dt.date.today()}")

with tab4:
    st.write("Auto-refresh loop while this tab stays open. For 24/7 alerts without the browser, "
             "use the GitHub Actions workflow in this repo.")
    live = st.toggle("Run live (re-scan every N seconds)")
    every = st.slider("Interval (s)", 60, 600, 300, 30)
    if live:
        und = run_underlyings(tuple(universe), rsi_lo, rsi_hi, min_rvol, min_price, 1_000_000, min_mcap, 10, direction)
        if not und.empty:
            top = run_contracts(und.to_json(), direction == "long", dte[0], dte[1], d_lo, d_hi, stop_pct, em_mult, min_rr)
            for r in top.to_dict("records"):
                push(core.contract_msg(r), f"{r['ticker']}{r['strike']}{r['exp']}{dt.date.today()}")
            st.dataframe(top.drop(columns="score", errors="ignore"), use_container_width=True, hide_index=True)
        if session in ("PRE-MARKET", "AFTER-HOURS"):
            for r in run_ah(tuple(core.parse_tickers(ah_extra)), ah_move).to_dict("records"):
                push(f"AH {r['ticker']} ${r['ext_price']} {r['move_pct']:+}% | {r['headline']}", f"AH{r['ticker']}{dt.date.today()}")
        st.caption(f"Last scan {dt.datetime.now(ET):%H:%M:%S} ET · next in {every}s")
        time.sleep(every)
        st.cache_data.clear()
        st.rerun()
