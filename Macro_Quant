"""Synoptic BI — Macro Quant page (Eric Rolex Joseph Macro Quantitative Engine).

A page of the Synoptic BI app: Streamlit lists any file in `pages/` in the sidebar.

Self-updating loop
  1. Pulls each indicator's history and official release calendar from FRED.
  2. When a release time passes, the page refetches, takes in the new actual and
     refits the model, then forecasts the next release automatically.
     If FRED has not posted the new number yet, the desk shows "Awaiting data"
     and stops signalling until it does.
  3. Nowcast: AR(1) with intercept, refit on the latest history (pandemic months
     Mar 2020 to Dec 2021 left out of estimation). Uncertainty sigma is the RMSE
     of the model's own rolling out-of-sample forecasts, so it widens when the
     model has been missing. Optional blend with a consensus figure you enter.
  4. Prices each Kalshi strike "release >= K" as P = 1 - Phi((K - step/2 - mu) / sigma),
     checks YES and NO asks, signals when edge clears the hurdle, sizes with
     fractional Kelly, and caps the whole release at the per-event risk limit.
  5. Scorecard: out-of-sample error, error of a naive "same as last" forecast,
     skill versus naive, and 1-sigma coverage, for every indicator.

Settings (Render > Environment, or .streamlit/secrets.toml):
  FRED_API_KEY   needed for automatic updates (the same key the Opportunity Engine uses)
  APP_PASSWORD   optional gate
Without FRED_API_KEY the page runs on the hand-entered fallback values in DESKS.

FOMC: the target range and meeting calendar update automatically, but the
rate-change nowcast (FOMC_MU, FOMC_SIGMA) is set by hand.

Not investment advice.
"""
import json
import math
import os
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    st.set_page_config(page_title="Macro Quant · Synoptic BI", layout="wide")
except Exception:
    pass

ET = ZoneInfo("America/New_York")
FRED = "https://api.stlouisfed.org/fred"
KALSHI_MARKETS = "https://api.elections.kalshi.com/trade-api/v2/markets"
FALLBACK_SPREAD = 0.02
COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"))


# ---------------------------------------------------------------- access
def _setting(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        try:
            val = str(st.secrets.get(name, ""))
        except Exception:
            val = ""
    return val


def _gate():
    pw = _setting("APP_PASSWORD")
    if not pw or st.session_state.get("mq_ok"):
        return
    st.title("Macro Quant")
    entered = st.text_input("Password", type="password", key="mq_pw")
    if st.button("Enter", key="mq_enter"):
        if entered == pw:
            st.session_state["mq_ok"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


_gate()


def _now() -> datetime:
    override = os.environ.get("MQ_NOW")  # for testing only
    if override:
        return datetime.fromisoformat(override).replace(tzinfo=ET)
    return datetime.now(ET)


NOW = _now()
FRED_KEY = _setting("FRED_API_KEY")


def _wide(fn, *args, **kwargs):
    """Full-width element on both newer (width="stretch") and older Streamlit."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


# ---------------------------------------------------------------- style
st.markdown("""<style>
.mq-head { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 18px 22px;
           border-radius: 10px; margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.1); }
.mq-title { font-size: 22px; font-weight: 800; letter-spacing: -0.4px; margin: 0; color: #ffffff !important; }
.mq-sub { font-size: 12px; color: #94a3b8 !important; margin-top: 4px; }
div[data-testid="stMetric"] { background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.2);
                              border-radius: 8px; padding: 8px 12px; }
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- settings
config = {}
if os.path.exists("config.json"):
    try:
        with open("config.json") as fh:
            config = json.load(fh)
    except Exception:
        config = {}

st.sidebar.markdown("### Macro Quant settings")
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=100.0,
                                   value=float(config.get("portfolio_bankroll", 10000.0)), step=500.0)
min_edge = st.sidebar.slider("Edge hurdle (probability points)", 0.010, 0.080,
                             float(config.get("edge_hurdle", 0.035)), 0.005, format="%.3f")
kelly_scale = st.sidebar.slider("Fractional Kelly (λ)", 0.05, 0.50,
                                float(config.get("kelly_fraction", 0.20)), 0.05)
max_event_cap = st.sidebar.slider("Risk cap per release (share of bankroll)", 0.01, 0.10,
                                  float(config.get("max_event_risk_cap", 0.05)), 0.01, format="%.2f")
model_weight = st.sidebar.slider("Weight on model vs consensus", 0.0, 1.0, 0.5, 0.1,
                                 help="Used only when you enter a consensus for the upcoming period.")
feed_mode = st.sidebar.radio("Quotes", ["Live Kalshi", "Illustrative only"], index=0)
if st.sidebar.button("Refresh data now"):
    st.cache_data.clear()

# ---------------------------------------------------------------- desks
# series/transform: how the FRED history becomes the number Kalshi settles on.
#   pct  = month-on-month % change of an index;  diff_k = monthly change in thousands
#   level = as published;  div1000 = published value / 1000
# mu0 / sigma0 / reading0 / release0 are fallbacks used only without FRED.
DESKS = [
    {"ticker": "KXCPICORE", "alt": ["CPICORE"], "name": "Core CPI (MoM)", "agency": "BLS", "unit": "%",
     "series": "CPILFESL", "transform": "pct", "freq": "M", "time": "08:30", "tick": 0.1, "fmt": "%.2f%%",
     "mu0": 0.2928, "sigma0": 0.1279, "reading0": "0.33%", "release0": "2026-10-14T08:30"},
    {"ticker": "KXCPI", "alt": ["CPI"], "name": "Headline CPI (MoM)", "agency": "BLS", "unit": "%",
     "series": "CPIAUCSL", "transform": "pct", "freq": "M", "time": "08:30", "tick": 0.1, "fmt": "%.2f%%",
     "mu0": 0.2450, "sigma0": 0.1650, "reading0": "0.22%", "release0": "2026-10-14T08:30"},
    {"ticker": "KXUSPPI", "alt": ["PPI"], "name": "PPI Final Demand (MoM)", "agency": "BLS", "unit": "%",
     "series": "PPIFIS", "transform": "pct", "freq": "M", "time": "08:30", "tick": 0.1, "fmt": "%.2f%%",
     "mu0": 0.1820, "sigma0": 0.2100, "reading0": "0.20%", "release0": "2026-10-15T08:30"},
    {"ticker": "KXPAYROLLS", "alt": ["PAYROLLS"], "name": "Nonfarm Payrolls", "agency": "BLS", "unit": "k jobs",
     "series": "PAYEMS", "transform": "diff_k", "freq": "M", "time": "08:30", "tick": 25.0, "fmt": "%+.0fk",
     "mu0": 172.5, "sigma0": 32.5, "reading0": "+142k", "release0": "2026-11-06T08:30"},
    {"ticker": "KXU3", "alt": ["U3"], "name": "U-3 Unemployment Rate", "agency": "BLS", "unit": "%",
     "series": "UNRATE", "transform": "level", "freq": "M", "time": "08:30", "tick": 0.1, "fmt": "%.1f%%",
     "mu0": 4.12, "sigma0": 0.11, "reading0": "4.20%", "release0": "2026-11-06T08:30"},
    {"ticker": "KXJOBLESSCLAIMS", "alt": ["JOBLESSCLAIMS"], "name": "Initial Jobless Claims", "agency": "DOL",
     "unit": "k claims", "series": "ICSA", "transform": "div1000", "freq": "W", "time": "08:30", "tick": 5.0,
     "fmt": "%.1fk", "mu0": 218.4, "sigma0": 7.8, "reading0": "225.0k", "release0": "weekly"},
    {"ticker": "KXPCECORE", "alt": ["KXPCE", "PCE"], "name": "Core PCE Prices (MoM)", "agency": "BEA", "unit": "%",
     "series": "PCEPILFE", "transform": "pct", "freq": "M", "time": "08:30", "tick": 0.05, "fmt": "%.2f%%",
     "mu0": 0.2380, "sigma0": 0.0810, "reading0": "0.26%", "release0": "2026-10-30T08:30"},
    {"ticker": "KXGDP", "alt": ["GDP"], "name": "Real GDP (QoQ annualized)", "agency": "BEA", "unit": "%",
     "series": "A191RL1Q225SBEA", "transform": "level", "freq": "Q", "time": "08:30", "tick": 0.5,
     "fmt": "%+.1f%%", "mu0": 2.85, "sigma0": 0.62, "reading0": "+3.00%", "release0": "2026-10-29T08:30"},
    {"ticker": "KXUSRETAIL", "alt": ["RETAIL"], "name": "Advance Retail Sales (MoM)", "agency": "Census",
     "unit": "%", "series": "RSAFS", "transform": "pct", "freq": "M", "time": "08:30", "tick": 0.1,
     "fmt": "%+.2f%%", "mu0": 0.41, "sigma0": 0.19, "reading0": "+0.10%", "release0": "2026-10-16T08:30"},
    {"ticker": "KXHOUSINGSTART", "alt": ["HOUSING"], "name": "Housing Starts (annualized)", "agency": "Census",
     "unit": "M units", "series": "HOUST", "transform": "div1000", "freq": "M", "time": "08:30", "tick": 0.04,
     "fmt": "%.3fM", "mu0": 1.365, "sigma0": 0.048, "reading0": "1.356M", "release0": "2026-10-19T08:30"},
    {"ticker": "KXJOLTSOPEN", "alt": ["JOLTS"], "name": "JOLTS Job Openings", "agency": "BLS",
     "unit": "M openings", "series": "JTSJOL", "transform": "div1000", "freq": "M", "time": "10:00",
     "tick": 0.2, "fmt": "%.2fM", "mu0": 7.780, "sigma0": 0.185, "reading0": "7.673M",
     "release0": "2026-11-03T10:00"},
    {"ticker": "KXFED", "alt": ["FED"], "name": "FOMC Rate Change", "agency": "Fed", "unit": "bps",
     "series": "DFEDTARU", "transform": "fomc", "freq": "FOMC", "time": "14:00", "tick": 25.0, "step": 25.0,
     "fmt": "%+.0f bps", "mu0": -25.0, "sigma0": 8.5, "reading0": "4.75% to 5.00%", "release0": "fomc"},
]

# FOMC: decision days (statement at 14:00 ET), federalreserve.gov. 2027 is tentative.
FOMC_DATES = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16",
              "2026-10-28", "2026-12-09", "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09",
              "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08"]
FOMC_MU, FOMC_SIGMA = -25.0, 8.5  # set by hand before each meeting (bps change in the upper bound)

# Consensus for the upcoming period, if you have one: ticker -> (period "YYYY-MM", value).
# A consensus for any other period is ignored, so stale entries never leak in.
CONSENSUS = {
    "KXPCECORE": ("2026-09", 0.25),
    "KXPAYROLLS": ("2026-10", 160.0),
    "KXU3": ("2026-10", 4.1),
    "KXJOLTSOPEN": ("2026-09", 7.7),
}


# ---------------------------------------------------------------- maths
def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def prob_at_least(k, mu, sigma, step=0.0) -> float:
    """P(release >= k). Step-valued outcomes (FOMC, 25 bps) shift the strike half a step."""
    return 1.0 - phi((k - step / 2.0 - mu) / sigma)


# ---------------------------------------------------------------- FRED
def _fred(path, **params):
    params.update(api_key=FRED_KEY, file_type="json")
    r = requests.get(f"{FRED}/{path}", params=params, timeout=12)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=900, show_spinner=False)
def fred_observations(series: str, start: str, release_token: str):
    """release_token changes when a release time passes, which forces a fresh pull."""
    obs = _fred("series/observations", series_id=series, observation_start=start)["observations"]
    s = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in obs if o["value"] not in (".", "")})
    return s.sort_index()


@st.cache_data(ttl=900, show_spinner=False)
def fred_last_updated(series: str, release_token: str):
    info = _fred("series", series_id=series)["seriess"][0]
    ts = pd.Timestamp(info["last_updated"])
    return (ts.tz_convert(ET) if ts.tzinfo else ts.tz_localize(ET)).to_pydatetime()


@st.cache_data(ttl=86400, show_spinner=False)
def fred_release_dates(series: str):
    rid = _fred("series/release", series_id=series)["releases"][0]["id"]
    start = (date.today() - timedelta(days=400)).isoformat()
    out = _fred("release/dates", release_id=rid, realtime_start=start, realtime_end="9999-12-31",
                include_release_dates_with_no_data="true", sort_order="asc", limit=1000)
    return sorted({d["date"] for d in out.get("release_dates", [])})


def schedule(desk):
    """(last release datetime or None, next release datetime or None) in ET."""
    hh, mm = map(int, desk["time"].split(":"))
    if desk["freq"] == "FOMC":
        days = FOMC_DATES
    else:
        days = fred_release_dates(desk["series"])
    stamps = [datetime.fromisoformat(d).replace(hour=hh, minute=mm, tzinfo=ET) for d in days]
    past = [s for s in stamps if s <= NOW]
    future = [s for s in stamps if s > NOW]
    return (past[-1] if past else None), (future[0] if future else None)


def transform(raw: pd.Series, how: str) -> pd.Series:
    if how == "pct":
        return (raw.pct_change() * 100).dropna()
    if how == "diff_k":
        return raw.diff().dropna()
    if how == "div1000":
        return raw / 1000.0
    return raw


# ---------------------------------------------------------------- model
WINDOW = {"M": 120, "W": 260, "Q": 60}
BACKTEST = {"M": 24, "W": 52, "Q": 12}


def _ar1(y: pd.Series):
    df = pd.DataFrame({"y": y, "x": y.shift(1)}).dropna()
    df = df[(df.index < COVID[0]) | (df.index > COVID[1])]
    if len(df) < 8:
        return None
    X = np.column_stack([np.ones(len(df)), df["x"].values])
    beta, *_ = np.linalg.lstsq(X, df["y"].values, rcond=None)
    beta[1] = float(np.clip(beta[1], -0.98, 0.98))
    return beta


def fit_and_backtest(y: pd.Series, freq: str):
    w, n_bt = WINDOW[freq], BACKTEST[freq]
    rows = []
    for i in range(max(len(y) - n_bt, 10), len(y)):
        hist = y.iloc[max(0, i - w):i]
        beta = _ar1(hist)
        if beta is None:
            continue
        f = beta[0] + beta[1] * hist.iloc[-1]
        rows.append({"date": y.index[i], "forecast": f, "actual": y.iloc[i], "naive": hist.iloc[-1]})
    bt = pd.DataFrame(rows)
    beta = _ar1(y.iloc[-w:])
    mu = float(beta[0] + beta[1] * y.iloc[-1]) if beta is not None else float(y.iloc[-1])
    if len(bt):
        err = bt["actual"] - bt["forecast"]
        rmse = float(np.sqrt((err ** 2).mean()))
        bt["error"] = err
        bt["inside_1sigma"] = err.abs() <= rmse
    else:
        rmse = float(y.diff().std())
    return mu, rmse, bt, beta


def next_period(last: pd.Timestamp, freq: str) -> pd.Timestamp:
    if freq == "M":
        return last + pd.DateOffset(months=1)
    if freq == "Q":
        return last + pd.DateOffset(months=3)
    return last + pd.Timedelta(days=7)


def period_label(ts: pd.Timestamp, freq: str) -> str:
    if freq == "M":
        return ts.strftime("%b %Y")
    if freq == "Q":
        return f"Q{(ts.month - 1) // 3 + 1} {ts.year}"
    return "week of " + ts.strftime("%b %d")


def consensus_for(ticker, period: pd.Timestamp, freq):
    edits = st.session_state.get("mq_cons_table")
    entry = None
    if isinstance(edits, pd.DataFrame):
        hit = edits[edits["Ticker"] == ticker]
        if len(hit) and pd.notna(hit["Consensus"].iloc[0]) and str(hit["Period (YYYY-MM)"].iloc[0]).strip():
            entry = (str(hit["Period (YYYY-MM)"].iloc[0]).strip(), float(hit["Consensus"].iloc[0]))
    entry = entry or CONSENSUS.get(ticker)
    if not entry or freq not in ("M", "Q"):
        return None
    return entry[1] if entry[0] == period.strftime("%Y-%m") else None


# ---------------------------------------------------------------- build each desk
def build_state(desk):
    """Everything the page needs for one indicator. Never raises."""
    st_ = {"desk": desk, "auto": False, "status": "Upcoming", "note": "", "consensus": None,
           "backtest": pd.DataFrame(), "history": pd.Series(dtype=float), "beta": None}
    if FRED_KEY:
        try:
            last_rel, next_rel = schedule(desk)
            token = last_rel.isoformat() if last_rel else "none"
            if last_rel and NOW - last_rel < timedelta(days=1):
                # first day after a release: re-check FRED every 5 minutes until the number lands
                token += f"|{int(NOW.timestamp() // 300)}"
            raw = fred_observations(desk["series"], "2009-01-01", token)
            updated = fred_last_updated(desk["series"], token)
            if desk["freq"] == "FOMC":
                lower = fred_observations("DFEDTARL", "2024-01-01", token)
                upper = raw
                prev = upper[upper.index < pd.Timestamp(last_rel.date())] if last_rel else upper
                change = (upper.iloc[-1] - prev.iloc[-1]) * 100 if len(prev) else 0.0
                st_.update(auto=True, mu=FOMC_MU, sigma=FOMC_SIGMA, model_mu=FOMC_MU,
                           reading=f"{lower.iloc[-1]:.2f}% to {upper.iloc[-1]:.2f}% "
                                   f"(last meeting {change:+.0f} bps)",
                           period="next meeting", release=next_rel, last_release=last_rel,
                           note="Target range and calendar update automatically; the rate-change "
                                "nowcast is set by hand (FOMC_MU).")
                return st_
            y = transform(raw, desk["transform"])
            mu, rmse, bt, beta = fit_and_backtest(y, desk["freq"])
            sigma = max(rmse, desk["tick"] / 4)
            period = next_period(y.index[-1], desk["freq"])
            cons = consensus_for(desk["ticker"], period, desk["freq"])
            blended = model_weight * mu + (1 - model_weight) * cons if cons is not None else mu
            awaiting = bool(last_rel and updated < last_rel and NOW - last_rel < timedelta(days=4))
            st_.update(auto=True, mu=blended, model_mu=mu, sigma=sigma, consensus=cons, backtest=bt,
                       history=y, beta=beta, period=period_label(period, desk["freq"]),
                       reading=f"{desk['fmt'] % y.iloc[-1]} ({period_label(y.index[-1], desk['freq'])})",
                       release=next_rel, last_release=last_rel, updated=updated)
            if awaiting:
                st_.update(status="Awaiting data",
                           note=f"Released {last_rel:%b %d %H:%M} ET; FRED has not posted it yet. "
                                "The page will refit as soon as it does.")
            return st_
        except Exception as exc:  # fall through to hand-entered values
            st_["note"] = f"Automatic update unavailable ({type(exc).__name__}); using fallback values."

    # fallback: hand-entered values
    rel = desk["release0"]
    hh, mm = map(int, desk["time"].split(":"))
    if rel == "weekly":
        days = (3 - NOW.weekday()) % 7
        dt = (NOW + timedelta(days=days)).replace(hour=hh, minute=mm, second=0, microsecond=0)
        dt = dt if dt > NOW else dt + timedelta(days=7)
    elif rel == "fomc":
        dt = next((datetime.fromisoformat(d).replace(hour=hh, minute=mm, tzinfo=ET)
                   for d in FOMC_DATES if datetime.fromisoformat(d).replace(hour=hh, minute=mm, tzinfo=ET) > NOW),
                  None)
    else:
        dt = datetime.fromisoformat(rel).replace(tzinfo=ET)
    st_.update(mu=desk["mu0"], model_mu=desk["mu0"], sigma=desk["sigma0"], reading=desk["reading0"],
               period="–", release=dt, last_release=None)
    if dt is not None and dt <= NOW:
        st_["status"] = "Released"
        st_["note"] = (st_["note"] + " " if st_["note"] else "") + \
            "Add FRED_API_KEY to update automatically, or edit mu0/sigma0/release0 by hand."
    return st_


# ---------------------------------------------------------------- quotes
def _ask(item, side):
    v = item.get(f"{side}_ask_dollars")
    if v not in (None, ""):
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    v = item.get(f"{side}_ask")
    if isinstance(v, (int, float)) and v > 0:
        return v / 100.0
    return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_kalshi(tickers: tuple):
    for t in tickers:
        try:
            r = requests.get(KALSHI_MARKETS, params={"series_ticker": t, "status": "open", "limit": 200},
                             timeout=4)
            if r.status_code == 200:
                markets = r.json().get("markets", [])
                if markets:
                    return t, markets
        except (requests.RequestException, ValueError):
            pass
    return None, []


def live_ladder(desk):
    """Strikes and asks from the nearest open Kalshi event with 'above' strikes."""
    _, markets = fetch_kalshi(tuple([desk["ticker"]] + desk["alt"]))
    events = {}
    for m in markets:
        if m.get("strike_type") not in ("greater", "greater_or_equal"):
            continue
        k, ya, na = m.get("floor_strike"), _ask(m, "yes"), _ask(m, "no")
        if k is None or ya is None or not 0 < ya < 1:
            continue
        ev = events.setdefault(m.get("event_ticker", ""), {"close": m.get("close_time", ""), "rows": []})
        ev["rows"].append((float(k), ya, na if na and 0 < na < 1 else None))
    events = {e: v for e, v in events.items() if len(v["rows"]) >= 2}
    if not events:
        return None
    ev_ticker = min(events, key=lambda e: events[e]["close"] or "9999")
    return {"source": f"Kalshi {ev_ticker}", "rows": sorted(events[ev_ticker]["rows"]), "live": True}


def illustrative_ladder(desk, state):
    """Strikes around the nowcast, priced by a 'market' centred on consensus (or the last
    reading) with a slightly wider spread. Shows how the engine reacts; not real prices."""
    tick, mu = desk["tick"], state["mu"]
    centre = round(mu / tick) * tick
    strikes = [round(centre + i * tick, 6) for i in range(-2, 3)]
    hist = state["history"]
    anchor = state["consensus"] if state["consensus"] is not None else (
        float(hist.iloc[-1]) if len(hist) else desk["mu0"])
    rows = []
    for k in strikes:
        a = float(np.clip(round(prob_at_least(k, anchor, state["sigma"] * 1.15, desk.get("step", 0.0)), 2),
                          0.01, 0.99))
        rows.append((k, a, min(0.99, round(1 - a + FALLBACK_SPREAD, 2))))
    return {"source": "Illustrative quotes (not live)", "rows": rows, "live": False}


# ---------------------------------------------------------------- signals
def kalshi_fee(contracts: int, price: float) -> float:
    """Kalshi taker fee: 7% x C x P x (1-P), rounded up to the cent."""
    if contracts <= 0:
        return 0.0
    return math.ceil(0.07 * contracts * price * (1 - price) * 100 - 1e-9) / 100


def evaluate(desk, state, ladder):
    mu, sigma, step = state["mu"], state["sigma"], desk.get("step", 0.0)
    blocked = state["status"] != "Upcoming" or state["release"] is None
    cands = []
    for k, yes_ask, no_ask in ladder["rows"]:
        p = prob_at_least(k, mu, sigma, step)
        sides = [("YES", p, yes_ask)] + ([("NO", 1 - p, no_ask)] if no_ask is not None else [])
        side, prob, price = max(sides, key=lambda s: s[1] - s[2])
        edge = prob - price
        raw = kelly_scale * edge / (1 - price) if edge >= min_edge and price < 1 else 0.0
        cands.append({"k": k, "p_yes": p, "yes_ask": yes_ask, "no_ask": no_ask, "side": side,
                      "price": price, "edge": edge, "raw": raw})
    total = sum(c["raw"] for c in cands)
    scale = max_event_cap / total if total > max_event_cap else 1.0
    for c in cands:
        frac = 0.0 if blocked else c["raw"] * scale
        contracts = int((frac * bankroll) // c["price"]) if frac > 0 else 0
        fee = kalshi_fee(contracts, c["price"])
        if blocked:
            state_lbl = state["status"].upper()
        elif contracts:
            state_lbl = f"BUY {c['side']}" + ("" if ladder["live"] else " (SIM)")
        else:
            state_lbl = "PASS"
        c.update(contracts=contracts, risk=round(contracts * c["price"], 2), fee=fee, state=state_lbl,
                 win=round(contracts * (1 - c["price"]) - fee, 2) if contracts else 0.0,
                 ev=round(contracts * c["edge"] - fee, 2) if contracts else 0.0)
    return cands


def ladder_table(desk, rows):
    f = desk["fmt"]
    return pd.DataFrame([{
        "Strike": f"≥ {f % r['k']}",
        "Model P(YES)": f"{r['p_yes']*100:.1f}%",
        "YES ask": f"{r['yes_ask']*100:.0f}¢",
        "NO ask": f"{r['no_ask']*100:.0f}¢" if r["no_ask"] is not None else "–",
        "Best side": r["side"],
        "Edge": f"{r['edge']*100:+.1f} pp",
        "Signal": r["state"],
        "Contracts": f"{r['contracts']:,}" if r["contracts"] else "–",
        "Risk ($)": f"{r['risk']:,.2f}" if r["contracts"] else "–",
        "Fee ($)": f"{r['fee']:,.2f}" if r["contracts"] else "–",
        "Profit if right ($)": f"{r['win']:,.2f}" if r["contracts"] else "–",
        "Expected value ($)": f"{r['ev']:+,.2f}" if r["contracts"] else "–",
    } for r in rows])


# ---------------------------------------------------------------- page
st.markdown(f"""<div class="mq-head">
<div class="mq-title">Macro Quantitative Engine</div>
<div class="mq-sub">Eric Rolex Joseph · Self-updating nowcasts of 12 US releases, refit after every print and
priced against Kalshi event contracts · {NOW:%b %d, %Y %H:%M} ET</div>
</div>""", unsafe_allow_html=True)

if not FRED_KEY:
    st.warning("FRED_API_KEY is not set, so the page uses hand-entered values and cannot update itself. "
               "Add the key in Render > Environment (the same one the Opportunity Engine uses).")

with st.expander("Consensus inputs (optional)"):
    st.caption("Enter the street consensus for the upcoming period. It is blended with the model using the "
               "sidebar weight, and ignored once that period has been released. Edits last for this visit; "
               "to keep them, add them to CONSENSUS in the code.")
    base = pd.DataFrame([{"Ticker": d["ticker"], "Indicator": d["name"],
                          "Period (YYYY-MM)": CONSENSUS.get(d["ticker"], ("", None))[0],
                          "Consensus": CONSENSUS.get(d["ticker"], ("", None))[1]}
                         for d in DESKS if d["freq"] in ("M", "Q")])
    st.session_state["mq_cons_table"] = st.data_editor(base, hide_index=True, disabled=["Ticker", "Indicator"],
                                                       key="mq_cons_editor")

with st.spinner("Updating indicators…"):
    states = {d["ticker"]: build_state(d) for d in DESKS}
results = {}
for d in DESKS:
    s = states[d["ticker"]]
    lad = live_ladder(d) if feed_mode == "Live Kalshi" else None
    lad = lad or illustrative_ladder(d, s)
    results[d["ticker"]] = (lad, evaluate(d, s, lad))

n_auto = sum(1 for s in states.values() if s["auto"])
n_live = sum(1 for v in results.values() if v[0]["live"])
n_signals = sum(1 for v in results.values() for r in v[1] if r["contracts"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Bankroll", f"${bankroll:,.0f}")
c2.metric("Risk cap per release", f"${bankroll*max_event_cap:,.0f}", f"{max_event_cap*100:.0f}% of bankroll",
          delta_color="off")
c3.metric("Auto-updating desks", f"{n_auto} of {len(DESKS)}")
c4.metric("Live quote feeds", f"{n_live} of {len(DESKS)}")
c5.metric("Open signals" if n_live else "Signals (simulated)", f"{n_signals}")

if n_live == 0:
    st.info("No live Kalshi quotes right now, so ladders use illustrative quotes. Signals marked SIM "
            "show how the engine would act; they are not tradable prices.")

tab_matrix, tab_desk, tab_score, tab_mc, tab_ledger, tab_method = st.tabs(
    ["Indicator matrix", "Desk detail", "Model scorecard", "Monte Carlo", "Paper ledger", "Method"])

with tab_matrix:
    summary = []
    for d in DESKS:
        s = states[d["ticker"]]
        lad, rows = results[d["ticker"]]
        actionable = [r for r in rows if r["contracts"]]
        if s["status"] != "Upcoming":
            action = s["status"]
        elif actionable:
            b = max(actionable, key=lambda r: r["edge"])
            action = (f"{b['state']} ≥ {d['fmt'] % b['k']}: {b['contracts']:,} contracts, "
                      f"${b['risk']:,.2f} at risk, edge {b['edge']*100:+.1f} pp")
        else:
            action = "Pass: no strike clears the hurdle"
        summary.append({
            "Ticker": d["ticker"], "Indicator": d["name"],
            "Data": "Auto (FRED)" if s["auto"] else "Manual",
            "Latest actual": s["reading"],
            "Forecasting": s["period"],
            "Nowcast (μ)": d["fmt"] % s["mu"],
            "σ": f"±{s['sigma']:.3g}",
            "Consensus": d["fmt"] % s["consensus"] if s["consensus"] is not None else "–",
            "Next release (ET)": s["release"].strftime("%a %b %d, %H:%M") if s["release"] else "–",
            "Quotes": "Live" if lad["live"] else "Illustrative",
            "Action": action,
        })
    _wide(st.dataframe, pd.DataFrame(summary), hide_index=True)
    st.caption("After each release the desk refits on the new number and rolls to the next one. Each release "
               "is sized as one event: all its strikes together stay under the risk cap.")

with tab_desk:
    labels = [f"{d['ticker']} · {d['name']}" for d in DESKS]
    choice = st.selectbox("Indicator", labels)
    d = DESKS[labels.index(choice)]
    s = states[d["ticker"]]
    lad, rows = results[d["ticker"]]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest actual", s["reading"])
    m2.metric(f"Nowcast for {s['period']}", d["fmt"] % s["mu"], f"σ = ±{s['sigma']:.3g}", delta_color="off")
    m3.metric("Model only / consensus", d["fmt"] % s["model_mu"],
              (d["fmt"] % s["consensus"]) if s["consensus"] is not None else "no consensus", delta_color="off")
    m4.metric("Next release (ET)", s["release"].strftime("%b %d, %H:%M") if s["release"] else "–",
              s["status"], delta_color="off")
    st.caption(f"Quotes: {lad['source']}" + (f" · {s['note']}" if s["note"] else ""))
    if s["status"] != "Upcoming":
        st.warning(s["note"] or "This desk is not trading right now.")

    _wide(st.dataframe, ladder_table(d, rows), hide_index=True)

    ks = [r["k"] for r in rows]
    lo = min(s["mu"] - 3.5 * s["sigma"], min(ks))
    hi = max(s["mu"] + 3.5 * s["sigma"], max(ks))
    xs = np.linspace(lo, hi, 200)
    curve = pd.DataFrame({"x": xs, "p": [prob_at_least(x, s["mu"], s["sigma"], d.get("step", 0.0)) for x in xs]})
    pts = pd.DataFrame({"x": ks, "p": [r["yes_ask"] for r in rows],
                        "label": [f"≥ {d['fmt'] % r['k']}: market {r['yes_ask']*100:.0f}¢, "
                                  f"model {r['p_yes']*100:.0f}%" for r in rows]})
    line = alt.Chart(curve).mark_line(strokeWidth=2.5).encode(
        x=alt.X("x:Q", title=f"{d['name']} ({d['unit']})"),
        y=alt.Y("p:Q", title="Probability of release ≥ strike", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])))
    dots = alt.Chart(pts).mark_circle(size=90, color="#ef4444").encode(x="x:Q", y="p:Q", tooltip=["label"])
    mu_rule = alt.Chart(pd.DataFrame({"x": [s["mu"]]})).mark_rule(strokeDash=[5, 4], color="#f59e0b").encode(x="x:Q")
    _wide(st.altair_chart, (line + dots + mu_rule).properties(height=280))
    st.caption("Line: model probability. Red dots: market YES ask. A dot below the line is cheap YES; "
               "far above it, NO is the cheaper side. Dashed line: the nowcast.")

    bt = s["backtest"]
    if len(bt):
        st.markdown("**How the model has done on recent releases** (each forecast made with only the data "
                    "available before that release)")
        plot = bt.melt(id_vars="date", value_vars=["actual", "forecast"], var_name="Series", value_name="v")
        _wide(st.altair_chart, alt.Chart(plot).mark_line(point=True).encode(
            x=alt.X("date:T", title=None), y=alt.Y("v:Q", title=d["unit"], scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top"))).properties(height=240))
        show = bt.tail(12).iloc[::-1].copy()
        show["Period"] = [period_label(t, d["freq"]) for t in show["date"]]
        show = show[["Period", "forecast", "actual", "error", "inside_1sigma"]].rename(columns={
            "forecast": "Forecast", "actual": "Actual", "error": "Error", "inside_1sigma": "Within ±1σ"})
        _wide(st.dataframe, show.round(3), hide_index=True)

with tab_score:
    score = []
    for d in DESKS:
        bt = states[d["ticker"]]["backtest"]
        if not len(bt):
            continue
        mae = (bt["actual"] - bt["forecast"]).abs().mean()
        naive = (bt["actual"] - bt["naive"]).abs().mean()
        score.append({"Indicator": d["name"], "Releases tested": len(bt),
                      "Model error (MAE)": round(mae, 3), "Naive error (MAE)": round(naive, 3),
                      "Skill vs naive": f"{(1 - mae / naive) * 100:+.0f}%" if naive else "–",
                      "Bias": round((bt["actual"] - bt["forecast"]).mean(), 3),
                      "Within ±1σ": f"{bt['inside_1sigma'].mean()*100:.0f}%"})
    if score:
        _wide(st.dataframe, pd.DataFrame(score), hide_index=True)
        st.caption("Out-of-sample: every forecast uses only data before that release. Naive = 'same as last "
                   "reading'. Positive skill means the model beat naive. About 68% within ±1σ means σ is well "
                   "calibrated. FRED serves revised data, so early prints may differ slightly from what Kalshi "
                   "settled on.")
    else:
        st.info("The scorecard needs FRED_API_KEY.")

with tab_mc:
    st.markdown("Simulates the bankroll over a year of trades, risking the per-release cap each time.")
    r1, r2, r3, r4 = st.columns(4)
    n_trades = r1.slider("Trades per year", 12, 100, 48, step=4)
    win_rate = r2.slider("Assumed win rate (%)", 30, 95, 60, step=5) / 100
    avg_price = r3.slider("Average entry price ($)", 0.10, 0.90, 0.48, step=0.01)
    n_paths = r4.select_slider("Simulated paths", [1000, 5000, 10000], value=5000)

    payoff = (1 - avg_price) / avg_price - 0.07 * (1 - avg_price)
    f = max_event_cap
    rng = np.random.default_rng(7)
    wins = rng.random((n_paths, n_trades)) < win_rate
    paths = bankroll * np.cumprod(np.where(wins, 1 + f * payoff, 1 - f), axis=1)
    ending = paths[:, -1]
    peaks = np.maximum.accumulate(np.hstack([np.full((n_paths, 1), bankroll), paths]), axis=1)[:, 1:]
    max_dd = ((peaks - paths) / peaks).max(axis=1)
    breakeven = 1 / (1 + payoff)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Median ending bankroll", f"${np.median(ending):,.0f}")
    s2.metric("5th to 95th percentile", f"${np.percentile(ending, 5):,.0f} to ${np.percentile(ending, 95):,.0f}")
    s3.metric("Chance of ending below start", f"{(ending < bankroll).mean()*100:.1f}%")
    s4.metric("Median max drawdown", f"{np.median(max_dd)*100:.1f}%")

    band = pd.DataFrame({"Trade": np.arange(1, n_trades + 1),
                         "p5": np.percentile(paths, 5, axis=0), "p50": np.percentile(paths, 50, axis=0),
                         "p95": np.percentile(paths, 95, axis=0)})
    area = alt.Chart(band).mark_area(opacity=0.25).encode(
        x="Trade:Q", y=alt.Y("p5:Q", title="Bankroll ($)"), y2="p95:Q")
    med = alt.Chart(band).mark_line(strokeWidth=2.5).encode(x="Trade:Q", y="p50:Q")
    start = alt.Chart(pd.DataFrame({"y": [bankroll]})).mark_rule(strokeDash=[4, 4], color="#94a3b8").encode(y="y:Q")
    _wide(st.altair_chart, (area + med + start).properties(height=300))
    st.caption(f"Shaded: 5th to 95th percentile; line: median. At a {avg_price:.2f} entry price the "
               f"break-even win rate after fees is {breakeven*100:.1f}%. Win rate and price are assumptions, "
               "not results.")

with tab_ledger:
    st.markdown("**Paper-trading replay** (no real money).")
    _wide(st.dataframe, pd.DataFrame([
        {"Order": "ORD-2026-0914-01", "Contract": "KXCPICORE ≥ 0.20%", "Side": "BUY YES", "Qty": 384,
         "Fill": "$0.68", "Risk": "$261.12", "Fee": "$5.85", "Result": "Won +$117.03"},
        {"Order": "ORD-2026-0914-02", "Contract": "KXCPICORE ≥ 0.30%", "Side": "BUY YES", "Qty": 433,
         "Fill": "$0.38", "Risk": "$164.54", "Fee": "$7.15", "Result": "Won +$261.31"},
        {"Order": "ORD-2026-0914-03", "Contract": "KXCPICORE ≥ 0.40%", "Side": "BUY YES", "Qty": 529,
         "Fill": "$0.14", "Risk": "$74.06", "Fee": "$4.46", "Result": "Lost −$78.52"},
    ]), hide_index=True)

with tab_method:
    st.markdown(f"""
**Self-updating.** History and release calendars come from FRED. When a release time passes, the page pulls
the data again, adds the new actual, refits, and forecasts the next release. Until FRED posts the number the
desk shows *Awaiting data* and does not signal. Data is otherwise refreshed every 15 minutes.

**Nowcast.** AR(1) with intercept on the last 10 years (monthly), 5 years (weekly) or 15 years (quarterly) of
the series, leaving March 2020 to December 2021 out of estimation. If you enter a consensus for the upcoming
period, μ = {model_weight:.1f} × model + {1 - model_weight:.1f} × consensus.

**Uncertainty.** σ is the root-mean-square error of the model's own out-of-sample forecasts over the last
24 months (52 weeks, 12 quarters), so it widens after misses and narrows after hits.

**Pricing.** P(release ≥ K) = 1 − Φ((K − μ) / σ); a NO contract is 1 − P. The FOMC desk moves in 25 bp steps,
so its strikes shift half a step. Its rate nowcast is set by hand (FOMC_MU).

**Edge and sizing.** Both sides are checked against their asks; a signal needs edge ≥ {min_edge*100:.1f} pp.
Fractional Kelly f = λ × edge ÷ (1 − price), λ = {kelly_scale:.2f}, scaled so a release's strikes together stay
under {max_event_cap*100:.0f}% of bankroll. Kalshi taker fee 7% × C × P × (1 − P), rounded up to the cent.

**Limits.** AR(1) is a baseline, not a full nowcast; it does not use other indicators released in the same
month. FRED serves revised data, so the scorecard is slightly flattering for series that revise. Normality is
an assumption. Not investment advice.
""")
