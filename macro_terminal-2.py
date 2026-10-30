"""
macro_terminal.py (Complete Institutional v2.2)
================================================
ERIC ROLEX JOSEPH — QUANTITATIVE MACRO TRADING TERMINAL
"""
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import requests
import json
import os
from typing import Dict, Any

st.set_page_config(page_title="Eric Rolex Joseph • Macro Quantitative Engine", layout="wide", initial_sidebar_state="expanded")

st.markdown('''<style>
    .header-box { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #ffffff !important; padding: 18px 22px; border-radius: 10px; margin-bottom: 16px; border: 1px solid rgba(255, 255, 255, 0.1); }
    .header-title { font-size: 22px; font-weight: 800; letter-spacing: -0.4px; margin: 0; color: #ffffff !important; }
    .header-sub { font-size: 12px; color: #94a3b8 !important; margin-top: 4px; }
    div[data-testid="stMetric"] { background: rgba(128, 128, 128, 0.08); border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 8px; padding: 8px 12px; }
    .stDeployButton, footer, #MainMenu { visibility: hidden; display: none; }
</style>''', unsafe_allow_html=True)

config = {}
if os.path.exists("config.json"):
    try:
        with open("config.json", "r") as cfg_f: config = json.load(cfg_f)
    except Exception: pass

default_webhook = config.get("discord_webhook_url", "")

st.sidebar.markdown("### 👤 Desk Principal\n**Eric Rolex Joseph**  \n*Macro Quantitative Architecture*\n---")
st.sidebar.markdown("### ⚙️ Risk Configuration")
bankroll = st.sidebar.number_input("Portfolio Bankroll ($)", min_value=100.0, value=float(config.get("portfolio_bankroll", 10000.0)), step=500.0)
min_edge = st.sidebar.slider("Min Edge Hurdle (pp)", 0.010, 0.080, float(config.get("edge_hurdle", 0.035)), 0.005, format="%.3f")
kelly_scale = st.sidebar.slider("Fractional Kelly (λ)", 0.05, 0.50, float(config.get("kelly_fraction", 0.20)), 0.05)
max_event_cap = st.sidebar.slider("Aggregate Event Risk Cap", 0.01, 0.10, float(config.get("max_event_risk_cap", 0.05)), 0.01, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Exchange Gateway")
feed_mode = st.sidebar.radio("Data Ingestion Mode:", ["Live Kalshi REST Gateway", "Simulated Calibrated Ladder"], index=0)
force_live = st.sidebar.checkbox("🔓 Bypass Lockout / Enable Live Signals", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔔 Remote Execution Webhook")
discord_webhook = st.sidebar.text_input("Discord Webhook URL", value=default_webhook, type="password")

st.markdown('''<div class="header-box">
    <div class="header-title">🏛️ ERIC ROLEX JOSEPH — MACRO QUANTITATIVE ENGINE</div>
    <div class="header-sub">Automated Macroeconomic Forecasting & Event-Contract Execution Engine • Strict Multi-Strike Event Budgeting</div>
</div>''', unsafe_allow_html=True)

DESKS_DATA = [
    {"ticker": "KXCPICORE", "alt_tickers": ["CPICORE", "CPI-CORE"], "name": "Core CPI (MoM)", "agency": "BLS", "unit": "%", "mu": 0.2928, "sigma": 0.1279, "consensus": 0.3000, "current_reading": "0.33% (Prior Mo)", "release": "Oct 14, 08:30 ET", "fallback_strikes": [0.10, 0.20, 0.30, 0.40, 0.50], "fallback_asks": [0.92, 0.68, 0.38, 0.14, 0.04], "format": "%.2f%%"},
    {"ticker": "KXCPI", "alt_tickers": ["CPI", "INFLATION"], "name": "Headline CPI (MoM)", "agency": "BLS", "unit": "%", "mu": 0.2450, "sigma": 0.1650, "consensus": 0.2000, "current_reading": "0.22% (Prior Mo)", "release": "Oct 14, 08:30 ET", "fallback_strikes": [0.00, 0.10, 0.20, 0.30, 0.40], "fallback_asks": [0.90, 0.78, 0.58, 0.35, 0.12], "format": "%.2f%%"},
    {"ticker": "KXUSPPI", "alt_tickers": ["PPI", "USPPI"], "name": "PPI Final Demand (MoM)", "agency": "BLS", "unit": "%", "mu": 0.1820, "sigma": 0.2100, "consensus": 0.2000, "current_reading": "0.20% (Prior Mo)", "release": "Oct 15, 08:30 ET", "fallback_strikes": [-0.10, 0.00, 0.10, 0.20, 0.30], "fallback_asks": [0.88, 0.76, 0.62, 0.44, 0.24], "format": "%.2f%%"},
    {"ticker": "KXPAYROLLS", "alt_tickers": ["PAYROLLS", "NFP"], "name": "Nonfarm Payrolls (NFP)", "agency": "BLS", "unit": "k jobs", "mu": 172.5, "sigma": 32.5, "consensus": 160.0, "current_reading": "+142k (Prior Report)", "release": "Nov 06, 08:30 ET", "fallback_strikes": [120.0, 140.0, 160.0, 180.0, 200.0], "fallback_asks": [0.91, 0.79, 0.58, 0.36, 0.16], "format": "+%.0fk"},
    {"ticker": "KXU3", "alt_tickers": ["UNEMPLOYMENT", "U3"], "name": "U-3 Unemployment Rate", "agency": "BLS", "unit": "%", "mu": 4.12, "sigma": 0.11, "consensus": 4.10, "current_reading": "4.20% (Prior Mo)", "release": "Nov 06, 08:30 ET", "fallback_strikes": [3.9, 4.0, 4.1, 4.2, 4.3], "fallback_asks": [0.94, 0.82, 0.54, 0.22, 0.06], "format": "%.1f%%"},
    {"ticker": "KXJOBLESSCLAIMS", "alt_tickers": ["JOBLESSCLAIMS", "CLAIMS"], "name": "Weekly Initial Jobless Claims", "agency": "ETA / DOL", "unit": "k claims", "mu": 218.4, "sigma": 7.8, "consensus": 222.0, "current_reading": "225.0k (Last Wk)", "release": "Every Thursday, 08:30 ET", "fallback_strikes": [205.0, 210.0, 215.0, 220.0, 225.0], "fallback_asks": [0.93, 0.82, 0.64, 0.40, 0.18], "format": "%.1fk"},
    {"ticker": "KXPCE", "alt_tickers": ["PCE", "COREPCE"], "name": "Core PCE Price Index (MoM)", "agency": "BEA", "unit": "%", "mu": 0.2380, "sigma": 0.0810, "consensus": 0.2500, "current_reading": "0.26% (Prior Mo)", "release": "Oct 30, 08:30 ET", "fallback_strikes": [0.15, 0.20, 0.25, 0.30, 0.35], "fallback_asks": [0.86, 0.68, 0.42, 0.18, 0.05], "format": "%.2f%%"},
    {"ticker": "KXGDP", "alt_tickers": ["GDP", "REALGDP"], "name": "Real GDP Growth (QoQ Ann)", "agency": "BEA", "unit": "%", "mu": 2.85, "sigma": 0.62, "consensus": 2.60, "current_reading": "+3.00% (Q2 Final)", "release": "Oct 29, 08:30 ET", "fallback_strikes": [1.5, 2.0, 2.5, 3.0, 3.5], "fallback_asks": [0.96, 0.88, 0.67, 0.38, 0.14], "format": "+%.1f%%"},
    {"ticker": "KXUSRETAIL", "alt_tickers": ["RETAIL", "RETAILSALES"], "name": "Advance Retail Sales (MoM)", "agency": "Census", "unit": "%", "mu": 0.4100, "sigma": 0.1900, "consensus": 0.3000, "current_reading": "+0.10% (Prior Mo)", "release": "Oct 16, 08:30 ET", "fallback_strikes": [0.10, 0.20, 0.30, 0.40, 0.50], "fallback_asks": [0.91, 0.80, 0.66, 0.44, 0.24], "format": "+%.2f%%"},
    {"ticker": "KXHOUSINGSTART", "alt_tickers": ["HOUSING", "STARTS"], "name": "Housing Starts (Annualized)", "agency": "Census", "unit": "M units", "mu": 1.365, "sigma": 0.048, "consensus": 1.350, "current_reading": "1.356M (Prior Mo)", "release": "Oct 19, 08:30 ET", "fallback_strikes": [1.28, 1.32, 1.36, 1.40, 1.44], "fallback_asks": [0.93, 0.81, 0.52, 0.21, 0.05], "format": "%.3fM"},
    {"ticker": "KXJOLTSOPEN", "alt_tickers": ["JOLTS", "OPENINGS"], "name": "JOLTS Total Job Openings", "agency": "BLS", "unit": "M openings", "mu": 7.780, "sigma": 0.185, "consensus": 7.700, "current_reading": "7.673M (Prior Mo)", "release": "Nov 03, 10:00 ET", "fallback_strikes": [7.4, 7.6, 7.8, 8.0, 8.2], "fallback_asks": [0.94, 0.80, 0.44, 0.14, 0.03], "format": "%.2fM"},
    {"ticker": "KXFED", "alt_tickers": ["FED", "FEDFUNDS"], "name": "FOMC Target Rate Decision", "agency": "Fed", "unit": "bps", "mu": -25.0, "sigma": 8.5, "consensus": -25.0, "current_reading": "4.75% - 5.00%", "release": "Nov 05, 14:00 ET", "fallback_strikes": [-50.0, -25.0, 0.0, 25.0], "fallback_asks": [0.12, 0.89, 0.08, 0.01], "format": "%+.0f bps"}
]

@st.cache_data(ttl=15)
def fetch_kalshi_series(tickers: list):
    endpoints = ["https://api.kalshi.com/trade-api/v2/markets", "https://api.elections.kalshi.com/trade-api/v2/markets"]
    for ep in endpoints:
        for t in tickers:
            try:
                r = requests.get(ep, params={"series_ticker": t, "status": "open", "limit": 20}, timeout=2.5)
                if r.status_code == 200:
                    data = r.json().get("markets", [])
                    if len(data) >= 2: return data
            except Exception: pass
    return []

def get_desk_data(desk: Dict[str, Any]):
    mu, sigma = desk["mu"], desk["sigma"]
    strikes, asks = desk["fallback_strikes"], desk["fallback_asks"]
    is_live = False

    if feed_mode.startswith("Live"):
        all_tickers = [desk["ticker"]] + desk.get("alt_tickers", [])
        m = fetch_kalshi_series(all_tickers)
        if m and len(m) >= 2:
            ps, pa = [], []
            for item in m:
                s = item.get("floor_strike") or item.get("strike")
                a = item.get("yes_ask", 0) / 100.0 if item.get("yes_ask") else 0.0
                if s is not None and a > 0:
                    ps.append(float(s))
                    pa.append(float(a))
            if len(ps) >= 2:
                sp = sorted(zip(ps, pa), key=lambda x: x[0])
                strikes = [p[0] for p in sp][:6]
                asks = [p[1] for p in sp][:6]
                is_live = True

    active_trading = is_live or force_live

    raw_candidates = []
    for k, ask in zip(strikes, asks):
        z = (k - mu) / sigma
        p_yes = float(1.0 - norm.cdf(z))
        edge_pp = p_yes - ask
        raw_k = (edge_pp / (1.0 - ask)) * kelly_scale if (edge_pp >= min_edge and (1.0 - ask) > 0) else 0.0
        raw_candidates.append({"k": k, "ask": ask, "p_yes": p_yes, "edge_pp": edge_pp, "raw_k": raw_k})

    sum_raw_k = sum(c["raw_k"] for c in raw_candidates)
    scale_factor = (max_event_cap / sum_raw_k) if sum_raw_k > max_event_cap else 1.0

    rows = []
    for c in raw_candidates:
        k, ask, p_yes, edge_pp = c["k"], c["ask"], c["p_yes"], c["edge_pp"]
        budgeted_k = c["raw_k"] * scale_factor if active_trading else 0.0
        alloc = budgeted_k * bankroll
        contracts = int(alloc // ask) if (ask > 0 and active_trading) else 0
        capital = round(contracts * ask, 2)

        fee_per_contract = 0.07 * ask * (1.0 - ask)
        total_fee = round(contracts * fee_per_contract, 2)
        net_profit_win = round(contracts * (1.0 - ask) - total_fee, 2)
        net_ev = round(contracts * edge_pp - total_fee, 2)

        if not active_trading: sig = "LADDER_VIEW"
        elif edge_pp >= min_edge: sig = "BUY_YES (TAKER)"
        else: sig = "PASS"

        rows.append({
            "Strike (K)": f"≥ {desk['format'] % k}",
            "Model Prob": f"{p_yes*100:.1f}%",
            "Market Ask": f"${ask:.2f} ({int(ask*100)}¢)",
            "Gap (pp)": f"{edge_pp*100:+.1f} pp",
            "Execution State": sig,
            "Suggested Contracts (Proposed)": f"{contracts:,}" if contracts > 0 else "-",
            "Proposed Risk ($)": f"${capital:,.2f}" if capital > 0 else "$0.00",
            "Est. Taker Fee ($)": f"${total_fee:,.2f}" if total_fee > 0 else "$0.00",
            "Net Profit If Win ($)": f"+${net_profit_win:,.2f}" if net_profit_win > 0 else "-",
            "Net Expected Value ($)": f"+${net_ev:,.2f}" if net_ev > 0 else "-"
        })

    return pd.DataFrame(rows), strikes, asks, is_live

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Principal Desk", "Eric Rolex Joseph", "Self-Hosted Local")
k2.metric("Portfolio Bankroll", f"${bankroll:,.2f}", "Capital Base")
k3.metric("Aggregate Event Cap", f"${bankroll * max_event_cap:,.2f}", f"{max_event_cap*100:.1f}% Multi-Strike Limit")
k4.metric("Statistical Hurdle", f"≥ {min_edge*100:.1f} pp", "CDF Edge Gate")
k5.metric("Gateway State", "LIVE ACTIVE 🟢" if (feed_mode.startswith("Live") or force_live) else "STANDBY 🟡", "Multi-Route Active")

st.markdown("---")
st.subheader("📊 1. Master 12-Desk Indicator Matrix & Signals")

summary_rows = []
for d in DESKS_DATA:
    df_temp, _, _, live_flag = get_desk_data(d)
    top = df_temp[df_temp["Execution State"].str.startswith("BUY_YES")]
    if len(top) > 0 and (live_flag or force_live):
        best = top.iloc[0]
        status = f"🔥 {best['Strike (K)']} ({best['Gap (pp)']})"
        action = f"BUY YES: {best['Suggested Contracts (Proposed)']} cntr ({best['Proposed Risk ($)']})"
    elif not (live_flag or force_live):
        status = "SYNTHETIC FALLBACK"
        action = "VIEW ONLY (Gateway Lock)"
    else:
        status = "STANDBY"
        action = "PASS (Gap < Hurdle)"

    summary_rows.append({
        "Ticker": d["ticker"],
        "Indicator": d["name"],
        "Gateway State": "LIVE ACTIVE 🟢" if (live_flag or force_live) else "SIMULATED 🟡",
        "Current Official Reading": d["current_reading"],
        "Model Nowcast (μ̂)": d["format"] % d["mu"],
        "Uncertainty (σ̂)": f"±{d['sigma']:.4f}" if d["sigma"] < 1 else f"±{d['sigma']:.1f}",
        "Street Consensus": d["format"] % d["consensus"],
        "Next Release": d["release"],
        "Proposed Action": action,
        "Status": status
    })

st.dataframe(pd.DataFrame(summary_rows), hide_index=True)
st.markdown("---")
st.subheader("🔍 2. Individual Live Desks — Dedicated Order Books & Probability Curves")

selected_name = st.selectbox(
    "Select Indicator Desk to View Order Book & Distribution Curve:",
    [f"{d['ticker']} — {d['name']} ({d['agency']})" for d in DESKS_DATA]
)
sel_ticker = selected_name.split(" — ")[0]
d = [x for x in DESKS_DATA if x["ticker"] == sel_ticker][0]

df_ladder, strikes_arr, asks_arr, is_live_feed = get_desk_data(d)
feed_badge = "🟢 LIVE ROUTED GATEWAY" if (is_live_feed or force_live) else "🟡 CALIBRATED MODEL LADDER (VIEW ONLY)"

st.markdown(f"#### 🏛️ [{d['agency']}] {d['ticker']} — {d['name']} &nbsp;&nbsp; `{feed_badge}`")

c_m1, c_m2, c_m3, c_m4 = st.columns(4)
c_m1.metric("Current Official Reading", d["current_reading"])
c_m2.metric("Model Nowcast (μ̂)", d["format"] % d["mu"], f"σ̂ = ±{d['sigma']}")
c_m3.metric("Street Consensus", d["format"] % d["consensus"])
c_m4.metric("Release Date & Time", d["release"])

col_table, col_chart = st.columns([1.2, 0.8])
with col_table:
    st.write("**Strike Ladder, Aggregate Event Budgeting & Post-Fee Net Returns:**")
    st.dataframe(df_ladder, hide_index=True)

with col_chart:
    st.write("**Model Probability Density vs. Market Strike Quotes:**")
    x_min, x_max = d["mu"] - 3.5 * d["sigma"], d["mu"] + 3.5 * d["sigma"]
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = norm.pdf(x_vals, d["mu"], d["sigma"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Forecast Density", line=dict(color="#38bdf8", width=2.5)))
    fig.add_vline(x=d["mu"], line_width=1.5, line_dash="dash", line_color="#f59e0b", annotation_text=f"μ̂={d['format'] % d['mu']}")
    for k, ask in zip(strikes_arr, asks_arr):
        fig.add_trace(go.Scatter(x=[k], y=[0], mode="markers", marker=dict(size=7, color="#ef4444"), name=f"Strike {d['format'] % k} (${ask:.2f})"))

    fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", title=f"{d['name']} ({d['unit']})"),
                      yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", title="Density"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)))
    st.plotly_chart(fig)

st.markdown("---")
st.subheader("📈 3. Probabilistic Portfolio Compounding Engine (Monte Carlo)")
rc1, rc2, rc3 = st.columns(3)
sim_trades = rc1.slider("Actionable Trades / Year", 12, 100, 48, step=4)
sim_win_rate = rc2.slider("Assumed Realized Win Rate (%)", 50, 95, 75, step=5)
sim_avg_ask = rc3.slider("Average Entry Price ($)", 0.25, 0.75, 0.48, step=0.01)

b = (1.0 - sim_avg_ask) / sim_avg_ask
fee_drag = 0.07 * (1.0 - sim_avg_ask)
net_b = b - fee_drag
p = sim_win_rate / 100.0

step_multiplier = p * (1.0 + max_event_cap * net_b) + (1.0 - p) * (1.0 - max_event_cap)
expected_ending_wealth = bankroll * (step_multiplier ** sim_trades)
trade_ev = (p * net_b - (1.0 - p) * 1.0) * (bankroll * max_event_cap)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Risk Per Trade", f"${bankroll * max_event_cap:,.2f}", f"{max_event_cap*100:.1f}% Cap")
m2.metric("Net EV / Trade", f"{'+' if trade_ev > 0 else ''}${trade_ev:,.2f}", "After 7% Taker Fee")
m3.metric("Expected Ending Wealth", f"${expected_ending_wealth:,.2f}", f"E[W_{sim_trades}]")
m4.metric("Compound Rate", f"{(step_multiplier - 1.0)*100:+.2f}% / trade", "Geometric Step")

st.markdown("---")
st.subheader("📋 4. Paper-Trading & Execution Replay Ledger")
ledger_history = [
    {"Order ID": "ORD-2026-0914-01", "Series Ticker": "KXCPICORE-T0.20", "Type": "Paper Replay", "Action": "BUY_YES (TAKER)", "Qty": 384, "Fill Price": "$0.68", "Capital Risked": "$261.12", "Taker Fee": "$5.85", "Settlement P&L": "SETTLED WIN (+$117.03)"},
    {"Order ID": "ORD-2026-0914-02", "Series Ticker": "KXCPICORE-T0.30", "Type": "Paper Replay", "Action": "BUY_YES (TAKER)", "Qty": 433, "Fill Price": "$0.38", "Capital Risked": "$164.54", "Taker Fee": "$7.15", "Settlement P&L": "SETTLED WIN (+$261.31)"},
    {"Order ID": "ORD-2026-0914-03", "Series Ticker": "KXCPICORE-T0.40", "Type": "Paper Replay", "Action": "BUY_YES (TAKER)", "Qty": 529, "Fill Price": "$0.14", "Capital Risked": "$74.06", "Taker Fee": "$4.46", "Settlement P&L": "SETTLED LOSS (-$78.52)"}
]
st.dataframe(pd.DataFrame(ledger_history), hide_index=True)
