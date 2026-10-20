import hmac
import json
import os
from datetime import datetime
from html import escape

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import banking_ui
import batch_ui
import charts as charts_mod
from banking import bank_share, catchment_branches
import memo as memo_mod
from interpret import band, band_legend_html, driver_sentence, guide_market, rating
from ai_assistant import (PURPOSES, DataTools, ai_enabled, brief_model, chat_model, generate_brief,
                          get_client, run_chat)
from data_sources import ALL_SECTORS, load_all, offline, run_diagnostics, secret

st.set_page_config(page_title="Synoptic BI", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

NAVY, TEAL, SLATE = "#0b1f3a", "#0e7490", "#64748b"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1400px; }
#MainMenu, footer { visibility: hidden; }
.hero { background: linear-gradient(120deg, #0b1f3a 0%, #0e4d64 55%, #0e7490 100%); border-radius: 20px;
        padding: 30px 36px; margin-bottom: 22px; color: #fff; box-shadow: 0 10px 30px rgba(11,31,58,.18); }
.hero-eyebrow { letter-spacing: .18em; font-size: .72rem; font-weight: 600; color: #7dd3e0; }
.hero-title { font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: 6px 0 8px 0; letter-spacing: -.02em; }
.hero-sub { font-size: 1.02rem; color: #cfe8ee; max-width: 760px; }
.pill { display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: .75rem; font-weight: 600; margin-right: 6px; }
.pill-live { background:#dcfce7; color:#166534; } .pill-sample { background:#fef3c7; color:#92400e; }
.hero .pill { margin-top: 14px; }
.kpi { background:#fff; border:1px solid #e5eaf1; border-radius:16px; padding:16px 20px; box-shadow:0 2px 10px rgba(11,31,58,.05); }
.kpi-label { font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; color:#64748b; font-weight:600; }
.kpi-value { font-size:1.65rem; font-weight:800; color:#0b1f3a; margin-top:2px; }
.kpi-sub { font-size:.82rem; color:#0e7490; font-weight:500; }
.section-title { font-size:1.15rem; font-weight:700; color:inherit; margin: 6px 0 2px 0; }  /* follows light/dark theme */
.section-sub { font-size:.88rem; color:inherit; opacity:.78; margin-bottom:10px; }
.stTabs { margin-top: 1.5rem; margin-bottom: 1rem; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; padding-bottom: 4px; }
.stTabs [data-baseweb="tab"] { background:#fff; border:1px solid #e5eaf1; border-radius:10px 10px 0 0; padding: 8px 18px; font-weight:600; color:#0b1f3a; }
.stTabs [aria-selected="true"] { background:#0b1f3a; color:#fff !important; }
.stTabs [aria-selected="true"] p, .stTabs [data-baseweb="tab"] p { color:inherit !important; }
.kpi-row-gap { height: .5rem; }
.read-line { margin:10px 0 6px 0; padding:10px 14px; border-left:4px solid #0e7490; background:rgba(14,116,144,.08);
  border-radius:0 10px 10px 0; font-size:.9rem; }
.site-footer { text-align:center; color:#94a3b8; font-size:.8rem; margin-top:36px; }
</style>
""",
    unsafe_allow_html=True,
)


def hero(pills: str = "") -> None:
    st.markdown(
        f'<div class="hero"><div class="hero-eyebrow">FLORIDA MARKET INTELLIGENCE</div>'
        f'<div class="hero-title">Synoptic BI</div>'
        f'<div class="hero-sub">Live economic, workforce and demographic signals to screen Florida markets for expansion, '
        f'hiring and sales.</div>{pills}</div>',
        unsafe_allow_html=True,
    )


def gate() -> None:
    """Optional password protection: set APP_PASSWORD in secrets/env to require it."""
    pw = secret("APP_PASSWORD")
    if not pw or st.session_state.get("auth"):
        return
    hero()
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), str(pw).encode()):
            st.session_state["auth"] = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


gate()

# ------------------------------------------------------------------ data
markets, sector_lq, status, loaded_at = load_all()
n_live = int((status["State"] == "Live").sum())
all_live = n_live == len(status)
pill_html = (
    f'<span class="pill {"pill-live" if all_live else "pill-sample"}">'
    f'{"● All sources live" if all_live else f"● {n_live}/{len(status)} sources live"}</span>'
    f'<span class="pill" style="background:rgba(255,255,255,.15);color:#fff">Updated {loaded_at}</span>'
)
hero(pill_html)

# ------------------------------------------------------------------ sidebar
# One-click role presets. Each sets every slider so the result is predictable. Banking sliders live in the Banking tab.
PRESETS = {
    "Custom (my own settings)": None,
    "↩ Reset to defaults": {"w_lq": 30, "w_gdp": 20, "w_inc": 30, "w_edu": 20, "income_floor": 40000,
                           "bachelor_floor": 15, "bk_w1": 40, "bk_w2": 40, "bk_w3": 20,
                           "note": "Back to the standard weights and filters."},
    "🏦 Banking expansion": {
        "w_lq": 0, "w_gdp": 40, "w_inc": 30, "w_edu": 30, "income_floor": 40000, "bachelor_floor": 15,
        "bk_w1": 20, "bk_w2": 50, "bk_w3": 30,
        "note": "Banking tab: 50% under-served gap (businesses per branch), 30% small-business density, 20% volume. "
                "Markets: growth and income weighted. (ZIP-level income growth is not in the data, so GDP growth stands in.)"},
    "💼 Tech / professional hiring": {
        "w_lq": 20, "w_gdp": 30, "w_inc": 0, "w_edu": 50, "income_floor": 30000, "bachelor_floor": 25,
        "note": "50% bachelor's+ attainment, 30% GDP growth, 20% sector concentration. Minimum bachelor's 25%."},
    "💎 Wealth management": {
        "w_lq": 0, "w_gdp": 40, "w_inc": 60, "w_edu": 0, "income_floor": 90000, "bachelor_floor": 0,
        "note": "60% median income, 40% growth, markets with median income of $90k or more. Raise the income minimum "
                "to $140k to see only the very highest-income markets (that leaves very few). Population growth is not "
                "in the data, so GDP growth stands in for it."},
}


def _apply_preset() -> None:
    cfg = PRESETS.get(st.session_state.get("preset"))
    if cfg:
        for k, v in cfg.items():
            if k != "note":
                st.session_state[k] = v


def _reset_all() -> None:
    st.session_state["preset"] = "↩ Reset to defaults"
    _apply_preset()


st.sidebar.markdown("### 🎛️ Filters")
st.sidebar.selectbox("🎯 Scoring preset", list(PRESETS), key="preset", on_change=_apply_preset,
                     help="One click sets the filters and weights for a type of buyer. You can still fine-tune them.")
_note = (PRESETS.get(st.session_state.get("preset")) or {}).get("note")
if _note:
    st.sidebar.caption(_note)
selected_sector = st.sidebar.selectbox("Industry sector", ALL_SECTORS, index=11)
income_floor = st.sidebar.slider("Minimum median income ($)", 30000, 150000, 40000, step=5000, key="income_floor")
bachelor_floor = st.sidebar.slider("Minimum bachelor's degree (%)", 0, 50, 15, key="bachelor_floor")
with st.sidebar.expander("⚖️ Scoring weights", expanded=False):
    w_lq = st.slider("Sector concentration (LQ)", 0, 100, 30, key="w_lq")
    w_gdp = st.slider("GDP growth", 0, 100, 20, key="w_gdp")
    w_inc = st.slider("Median income", 0, 100, 30, key="w_inc")
    w_edu = st.slider("Education (bachelor's+)", 0, 100, 20, key="w_edu")
if st.sidebar.button("🔄 Refresh live data", width="stretch"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("Data is cached for 6 hours to stay fast and polite to public APIs.")

total_w = w_lq + w_gdp + w_inc + w_edu
if total_w == 0:
    st.error("Set at least one scoring weight above zero.")
    st.stop()

if not all_live:
    samples = ", ".join(status.loc[status["State"] != "Live", "Source"])
    missing = [k for k in ("CENSUS_API_KEY", "BEA_API_KEY") if not secret(k)]
    seen_names = sorted(k for k in os.environ if "CENSUS" in k.upper() or "BEA" in k.upper())
    seen_txt = ("; this server sees these setting names: " + ", ".join(repr(k) for k in seen_names)) if seen_names \
        else "; this server sees no setting whose name contains CENSUS or BEA"
    tip = (f" Missing keys: {', '.join(missing)} (both are free). Add them in your host's secrets or environment "
           f"settings, then redeploy and click Refresh live data{seen_txt}."
           if missing else " Open the Data Sources tab and run the connection check to see exactly what failed.")
    st.warning(f"Sample data in use for: {samples}. Rankings are illustrative until those sources are live." + tip)


# ------------------------------------------------------------------ scoring
def norm(s: pd.Series) -> pd.Series:
    filled = s.fillna(s.median()).fillna(0)
    rng = filled.max() - filled.min()
    return (filled - filled.min()) / rng if rng else pd.Series(0.5, index=s.index)


lq_sel = sector_lq.loc[sector_lq["sector"] == selected_sector, ["zip_code", "lq"]]
df = markets.merge(lq_sel, on="zip_code", how="left")
df["highly_educated_pct"] = df["edu_bachelor_pct"] + df["edu_graduate_pct"]
parts = {
    "Sector concentration": (norm(df["lq"]), w_lq),
    "GDP growth": (norm(df["gdp_growth_pct"]), w_gdp),
    "Median income": (norm(df["median_income"]), w_inc),
    "Education": (norm(df["highly_educated_pct"]), w_edu),
}
for name, (series, w) in parts.items():
    df[f"c_{name}"] = series * w / total_w * 100
df["BI_Score"] = df[[f"c_{n}" for n in parts]].sum(axis=1)

view = (
    df[(df["median_income"] >= income_floor) & (df["edu_bachelor_pct"] >= bachelor_floor)]
    .sort_values("BI_Score", ascending=False)
    .reset_index(drop=True)
    .copy()
)
view.insert(0, "rank", view.index + 1)

if view.empty:
    st.warning("⚠️ No markets match your filters, so there is nothing to show. Lower the income or education minimum "
               "in the sidebar, or reset everything with the button below.")
    st.button("↩ Reset filters and weights", type="primary", on_click=_reset_all)
    st.stop()


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="kpi-label">{escape(label)}</div>'
            f'<div class="kpi-value">{escape(value)}</div><div class="kpi-sub">{escape(sub)}</div></div>')


top = view.iloc[0]
k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi("Top market", top["city"], f'Score {top["BI_Score"]:.0f} / 100 · {rating(top["BI_Score"])}'), unsafe_allow_html=True)
k2.markdown(kpi("Markets matching", f"{len(view)} of {len(df)}", selected_sector.split(" (")[0]), unsafe_allow_html=True)
k3.markdown(kpi("Avg median income", f'${view["median_income"].mean():,.0f}', "across matching markets"), unsafe_allow_html=True)
k4.markdown(kpi("Avg GDP growth", f'{view["gdp_growth_pct"].mean():.1f}%', "county-level"), unsafe_allow_html=True)
st.write("")

tab_over, tab_matrix, tab_compare, tab_bank, tab_batch, tab_ai, tab_memo, tab_brief, tab_src = st.tabs(
    ["📍 Overview", "🧮 Market Matrix", "⚖️ Compare", "🏦 Banking", "📥 Batch", "🤖 AI Analyst", "📄 Memo",
     "📖 Brief", "🔌 Data Sources"]
)

# Read-only data tools the AI can query. Race/ethnicity columns are never passed to it.
filters_state = {"min_median_income": income_floor, "min_bachelors_only_pct": bachelor_floor}
weights_pct = {"sector_concentration": round(w_lq / total_w * 100), "gdp_growth": round(w_gdp / total_w * 100),
               "median_income": round(w_inc / total_w * 100), "education": round(w_edu / total_w * 100)}
ai_tools = DataTools(
    scored=df, sector_lq=sector_lq, sector=selected_sector, passing_cities=list(view["city"]),
    filters=filters_state, weights_pct=weights_pct, sources=status[["Source", "State"]].to_dict("records"),
)

# ------------------------------------------------------------------ OVERVIEW
with tab_over:
    guide_market(st)
    c_map, c_rank = st.columns([6, 5], gap="large")

    with c_map:
        st.markdown('<div class="section-title">Opportunity map</div>'
                    '<div class="section-sub">Bigger, darker circles = higher opportunity score. Hover for details.</div>',
                    unsafe_allow_html=True)
        m = view.copy()
        t = (m["BI_Score"] / 100).clip(0, 1)
        low, high = np.array([186, 220, 228]), np.array([11, 31, 58])
        m["color"] = [[int(v) for v in (low + (high - low) * x)] + [215] for x in t]
        m["radius"] = 5000 + m["BI_Score"] * 650
        m["score_txt"] = m["BI_Score"].map("{:.0f}".format)
        m["income_txt"] = m["median_income"].map("${:,.0f}".format)
        layer = pdk.Layer(
            "ScatterplotLayer", data=m[["city", "county", "longitude", "latitude", "color", "radius",
                                        "score_txt", "income_txt"]],
            get_position="[longitude, latitude]", get_fill_color="color", get_radius="radius",
            pickable=True, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1.5,
        )
        st.pydeck_chart(pdk.Deck(
            layers=[layer], initial_view_state=pdk.ViewState(latitude=28.1, longitude=-82.4, zoom=5.3),
            tooltip={"html": "<b>{city}</b> · {county}<br/>Score {score_txt} · Income {income_txt}",
                     "style": {"backgroundColor": NAVY, "color": "white"}},
        ), height=500)

    with c_rank:
        st.markdown('<div class="section-title">Ranking</div>'
                    '<div class="section-sub">Weighted score of concentration, growth, income and education.</div>',
                    unsafe_allow_html=True)
        chart = (
            alt.Chart(view).mark_bar(cornerRadiusEnd=5, height=18)
            .encode(
                x=alt.X("BI_Score:Q", title=None, scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("city:N", sort="-x", title=None),
                color=alt.Color("BI_Score:Q", scale=alt.Scale(range=["#a5d8d4", TEAL, NAVY]), legend=None),
                tooltip=[alt.Tooltip("city:N"), alt.Tooltip("BI_Score:Q", format=".1f", title="Score")],
            ).properties(height=max(260, 30 * len(view)))
        )
        st.altair_chart(chart, width="stretch")
        st.markdown(band_legend_html(), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🔎 Market spotlight</div>'
                '<div class="section-sub">Pick a market to see what is driving its score.</div>', unsafe_allow_html=True)
    pick = st.selectbox("Market", view["city"], label_visibility="collapsed")
    row = view[view["city"] == pick].iloc[0]
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Rank", f'#{int(row["rank"])} of {len(view)}')
    s2.metric("Score", f'{row["BI_Score"]:.0f} · {rating(row["BI_Score"])}')
    s3.metric("Median income", f'${row["median_income"]:,.0f}')
    s4.metric("GDP growth", "n/a" if pd.isna(row["gdp_growth_pct"]) else f'{row["gdp_growth_pct"]:.1f}%')
    s5.metric("Bachelor's+", f'{row["highly_educated_pct"]:.0f}%')
    _lab, _col, _mean = band(row["BI_Score"])
    st.markdown(f'<div class="read-line" style="border-left-color:{_col}"><b>{escape(_lab)} ({row["BI_Score"]:.0f}/100):</b> {escape(_mean)} '
                f'{escape(driver_sentence(row, parts, total_w))}</div>', unsafe_allow_html=True)
    contrib = pd.DataFrame({"Component": list(parts), "Points": [row[f"c_{n}"] for n in parts]})
    bars = (alt.Chart(contrib).mark_bar(cornerRadiusEnd=5, color=TEAL, height=22)
            .encode(x=alt.X("Points:Q", title="Points contributed to score"),
                    y=alt.Y("Component:N", sort=None, title=None),
                    tooltip=["Component", alt.Tooltip("Points:Q", format=".1f")]).properties(height=150))
    st.altair_chart(bars, width="stretch")

# ------------------------------------------------------------------ MATRIX
with tab_matrix:
    st.markdown('<div class="section-title">Market Matrix</div>'
                '<div class="section-sub">Search, switch views, sort, and download. Click any column header to re-sort.</div>',
                unsafe_allow_html=True)
    f1, f2, f3 = st.columns([3, 3, 3])
    query = f1.text_input("🔍 Search city or county", placeholder="e.g. Tampa or Broward")
    view_name = f2.radio("View", ["Summary", "Economy", "Education", "Demographics", "Everything"],
                         horizontal=True)
    sort_by = f3.selectbox("Sort by", ["Opportunity score", "Median income", "GDP growth",
                                       "Bachelor's+ %", "City (A–Z)"])

    mx = view.copy()
    if query:
        q = query.lower()
        mx = mx[mx["city"].str.lower().str.contains(q) | mx["county"].str.lower().str.contains(q)]
    sort_map = {"Opportunity score": ("BI_Score", False), "Median income": ("median_income", False),
                "GDP growth": ("gdp_growth_pct", False), "Bachelor's+ %": ("highly_educated_pct", False),
                "City (A–Z)": ("city", True)}
    col, asc = sort_map[sort_by]
    mx = mx.sort_values(col, ascending=asc)
    mx["rating"] = mx["BI_Score"].map(rating)

    VIEWS = {
        "Summary": ["rank", "city", "county", "BI_Score", "rating", "median_income", "gdp_growth_pct", "highly_educated_pct", "lq"],
        "Economy": ["city", "county", "median_income", "gdp_growth_pct", "lq", "BI_Score"],
        "Education": ["city", "edu_associate_pct", "edu_bachelor_pct", "edu_graduate_pct", "highly_educated_pct"],
        "Demographics": ["city", "county", "race_white_pct", "race_black_pct", "race_asian_pct", "race_hispanic_latino_pct"],
        "Everything": ["rank", "city", "county", "zip_code", "BI_Score", "rating", "lq", "median_income", "gdp_growth_pct",
                       "edu_associate_pct", "edu_bachelor_pct", "edu_graduate_pct", "highly_educated_pct",
                       "race_white_pct", "race_black_pct", "race_asian_pct", "race_hispanic_latino_pct"],
    }
    cc = st.column_config
    pct_bar = lambda label: cc.ProgressColumn(label, min_value=0, max_value=100, format="%.1f%%")  # noqa: E731
    config = {
        "rank": cc.NumberColumn("#", width="small"),
        "city": cc.TextColumn("Market", width="medium"),
        "county": cc.TextColumn("County"),
        "zip_code": cc.TextColumn("ZIP"),
        "BI_Score": cc.ProgressColumn("Opportunity score", min_value=0, max_value=100, format="%.0f"),
        "rating": cc.TextColumn("Rating", help="80+ Exceptional, 65+ Strong, 50+ Solid, 35+ Moderate, below Weak"),
        "lq": cc.NumberColumn("Sector LQ", format="%.2f", help="Location quotient: >1 means more concentrated than the U.S."),
        "median_income": cc.NumberColumn("Median income", format="$%d"),
        "gdp_growth_pct": cc.NumberColumn("GDP growth", format="%.1f%%"),
        "edu_associate_pct": pct_bar("Associate"),
        "edu_bachelor_pct": pct_bar("Bachelor's"),
        "edu_graduate_pct": pct_bar("Graduate"),
        "highly_educated_pct": pct_bar("Bachelor's+"),
        "race_white_pct": pct_bar("White (non-Hisp.)"),
        "race_black_pct": pct_bar("Black (non-Hisp.)"),
        "race_asian_pct": pct_bar("Asian (non-Hisp.)"),
        "race_hispanic_latino_pct": pct_bar("Hispanic/Latino"),
    }
    cols = VIEWS[view_name]
    st.dataframe(mx[cols], column_config=config, hide_index=True, width="stretch",
                 height=min(80 + 36 * len(mx), 640))
    st.caption(f"{len(mx)} markets · Sector: {selected_sector} · LQ and GDP are county-level, so ZIPs in the same "
               "county share a value. Hispanic/Latino is an ethnicity measured separately from race.")
    st.download_button("⬇️ Download this view (CSV)", mx[cols].to_csv(index=False).encode(),
                       file_name="synoptic_bi_markets.csv", mime="text/csv")

# ------------------------------------------------------------------ COMPARE
with tab_compare:
    st.markdown('<div class="section-title">Side-by-side comparison</div>'
                '<div class="section-sub">Choose up to four markets.</div>', unsafe_allow_html=True)
    chosen = st.multiselect("Markets", list(view["city"]), default=list(view["city"][:3]), max_selections=4)
    if chosen:
        cmp = view[view["city"].isin(chosen)].set_index("city").loc[chosen]
        table = pd.DataFrame({
            "Opportunity score": cmp["BI_Score"].map("{:.0f}".format),
            "Median income": cmp["median_income"].map("${:,.0f}".format),
            "GDP growth": cmp["gdp_growth_pct"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1f}%"),
            "Sector LQ": cmp["lq"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}"),
            "Bachelor's+": cmp["highly_educated_pct"].map("{:.1f}%".format),
            "Associate": cmp["edu_associate_pct"].map("{:.1f}%".format),
        }).T
        st.dataframe(table, width="stretch")
        long = pd.DataFrame([{"Market": c, "Component": n, "Points": cmp.loc[c, f"c_{n}"]}
                             for c in chosen for n in parts])
        st.altair_chart(
            alt.Chart(long).mark_bar().encode(
                x=alt.X("Market:N", sort=chosen, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Points:Q", title="Score points"),
                color=alt.Color("Component:N", scale=alt.Scale(range=[NAVY, TEAL, "#5eb8c4", "#f59e0b"])),
                tooltip=["Market", "Component", alt.Tooltip("Points:Q", format=".1f")],
            ).properties(height=320),
            width="stretch",
        )

# ------------------------------------------------------------------ BANKING
with tab_bank:
    banking_ui.render()
ai_tools.attach_banking(st.session_state.get("banking_ctx"))

# ------------------------------------------------------------------ BATCH
with tab_batch:
    batch_ui.render()

# ------------------------------------------------------------------ AI ANALYST
with tab_ai:
    st.markdown('<div class="section-title">AI Analyst</div>'
                '<div class="section-sub">Claude explains and queries your dashboard data. Every figure it states comes '
                'from the app\'s own numbers, and race/ethnicity columns are never shared with it.</div>',
                unsafe_allow_html=True)
    if not ai_enabled():
        st.info("The AI Analyst is switched off. Add `ANTHROPIC_API_KEY` to your secrets to enable it "
                "(create a key in the Claude Console). See DEPLOY.md for where secrets go.")
    else:
        client = get_client()
        if not ai_tools.banking:
            st.warning("Bank questions can't be answered yet because the banking data isn't loaded. Reason:")
            bs = st.session_state.get("banking_status")
            if bs is not None:
                st.dataframe(bs, hide_index=True, width="stretch")
            else:
                st.caption("Open the 🏦 Banking tab once, then come back.")
        sub_brief, sub_chat = st.tabs(["📝 Brief generator", "💬 Ask the data"])

        # ---- Brief generator
        with sub_brief:
            c1, c2 = st.columns([2, 1])
            purpose = c1.selectbox("Brief for", PURPOSES)
            top_n = c2.slider("Markets to cover", 3, 8, 5)
            signature = json.dumps([purpose, top_n, selected_sector, filters_state, weights_pct, loaded_at])
            if st.button("✨ Generate brief", type="primary"):
                with st.spinner("Writing your brief…"):
                    try:
                        text = generate_brief(client, brief_model(), ai_tools, purpose, top_n)
                        st.session_state["brief"] = {"text": text, "sig": signature,
                                                     "label": f"{purpose} · {selected_sector} · data as of {loaded_at}"}
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"The AI request failed ({type(exc).__name__}). Check ANTHROPIC_API_KEY and try again.")
            saved = st.session_state.get("brief")
            if saved:
                if saved["sig"] != signature:
                    st.caption("⚠️ Settings or data changed since this brief was generated. Regenerate to refresh it.")
                with st.container(border=True):
                    st.markdown(saved["text"])
                st.caption(f"AI-generated from dashboard figures: {saved['label']}. Verify before acting on it.")
                dl1, dl2 = st.columns(2)
                dl1.download_button("⬇️ Download brief (Word .docx)",
                                    memo_mod.build_brief_docx(saved["text"], saved["label"],
                                                              [f"{r.Source}: {r.State}" for r in status[["Source", "State"]].itertuples()],
                                                              loaded_at),
                                    file_name="synoptic_bi_brief.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    width="stretch")
                dl2.download_button("⬇️ Download brief (Markdown)", saved["text"].encode(),
                                    file_name="synoptic_bi_brief.md", mime="text/markdown", width="stretch")

        # ---- Ask the data
        with sub_chat:
            MAX_TURNS = 25
            chat = st.session_state.setdefault("ai_chat", [])
            st.caption("Answers use the sector, filters and weights currently set in the sidebar.")
            examples = [
                "Which markets have bachelor's+ above 40% and income over $70k?",
                "Rank the top 5 markets for healthcare concentration.",
                f"Why is {top['city']} ranked first?",
                "Which ZIPs have many small businesses but few bank branches?",
                "Who are the main bank competitors in ZIP 33131?",
                "Which ZIPs look saturated with bank branches?",
            ]
            for row_start in range(0, len(examples), 3):
                ex_cols = st.columns(3)
                for j, q in enumerate(examples[row_start:row_start + 3]):
                    if ex_cols[j].button(q, key=f"ex{row_start + j}", width="stretch"):
                        st.session_state["pending_q"] = q

            for m in chat:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if m.get("tools"):
                        with st.expander("How this was answered"):
                            for t in m["tools"]:
                                st.code(f'{t["name"]}({json.dumps(t["input"])})', language="python")

            prompt = st.chat_input("Ask about markets, sectors, income, education…", max_chars=500)
            prompt = prompt or st.session_state.pop("pending_q", None)
            if prompt:
                if sum(1 for m in chat if m["role"] == "user") >= MAX_TURNS:
                    st.warning("Chat limit reached for this session. Clear the chat to keep going.")
                else:
                    chat.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing…"):
                            try:
                                out = run_chat(client, chat_model(), ai_tools,
                                               [{"role": m["role"], "content": m["content"]} for m in chat])
                            except Exception as exc:  # noqa: BLE001
                                out = None
                                chat.pop()  # drop the unanswered question so the history stays valid
                                st.error(f"The AI request failed ({type(exc).__name__}). "
                                         "Check ANTHROPIC_API_KEY and try again.")
                        if out:
                            st.markdown(out["text"])
                            chat.append({"role": "assistant", "content": out["text"], "tools": out["tools"]})
                            if out["tools"]:
                                with st.expander("How this was answered"):
                                    for t in out["tools"]:
                                        st.code(f'{t["name"]}({json.dumps(t["input"])})', language="python")
            if chat and st.button("🗑️ Clear chat"):
                st.session_state["ai_chat"] = []
                st.rerun()

# ------------------------------------------------------------------ MEMO
with tab_memo:
    st.markdown('<div class="section-title">Board-ready memo</div>'
                '<div class="section-sub">One click turns the current view into a branded Word and PDF memo: executive '
                'summary, macro scorecard, sector mix, bank-branch analysis and a compliance note.</div>',
                unsafe_allow_html=True)
    saved_brief = st.session_state.get("brief")
    m1, m2 = st.columns([3, 2])
    prepared_for = m1.text_input("Prepared for (optional)", placeholder="e.g. Investment Committee, Q4 expansion review",
                                 key="memo_for")
    if len(view) > 3:  # a slider needs min < max; with 3 or fewer markets, include them all
        n_mk = m2.slider("Markets to include", 3, min(12, len(view)), min(8, len(view)), key="memo_n")
    else:
        n_mk = len(view)
        m2.caption(f"Including all {n_mk} market(s) that match your filters.")
    if saved_brief:
        st.success(f"Executive summary will use your AI brief: {saved_brief['label']}")
    elif ai_enabled():
        st.info("No AI brief yet. Generate one in the 🤖 AI Analyst tab first for a written executive summary; "
                "otherwise the memo is built from the data tables alone.")
    else:
        st.info("The AI Analyst is off, so the memo is built from the data tables alone. Add ANTHROPIC_API_KEY to "
                "include a written executive summary.")
    if "banking_ctx" not in st.session_state:
        st.caption("The banking section is included only when the 🏦 Banking tab has loaded.")
    if st.button("📄 Build memo files", type="primary", key="memo_build"):
        with st.spinner("Building your memo…"):
            try:
                purpose_lbl = saved_brief["label"].split(" · ")[0] if saved_brief else "Market screening"
                payload = memo_mod.collect(view, sector_lq, selected_sector, weights_pct, filters_state, loaded_at, status,
                                           st.session_state.get("banking_ctx"), saved_brief["text"] if saved_brief else None,
                                           purpose_lbl, prepared_for, top_n=n_mk)
                st.session_state["memo_files"] = {"docx": memo_mod.build_docx(payload), "pdf": memo_mod.build_pdf(payload),
                                                  "built": datetime.now().strftime("%H:%M")}
            except Exception as exc:  # noqa: BLE001
                st.session_state.pop("memo_files", None)
                st.error(f"Could not build the memo ({type(exc).__name__}: {exc}). "
                         "If this mentions a missing module, add python-docx and reportlab to requirements.txt.")
    mf = st.session_state.get("memo_files")
    if mf:
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download Word (.docx)", mf["docx"], file_name="synoptic_bi_memo.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", width="stretch")
        d2.download_button("⬇️ Download PDF", mf["pdf"], file_name="synoptic_bi_memo.pdf", mime="application/pdf",
                           width="stretch")
        st.caption(f"Built at {mf['built']} from the settings you have now. Change a filter or weight, then build again "
                   "to refresh it.")

    # ---- chart images for articles and slides
    st.markdown("---")
    st.markdown('<div class="section-title">📰 Charts for your article</div>'
                '<div class="section-sub">Clean 1200 x 675 PNG images with the title, sources and data date built in, '
                'ready to drop into a newsletter, post or slide. For writing help, choose '
                '"Newsletter article outline" in the AI Analyst brief generator.</div>', unsafe_allow_html=True)
    if not all_live:
        st.warning("Some sources are on sample data. Do not publish these charts as fact until every source is live.")
    if st.button("🖼️ Prepare chart images", key="chart_build"):
        with st.spinner("Drawing charts…"):
            try:
                files = {"market_scores": charts_mod.market_scores_png(view, selected_sector, loaded_at)}
                ctx_b = st.session_state.get("banking_ctx")
                if ctx_b:
                    tb = ctx_b["table"]
                    files["banking_opportunity"] = charts_mod.banking_opportunity_png(
                        tb, ctx_b["small_def"], ctx_b.get("catch_label", f'{ctx_b["radius"]} mi'), loaded_at)
                    r0 = tb.iloc[0]
                    ring0 = (ctx_b.get("catchments") or {}).get(r0["zip_code"])
                    nb0 = catchment_branches(ctx_b["branches"], r0["latitude"], r0["longitude"], ctx_b["radius"], ring0)
                    if not nb0.empty:
                        hhi_txt = (f'{str(r0["hhi_label"]).lower()} (HHI {r0["hhi_used"]:,.0f})'
                                   if pd.notna(r0["hhi_used"]) else "deposit concentration n/a")
                        files["deposit_share"] = charts_mod.deposit_share_png(
                            bank_share(nb0), f'{r0["city"]} ({r0["zip_code"]})',
                            ctx_b.get("catch_label", f'{ctx_b["radius"]} mi'), hhi_txt, loaded_at)
                st.session_state["chart_files"] = files
            except Exception as exc:  # noqa: BLE001
                st.session_state.pop("chart_files", None)
                st.error(f"Could not draw the charts ({type(exc).__name__}: {exc}). If this mentions matplotlib, "
                         "add matplotlib to requirements.txt.")
    cf = st.session_state.get("chart_files")
    if cf:
        names = {"market_scores": "Markets ranked by score", "banking_opportunity": "Banking opportunity by ZIP",
                 "deposit_share": "Deposit share around the top ZIP"}
        for key, label in names.items():
            if key in cf:
                st.image(cf[key], caption=label, width="stretch")
                st.download_button(f"⬇️ Download: {label}", cf[key], file_name=f"synoptic_bi_{key}.png",
                                   mime="image/png", key=f"dl_{key}")
        if "banking_opportunity" not in cf:
            st.caption("Open the 🏦 Banking tab first to include the banking charts.")
        st.caption("Charts show the settings you have right now. Re-prepare them after changing a filter or weight.")

# ------------------------------------------------------------------ BRIEF
with tab_brief:
    st.markdown('<div class="section-title">How to use this platform</div>', unsafe_allow_html=True)
    st.info("Synoptic BI combines public economic and demographic indicators into one transparent score to "
            "support early-stage market screening. It supports, and does not replace, on-the-ground due diligence.")
    b1, b2 = st.columns(2, gap="large")
    with b1:
        st.markdown("#### 🎯 Business applications")
        st.markdown(
            "- **Site selection:** screen markets on growth, income, education and sector concentration.\n"
            "- **Recruiting & staffing:** compare associate, bachelor's and graduate attainment.\n"
            "- **B2B sales prospecting:** use the income and education filters to prioritize higher-spending markets.\n"
            "- **Market context:** demographic columns are informational only. They are not a compliance tool and "
            "must not be used to include or exclude communities from services, lending or housing. Consult counsel on "
            "fair-lending, fair-housing and civil rights obligations."
        )
    with b2:
        st.markdown("#### ⚙️ Quick guide")
        st.markdown(
            "1. Pick one of 20 NAICS sectors; the concentration (LQ) component updates for it.\n"
            "2. Set income and education minimums in the sidebar.\n"
            "3. Adjust scoring weights to fit your priorities; they are normalized to sum to 100%.\n"
            "4. Use **Overview** for the map and ranking, **Market Matrix** to search and download, "
            "and **Compare** for side-by-side views.\n"
            "5. Use **AI Analyst** to generate a written brief or ask questions in plain English "
            "(requires an Anthropic API key)."
        )
    st.markdown("#### 🧮 Score methodology")
    st.write("Opportunity score = weighted average of four min-max normalized components (0-1) scaled to 0-100: sector "
             "location quotient, GDP growth, median income and bachelor's-plus attainment. Defaults are 30/20/30/20. "
             "Scores are relative to all markets in the dataset (before filtering). Missing values are replaced by the "
             "median for scoring only.")

# ------------------------------------------------------------------ DATA SOURCES
with tab_src:
    st.markdown('<div class="section-title">Data sources & freshness</div>'
                f'<div class="section-sub">Last refreshed {loaded_at}. Cached for 6 hours; use the sidebar button to refresh.</div>',
                unsafe_allow_html=True)
    st.dataframe(status, hide_index=True, width="stretch",
                 column_config={"State": st.column_config.TextColumn("State", width="small")})
    if offline():
        st.info("Offline mode is on (SYNOPTIC_OFFLINE). Live fetching is disabled.")
    st.markdown(
        "**Make everything live**\n"
        "- **Census (ACS and Business Patterns):** a free key is REQUIRED; Census rejects requests without one. "
        "Get it at api.census.gov/data/key_signup.html and set `CENSUS_API_KEY`.\n"
        "- **BLS QCEW:** open data, no key. If it fails from your host, set `BLS_CONTACT_EMAIL` (BLS asks "
        "automated clients to identify a contact).\n"
        "- **BEA GDP:** needs a free key from apps.bea.gov/API/signup. Set `BEA_API_KEY`.\n\n"
        "- **Banking tab:** small-business counts come from Census CBP and need `CENSUS_API_KEY`; bank branches come "
        "from the FDIC Summary of Deposits. If FDIC asks for a key, a free api.data.gov key goes in `FDIC_API_KEY`.\n"
        "- **AI Analyst:** set `ANTHROPIC_API_KEY` (Claude Console). Optional: `AI_CHAT_MODEL` and `AI_BRIEF_MODEL` "
        "to change the models.\n\n"
        "Set keys in `.streamlit/secrets.toml` locally or in your host's secrets settings."
    )
    st.caption(f"AI Analyst: {'enabled' if ai_enabled() else 'disabled (no ANTHROPIC_API_KEY)'}")

    st.markdown("---")
    st.markdown('<div class="section-title">🩺 Connection check</div>'
                '<div class="section-sub">Tests every source once from this server and says exactly what is wrong. '
                'Key values are never shown.</div>', unsafe_allow_html=True)
    if st.button("Run connection check"):
        with st.spinner("Testing each source…"):
            st.session_state["diag"] = run_diagnostics()
    if "diag" in st.session_state:
        st.dataframe(st.session_state["diag"], hide_index=True, width="stretch")

st.markdown('<div class="site-footer">© Synoptic BI · Data: U.S. Census Bureau (ACS), Bureau of Labor Statistics '
            '(QCEW), Bureau of Economic Analysis. For informational purposes only; not legal or investment advice.</div>',
            unsafe_allow_html=True)
