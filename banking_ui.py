"""Streamlit UI for the Banking tab: small businesses and FDIC branches by ZIP code."""
from __future__ import annotations

import re
from html import escape

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from banking import (CAPTURE_LABELS, DEFAULT_WEIGHTS, HHI_LABELS, SMALL_DEFS, bank_share, branch_outliers,
                     build_banking_table, catchment_branches)
from interpret import band, guide_banking, rating
from data_sources import MARKET_COLS, MARKETS, load_banking
from routing import DRIVE_MINUTES, load_catchments, ors_key

NAVY, TEAL = "#0b1f3a", "#0e7490"
COVERAGE_ORDER = ["Underserved", "Balanced", "Saturated", "No branch in ZIP", "n/a"]


# One-tap ZIP groups so nobody has to type 5-digit codes on a phone. Add or edit freely (Florida ZIPs only).
METROS = {
    "Miami urban core": ["33130", "33132", "33133", "33135", "33137", "33140", "33141", "33143", "33145", "33156"],
    "Broward core": ["33304", "33309", "33312", "33317", "33321", "33324", "33020", "33024", "33062", "33309"],
    "Palm Beach": ["33401", "33405", "33407", "33411", "33431", "33432", "33496", "33458"],
    "Tampa Bay": ["33606", "33607", "33609", "33611", "33629", "33701", "33704", "33707", "33713", "33761"],
    "Orlando area": ["32803", "32804", "32806", "32819", "32835", "32839", "32792"],
    "Jacksonville": ["32204", "32207", "32210", "32216", "32224", "32246"],
    "Southwest Florida": ["34103", "34108", "34109", "33904", "33908", "34231", "34202"],
}
MAX_EXTRA_ZIPS = 25


SPOT_CSS = """
<style>
.sp-banner { display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;
  background:linear-gradient(120deg,#0b1f3a 0%,#0e4d64 60%,#0e7490 100%); color:#fff; border-radius:16px;
  padding:18px 22px; margin:6px 0 14px 0; box-shadow:0 6px 20px rgba(11,31,58,.15); }
.sp-eyebrow { letter-spacing:.16em; font-size:.68rem; font-weight:600; color:#7dd3e0; }
.sp-title { font-size:1.55rem; font-weight:800; line-height:1.15; overflow-wrap:anywhere; }
.sp-sub { font-size:.86rem; color:#cfe8ee; margin-top:2px; }
.sp-pills { display:flex; gap:8px; flex-wrap:wrap; }
.sp-pill { display:inline-flex; flex-direction:column; border-radius:12px; padding:6px 12px; min-width:104px; }
.sp-pill small { font-size:.62rem; letter-spacing:.08em; text-transform:uppercase; opacity:.75; font-weight:600; }
.sp-pill b { font-size:.82rem; font-weight:700; line-height:1.2; }
.sp-group { font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; color:#0e7490; margin:10px 0 6px 2px; }
.sp-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
.sp-card { background:#fff; border:1px solid #e5eaf1; border-radius:12px; padding:10px 14px; min-width:0; }
.sp-label { font-size:.68rem; text-transform:uppercase; letter-spacing:.07em; color:#64748b; font-weight:600; }
.sp-value { font-size:1.2rem; font-weight:800; color:#0b1f3a; line-height:1.25; margin-top:1px; overflow-wrap:anywhere; }
.sp-long { font-size:.95rem; line-height:1.3; }
.sp-note { font-size:.72rem; color:#0e7490; margin-top:1px; overflow-wrap:anywhere; }
.sp-read { margin:12px 0 10px 0; padding:10px 14px; border-left:4px solid #0e7490; background:rgba(14,116,144,.08);
  border-radius:0 10px 10px 0; font-size:.9rem; }
</style>
"""
COVER_STYLE = {"Underserved": ("#dff3f6", "#0b5f73"), "Balanced": ("#e8eef5", "#0b1f3a"),
               "Saturated": ("#fef3c7", "#92400e"), "No branch in ZIP": ("#fee2e2", "#991b1b")}
CAPTURE_STYLE = {"Under-captured": ("#dff3f6", "#0b5f73"), "Typical": ("#e8eef5", "#0b1f3a"),
                 "Deposit-rich": ("#e0e7ff", "#312e81")}
HHI_STYLE = {"Highly concentrated": ("#fef3c7", "#92400e"), "Moderately concentrated": ("#e8eef5", "#0b1f3a"),
             "Fragmented": ("#dff3f6", "#0b5f73")}


def _pill(label: str, value: str, styles: dict) -> str:
    bg, fg = styles.get(value, ("#e8eef5", "#0b1f3a"))
    return f'<div class="sp-pill" style="background:{bg};color:{fg}"><small>{escape(label)}</small><b>{escape(str(value))}</b></div>'


def _pill_color(label: str, value: str, color: str) -> str:
    return (f'<div class="sp-pill" style="background:{color};color:#fff"><small style="opacity:.85">{escape(label)}</small>'
            f'<b>{escape(value)}</b></div>')


def _cards(items: list[tuple[str, str, str]]) -> str:
    inner = "".join(f'<div class="sp-card"><div class="sp-label">{escape(l)}</div>'
                    f'<div class="sp-value{" sp-long" if len(str(v)) > 16 else ""}">{escape(str(v))}</div>'
                    f'<div class="sp-note">{escape(n)}</div></div>' for l, v, n in items)
    return f'<div class="sp-grid">{inner}</div>'


def _money(v) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    return f"${v / 1e9:,.1f}B" if v >= 1e9 else f"${v / 1e6:,.0f}M" if v >= 1e6 else f"${v / 1e3:,.0f}K"


def _kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="kpi-label">{escape(label)}</div>'
            f'<div class="kpi-value">{escape(value)}</div><div class="kpi-sub">{escape(sub)}</div></div>')


def _parse_custom_zips(text: str, existing: set[str]) -> tuple[list[str], list[str]]:
    good, bad = [], []
    for token in re.findall(r"\d+", text or ""):
        if re.fullmatch(r"3[234]\d{3}", token) and token not in existing and token not in good:
            good.append(token)
        else:
            bad.append(token)
    return good[:MAX_EXTRA_ZIPS], bad


def _color(score: float) -> list[int]:
    t = float(np.clip(score / 100, 0, 1))
    low, high = np.array([186, 220, 228]), np.array([11, 31, 58])
    return [int(v) for v in low + (high - low) * t] + [215]


def render() -> None:
    st.markdown('<div class="section-title">Banking: small businesses & branches by ZIP</div>'
                '<div class="section-sub">Where are small businesses concentrated, and how many bank branches serve them? '
                'Sources: Census Business Patterns and the FDIC Summary of Deposits.</div>', unsafe_allow_html=True)

    base = pd.DataFrame(MARKETS, columns=MARKET_COLS)[["zip_code", "city", "county", "latitude", "longitude"]]

    c1, c2, c3 = st.columns([3, 3, 4])
    has_ors = bool(ors_key())
    mode = c1.radio("Competitor area", ["Straight-line radius", "Drive time"], horizontal=True, key="bk_mode",
                    help="Drive time uses road networks (OpenRouteService) so canals, bays and highways are respected.")
    drive_min = 10
    if mode == "Drive time" and has_ors:
        drive_min = c1.select_slider("Minutes by car", options=list(DRIVE_MINUTES), value=10, key="bk_min")
        radius = 3
    else:
        radius = c1.slider("Radius (miles)", 1, 10, 3, key="bk_radius")
    small_def = c2.radio("Small business means", SMALL_DEFS, horizontal=True)
    metros = c3.multiselect("➕ Add a metro area", list(METROS), key="bk_metros",
                            placeholder="Choose one or more areas",
                            help="Adds a group of ZIP codes to the analysis, so you do not have to type them.")
    extra_text = c3.text_input("…or type ZIP codes", placeholder="e.g. 33321, 33024",
                               help=f"Any Florida ZIP (32xxx-34xxx). Up to {MAX_EXTRA_ZIPS} extra in total.")
    use_drive = mode == "Drive time" and has_ors
    if mode == "Drive time" and not has_ors:
        st.info("Drive-time areas need a free OpenRouteService key. Add `ORS_API_KEY` in Secrets "
                "(openrouteservice.org, free sign-up). Using straight-line radius until then.")
    typed = " ".join([extra_text or ""] + [z for m_ in metros for z in METROS[m_]])
    custom, ignored = _parse_custom_zips(typed, set(base["zip_code"]))
    if ignored:
        st.caption(f"Ignored (not a new Florida ZIP): {', '.join(ignored)}")
    with st.expander("⚖️ Opportunity weights", expanded=False):
        w_vol = st.slider("Number of small businesses", 0, 100, DEFAULT_WEIGHTS["volume"], key="bk_w1")
        w_und = st.slider("Small businesses per branch (under-served)", 0, 100, DEFAULT_WEIGHTS["underserved"], key="bk_w2")
        w_den = st.slider("Small businesses per 10,000 residents", 0, 100, DEFAULT_WEIGHTS["density"], key="bk_w3")
    if w_vol + w_und + w_den == 0:
        st.error("Set at least one weight above zero.")
        return

    universe = pd.concat([base, pd.DataFrame({"zip_code": custom})], ignore_index=True)
    data = load_banking(tuple(sorted(universe["zip_code"])))
    status = data["status"]

    if data["cbp"] is None or data["branches"] is None:
        st.session_state.pop("banking_ctx", None)
        st.session_state["banking_status"] = status
        load_banking.clear()  # never keep a failed load; retry on the next run
        st.warning("Banking data isn't available yet. Both the Census business data and the FDIC branch data are needed.")
        st.dataframe(status, hide_index=True, width="stretch")
        st.markdown("**Fix:** Census rejects requests without a key, so add a free `CENSUS_API_KEY` "
                    "(api.census.gov/data/key_signup.html) to your secrets, then click **Refresh live data** in the "
                    "sidebar. For anything else, run the **Connection check** on the Data Sources tab.")
        return

    branches = data["branches"]
    if data["population"] is not None:
        universe = universe.merge(data["population"][["zip_code", "population"]], on="zip_code", how="left")
    # Name custom ZIPs after the most common branch city in that ZIP.
    city_by_zip = branches.groupby("zip5")["city"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "")
    universe["city"] = universe["city"].fillna(universe["zip_code"].map(city_by_zip)).fillna("")
    universe["city"] = [c if c else f"ZIP {z}" for c, z in zip(universe["city"], universe["zip_code"])]

    catchments, catch_errors = {}, {}
    if use_drive:
        # Points: each ZIP's market coordinates, else the average location of its branches.
        cen = branches.dropna(subset=["lat", "lon"]).groupby("zip5")[["lat", "lon"]].mean()
        pts = []
        for z, la, lo in zip(universe["zip_code"], universe["latitude"], universe["longitude"]):
            if pd.isna(la) and z in cen.index:
                la, lo = cen.loc[z, "lat"], cen.loc[z, "lon"]
            pts.append((z, None if pd.isna(la) else float(la), None if pd.isna(lo) else float(lo)))
        catchments, catch_errors = load_catchments(tuple(pts), int(drive_min), ors_key())
        if catch_errors:
            load_catchments.clear()  # do not keep failures for a week
        if catch_errors and not catchments:
            st.warning("Drive-time areas could not be drawn, so straight-line radius is used. First error: "
                       + next(iter(catch_errors.values())))
            use_drive = False
        elif catch_errors:
            st.caption(f"Drive time unavailable for {len(catch_errors)} ZIP(s); those use a {radius}-mile radius.")
    catch_label = f"{drive_min}-min drive" if use_drive else f"{radius} mi"

    table = build_banking_table(universe, data["cbp"], branches, radius, small_def,
                                {"volume": w_vol, "underserved": w_und, "density": w_den},
                                catchments=catchments if use_drive else None)
    table["label"] = table["city"] + " (" + table["zip_code"] + ")"
    st.session_state.pop("banking_status", None)
    st.session_state["banking_ctx"] = {"table": table, "branches": branches, "radius": radius,
                                       "small_def": small_def, "catchments": catchments if use_drive else {},
                                       "catch_label": catch_label, "drive_minutes": int(drive_min) if use_drive else None}

    n_live = int((status["State"] == "Live").sum())
    st.caption(f"Data sources live: {n_live}/{len(status)} · refreshed {data['loaded_at']}. "
               "Details are in the Data Sources tab.")

    # ---- KPIs
    top = table.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(_kpi("Small businesses", f'{table["small_biz"].sum():,.0f}', f"{small_def.lower()}, {len(table)} ZIPs"),
                unsafe_allow_html=True)
    k2.markdown(_kpi("Bank branches", f'{table["branches_in_zip"].sum():,.0f}', "inside these ZIPs"),
                unsafe_allow_html=True)
    med = table.loc[table["branches_in_zip"] > 0, "small_biz_per_branch"].median()
    k3.markdown(_kpi("Median per branch", "n/a" if pd.isna(med) else f"{med:,.0f}", "small businesses per branch"),
                unsafe_allow_html=True)
    k4.markdown(_kpi("Top opportunity", top["city"], f'ZIP {top["zip_code"]} · {rating(top["opportunity"])} ({top["opportunity"]:.0f})'),
                unsafe_allow_html=True)
    st.write("")

    # ---- Map + ranking
    c_map, c_rank = st.columns([6, 5], gap="large")
    with c_map:
        mk = table.dropna(subset=["latitude", "longitude"]).copy()
        mk["color"] = mk["opportunity"].map(_color)
        mk["radius"] = 1500 + mk["opportunity"] * 60
        mk["tip"] = (f"<b>" + mk["label"] + "</b><br/>Opportunity " + mk["opportunity"].round(0).astype(int).astype(str)
                     + " · " + mk["coverage"] + "<br/>" + mk["small_biz"].round(0).astype(int).astype(str)
                     + " small businesses · " + mk["branches_in_zip"].astype(int).astype(str) + " branches")
        bk = branches[branches["zip5"].isin(table["zip_code"])].dropna(subset=["lat", "lon"]).copy()
        bk["tip"] = ("<b>" + bk["bank"] + "</b><br/>" + bk["address"].astype(str) + ", " + bk["city"].astype(str)
                     + "<br/>$" + (bk["deposits_usd"] / 1e6).round(1).astype(str) + "M deposits")
        layers = [
            pdk.Layer("ScatterplotLayer", data=mk[["longitude", "latitude", "color", "radius", "tip"]],
                      get_position="[longitude, latitude]", get_fill_color="color", get_radius="radius",
                      pickable=True, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1.5),
            pdk.Layer("ScatterplotLayer", data=bk[["lon", "lat", "tip"]], get_position="[lon, lat]",
                      get_fill_color=[245, 158, 11, 220], get_radius=120, radius_min_pixels=3, pickable=True),
        ]
        center = (mk["latitude"].mean(), mk["longitude"].mean()) if len(mk) else (28.1, -82.4)
        st.markdown("**Map:** circles = ZIP markets (darker = higher opportunity), orange dots = bank branches.")
        st.pydeck_chart(pdk.Deck(layers=layers,
                                 initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=5.5),
                                 tooltip={"html": "{tip}", "style": {"backgroundColor": NAVY, "color": "white"}}),
                        height=480)
    with c_rank:
        st.markdown("**Banking opportunity ranking**")
        chart = (alt.Chart(table).mark_bar(cornerRadiusEnd=5, height=18)
                 .encode(x=alt.X("opportunity:Q", title=None, scale=alt.Scale(domain=[0, 100])),
                         y=alt.Y("label:N", sort="-x", title=None),
                         color=alt.Color("opportunity:Q", scale=alt.Scale(range=["#a5d8d4", TEAL, NAVY]), legend=None),
                         tooltip=["label", alt.Tooltip("opportunity:Q", format=".0f"), "coverage",
                                  alt.Tooltip("small_biz:Q", format=",.0f", title="Small businesses"),
                                  alt.Tooltip("branches_in_zip:Q", title="Branches in ZIP")])
                 .properties(height=max(260, 28 * len(table))))
        st.altair_chart(chart, width="stretch")

    guide_banking(st)

    # ---- Matrix
    st.markdown("---")
    st.markdown('<div class="section-title">ZIP matrix</div>'
                '<div class="section-sub">Switch views, click a header to sort, download the result.</div>',
                unsafe_allow_html=True)
    f1, f2 = st.columns([2, 3])
    q = f1.text_input("🔍 Search city or ZIP", key="bk_search")
    view_name = f2.radio("View", ["Opportunity", "Small business", "Banks & deposits", "Competitor vulnerability"],
                         horizontal=True, key="bk_view")
    mx = table.copy()
    mx["rating"] = mx["opportunity"].map(rating)
    if q:
        mx = mx[mx["label"].str.lower().str.contains(q.lower())]
    mx["deposits_in_zip_m"] = mx["deposits_in_zip"] / 1e6
    mx["deposits_nearby_m"] = mx["deposits_nearby"] / 1e6
    mx["deposits_per_small_biz_k"] = mx["deposits_per_small_biz"] / 1e3
    VIEWS = {
        "Opportunity": ["rank", "city", "zip_code", "opportunity", "rating", "coverage", "small_biz", "branches_in_zip",
                        "small_biz_per_branch", "branches_nearby"],
        "Small business": ["city", "zip_code", "small_biz", "estab_lt5", "estab_5_19", "estab_20plus", "estab_total",
                           "employees", "small_biz_per_10k_pop", "population"],
        "Banks & deposits": ["city", "zip_code", "branches_in_zip", "banks_in_zip", "deposits_in_zip_m",
                             "deposits_per_small_biz_k", "top_bank", "top_bank_share_pct", "deposit_hhi",
                             "branches_nearby", "deposits_nearby_m"],
        "Competitor vulnerability": ["city", "zip_code", "capture_index", "capture_label", "hhi_used", "hhi_label",
                                     "top_bank", "top_bank_share_pct", "underperforming_branches", "branches_nearby",
                                     "deposits_per_small_biz_k"],
    }
    cc = st.column_config
    config = {
        "rank": cc.NumberColumn("#", width="small"), "city": cc.TextColumn("Market"), "zip_code": cc.TextColumn("ZIP"),
        "opportunity": cc.ProgressColumn("Opportunity", min_value=0, max_value=100, format="%.0f"),
        "rating": cc.TextColumn("Rating", help="80+ Exceptional, 65+ Strong, 50+ Solid, 35+ Moderate, below Weak"),
        "coverage": cc.TextColumn("Coverage", help="Compared with the median small-businesses-per-branch of these ZIPs"),
        "small_biz": cc.NumberColumn("Small businesses", format="%d"),
        "estab_lt5": cc.NumberColumn("Under 5 emp.", format="%d"),
        "estab_5_19": cc.NumberColumn("5 to 19 emp.", format="%d"),
        "estab_20plus": cc.NumberColumn("20+ emp.", format="%d"),
        "estab_total": cc.NumberColumn("All employer businesses", format="%d"),
        "employees": cc.NumberColumn("Employees", format="%d"),
        "small_biz_per_10k_pop": cc.NumberColumn("Per 10k residents", format="%.0f"),
        "population": cc.NumberColumn("Residents", format="%d"),
        "branches_in_zip": cc.NumberColumn("Branches in ZIP", format="%d"),
        "banks_in_zip": cc.NumberColumn("Banks in ZIP", format="%d"),
        "small_biz_per_branch": cc.NumberColumn("Businesses per branch", format="%.0f"),
        "branches_nearby": cc.NumberColumn(f"Branches within {catch_label}", format="%d"),
        "capture_index": cc.NumberColumn("Deposit capture index", format="%.0f",
                                         help="Deposits per small business vs the median ZIP (100 = median). "
                                              "Low = deposits are thin relative to the businesses there."),
        "capture_label": cc.TextColumn("Capture"),
        "hhi_used": cc.NumberColumn("Deposit HHI (area)", format="%.0f",
                                    help="Deposit concentration among banks in the competitor area (0-10,000)."),
        "hhi_label": cc.TextColumn("Concentration"),
        "underperforming_branches": cc.NumberColumn("Weak branches", format="%d",
                                                    help="Branches booking under half the median branch in the area."),
        "deposits_in_zip_m": cc.NumberColumn("Deposits in ZIP", format="$%.1fM"),
        "deposits_nearby_m": cc.NumberColumn(f"Deposits within {catch_label}", format="$%.1fM"),
        "deposits_per_small_biz_k": cc.NumberColumn("Deposits per small biz", format="$%.0fK"),
        "top_bank": cc.TextColumn("Largest bank"),
        "top_bank_share_pct": cc.NumberColumn("Largest bank share", format="%.0f%%"),
        "deposit_hhi": cc.NumberColumn("Deposit HHI", format="%.0f",
                                       help="Concentration of deposits among banks in the ZIP (0-10,000). Informational only."),
    }
    cols = VIEWS[view_name]
    st.dataframe(mx[cols], column_config=config, hide_index=True, width="stretch", height=min(80 + 36 * len(mx), 600))
    st.download_button("⬇️ Download this view (CSV)", mx[cols].to_csv(index=False).encode(),
                       file_name="synoptic_bi_banking_zips.csv", mime="text/csv")

    # ---- Spotlight
    st.markdown("---")
    st.markdown(SPOT_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔎 ZIP spotlight: who is competing here?</div>'
                '<div class="section-sub">Pick a ZIP to see demand, competition and the branches around it.</div>',
                unsafe_allow_html=True)
    pick = st.selectbox("ZIP", table["label"], label_visibility="collapsed", key="bk_pick")
    row = table[table["label"] == pick].iloc[0]

    ring = catchments.get(row["zip_code"]) if use_drive else None
    nb = catchment_branches(branches, row["latitude"], row["longitude"], radius, ring)
    scope = catch_label if (ring is not None or not nb.empty) else "this ZIP"
    if nb.empty:  # no coordinates available: fall back to branches inside the ZIP itself
        nb = branches[branches["zip5"] == row["zip_code"]].assign(distance_mi=np.nan).sort_values(
            "deposits_usd", ascending=False)
    out = branch_outliers(nb) if not nb.empty else nb
    flagged = (out[out["flag"] != "In line"] if not nb.empty else nb)

    def num(v, fmt="{:,.0f}"):
        return "n/a" if pd.isna(v) else fmt.format(v)

    opp_txt = f'{row["opportunity"]:.0f} · {rating(row["opportunity"])}'
    opp_color = band(row["opportunity"])[1]
    # Banner: who and where, with the status labels as pills.
    st.markdown(
        f'<div class="sp-banner"><div><div class="sp-eyebrow">ZIP {escape(str(row["zip_code"]))}</div>'
        f'<div class="sp-title">{escape(str(row["city"]))}</div>'
        f'<div class="sp-sub">{len(nb)} branches within {escape(scope)} · {num(row["small_biz"])} small businesses</div></div>'
        f'<div class="sp-pills">{_pill_color("Opportunity", opp_txt, opp_color)}'
        f'{_pill("Coverage", row["coverage"], COVER_STYLE)}'
        f'{_pill("Capture", row["capture_label"], CAPTURE_STYLE)}'
        f'{_pill("Deposits", row["hhi_label"], HHI_STYLE)}</div></div>', unsafe_allow_html=True)

    top_name = (str(row["top_bank"]).replace(", National Association", " N.A.").replace(" National Association", " N.A.")
                if pd.notna(row["top_bank"]) and row["top_bank"] else "None")
    share_txt = "" if pd.isna(row["top_bank_share_pct"]) else f'{row["top_bank_share_pct"]:.0f}% of ZIP deposits'
    demand = [("Small businesses", num(row["small_biz"]), "employer locations"),
              ("Branches in ZIP", num(row["branches_in_zip"]), f'{num(row["banks_in_zip"])} different banks'),
              ("Businesses per branch", num(row["small_biz_per_branch"]), "higher = fewer banks per business"),
              ("Deposits in ZIP", _money(row["deposits_in_zip"]), "booked at branches in the ZIP")]
    compete = [("Deposit capture index", num(row["capture_index"]), "100 = median ZIP"),
               ("Deposit HHI", num(row["hhi_used"]), f"area concentration (0-10,000)"),
               ("Largest bank", top_name, share_txt),
               ("Weak branches", num(row["underperforming_branches"]), "under half the median branch")]
    st.markdown(f'<div class="sp-group">Demand</div>{_cards(demand)}<div class="sp-group">Competition</div>{_cards(compete)}',
                unsafe_allow_html=True)

    # One plain-English reading so the numbers are not left to the reader.
    bits = [f'{row["coverage"].lower()} coverage' if row["coverage"] not in ("n/a",) else "coverage not rated"]
    if pd.notna(row["capture_index"]):
        bits.append(f'deposits per small business are {row["capture_label"].lower()} (index {row["capture_index"]:.0f})')
    if pd.notna(row["hhi_used"]):
        bits.append(f'{row["hhi_label"].lower()} deposits')
    if pd.notna(row["underperforming_branches"]) and row["underperforming_branches"] > 0:
        bits.append(f'{row["underperforming_branches"]:.0f} weak branch(es) nearby')
    st.markdown(f'<div class="sp-read"><b>In short:</b> {escape("; ".join(bits))}.</div>', unsafe_allow_html=True)

    if nb.empty:
        st.info("No FDIC-insured branches found in this area.")
    else:
        t_comp, t_share, t_out, t_map = st.tabs(["🏦 Branches", "📊 Deposit share", f"⚠️ Outliers ({len(flagged)})", "🗺️ Map"])
        with t_comp:
            area_total = float(nb["deposits_usd"].sum())
            show = nb.assign(deposits_m=nb["deposits_usd"] / 1e6,
                             where=nb["address"].astype(str).str.strip() + ", " + nb["zip5"].astype(str),
                             share=(nb["deposits_usd"] / area_total * 100) if area_total > 0 else 0.0)
            full = st.toggle(f"Show all {len(show)} branches", value=False, key="bk_all") if len(show) > 12 else True
            show = show if full else show.head(12)
            st.dataframe(show[["bank", "where", "distance_mi", "deposits_m", "share"]], hide_index=True, width="stretch",
                         height=min(60 + 36 * len(show), 480),
                         column_config={"bank": cc.TextColumn("Bank", width="medium"),
                                        "where": cc.TextColumn("Location", width="large"),
                                        "distance_mi": cc.NumberColumn("Miles", format="%.1f", width="small"),
                                        "deposits_m": cc.NumberColumn("Deposits", format="$%.0fM", width="small"),
                                        "share": cc.NumberColumn("Share", format="%.1f%%", width="small",
                                                                 help="Share of deposits in the competitor area")})
            if not full:
                st.caption("Largest 12 shown, by deposits.")
        with t_share:
            share = bank_share(nb).head(8)
            st.altair_chart(
                alt.Chart(share).mark_bar(cornerRadiusEnd=5, color=TEAL, height=18).encode(
                    x=alt.X("share_pct:Q", title=f"Share of deposits within {scope} (%)"),
                    y=alt.Y("bank:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
                    tooltip=["bank", alt.Tooltip("share_pct:Q", format=".1f"), alt.Tooltip("branches:Q")],
                ).properties(height=max(160, 30 * len(share))), width="stretch")
        with t_out:
            if flagged.empty:
                st.success("No branches stand out from the median branch in this area.")
            else:
                st.caption("Underperforming = under half the median branch's deposits (possible weak branch, acquisition "
                           "or share-gain target). Overperforming = over twice the median, often a headquarters or "
                           "online-booked branch, so verify before drawing conclusions.")
                fl = flagged.assign(deposits_m=lambda d: d["deposits_usd"] / 1e6, peer_m=lambda d: d["peer_median_usd"] / 1e6)
                st.dataframe(fl.sort_values("vs_peer")[["flag", "bank", "address", "deposits_m", "peer_m", "vs_peer"]],
                             hide_index=True, width="stretch",
                             column_config={"flag": cc.TextColumn("Flag", width="small"), "bank": cc.TextColumn("Bank", width="medium"),
                                            "address": cc.TextColumn("Address", width="large"),
                                            "deposits_m": cc.NumberColumn("Deposits", format="$%.0fM", width="small"),
                                            "peer_m": cc.NumberColumn("Median branch", format="$%.0fM", width="small"),
                                            "vs_peer": cc.NumberColumn("× median", format="%.2f", width="small")})
        with t_map:
            pts = nb.dropna(subset=["lat", "lon"]).assign(
                tip=lambda d: "<b>" + d["bank"] + "</b><br/>" + d["address"].astype(str) + "<br/>$"
                              + (d["deposits_usd"] / 1e6).round(1).astype(str) + "M")
            layers = []
            if ring is not None and len(ring) > 2:
                layers.append(pdk.Layer("PolygonLayer", data=pd.DataFrame({"polygon": [[list(p) for p in ring]]}),
                                        get_polygon="polygon", get_fill_color=[14, 116, 144, 45],
                                        get_line_color=[14, 116, 144, 220], line_width_min_pixels=2))
            layers.append(pdk.Layer("ScatterplotLayer", data=pts[["lon", "lat", "tip"]], get_position="[lon, lat]",
                                    get_fill_color=[245, 158, 11, 230], get_radius=90, radius_min_pixels=4, pickable=True))
            if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
                layers.append(pdk.Layer("ScatterplotLayer",
                                        data=pd.DataFrame({"lon": [row["longitude"]], "lat": [row["latitude"]],
                                                           "tip": [f'ZIP {row["zip_code"]}']}),
                                        get_position="[lon, lat]", get_fill_color=[11, 31, 58, 240], get_radius=140,
                                        radius_min_pixels=7, pickable=True))
            clat = float(row["latitude"]) if pd.notna(row["latitude"]) else float(pts["lat"].mean())
            clon = float(row["longitude"]) if pd.notna(row["longitude"]) else float(pts["lon"].mean())
            st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=clat, longitude=clon, zoom=11),
                                     tooltip={"html": "{tip}", "style": {"backgroundColor": NAVY, "color": "white"}}),
                            height=380)
            st.caption("Navy dot = the ZIP point, orange dots = bank branches" + (", teal area = drive-time catchment." if ring is not None else "."))

    st.caption("How to read this: counts are employer establishments (locations), not companies, and exclude sole "
               "proprietors with no employees. Branch deposits are as booked at each branch and can be distorted by "
               "online or headquarters branches. Coverage and saturation are relative to the ZIPs shown. Business and "
               "branch data only: do not use this to include or exclude communities from lending or services on the "
               "basis of race or other protected characteristics.")
