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
    k4.markdown(_kpi("Top opportunity", top["city"], f'ZIP {top["zip_code"]} · {top["coverage"]}'),
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
    if q:
        mx = mx[mx["label"].str.lower().str.contains(q.lower())]
    mx["deposits_in_zip_m"] = mx["deposits_in_zip"] / 1e6
    mx["deposits_nearby_m"] = mx["deposits_nearby"] / 1e6
    mx["deposits_per_small_biz_k"] = mx["deposits_per_small_biz"] / 1e3
    VIEWS = {
        "Opportunity": ["rank", "city", "zip_code", "opportunity", "coverage", "small_biz", "branches_in_zip",
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
    st.markdown('<div class="section-title">🔎 ZIP spotlight: who is competing here?</div>', unsafe_allow_html=True)
    pick = st.selectbox("ZIP", table["label"], label_visibility="collapsed", key="bk_pick")
    row = table[table["label"] == pick].iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Small businesses", f'{row["small_biz"]:,.0f}')
    m2.metric("Branches in ZIP", f'{row["branches_in_zip"]:,.0f}')
    m3.metric("Businesses per branch", f'{row["small_biz_per_branch"]:,.0f}')
    m4.metric("Deposits in ZIP", f'${row["deposits_in_zip"] / 1e6:,.1f}M')
    m5.metric("Coverage", row["coverage"])

    ring = catchments.get(row["zip_code"]) if use_drive else None
    nb = catchment_branches(branches, row["latitude"], row["longitude"], radius, ring)
    scope = catch_label if (ring is not None or not nb.empty) else "this ZIP"
    if nb.empty:  # no coordinates available: fall back to branches inside the ZIP itself
        nb = branches[branches["zip5"] == row["zip_code"]].assign(distance_mi=np.nan).sort_values(
            "deposits_usd", ascending=False)
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Deposit capture index", "n/a" if pd.isna(row["capture_index"]) else f'{row["capture_index"]:,.0f}',
              row["capture_label"], delta_color="off")
    v2.metric("Deposit HHI (area)", "n/a" if pd.isna(row["hhi_used"]) else f'{row["hhi_used"]:,.0f}',
              row["hhi_label"], delta_color="off")
    v3.metric("Largest bank in ZIP", str(row["top_bank"] or "none")[:28],
              "" if pd.isna(row["top_bank_share_pct"]) else f'{row["top_bank_share_pct"]:.0f}% of deposits', delta_color="off")
    v4.metric("Weak branches in area", "n/a" if pd.isna(row["underperforming_branches"]) else f'{row["underperforming_branches"]:.0f}')
    st.markdown(f"**{len(nb)} branches within {scope}** (largest deposits first)")
    if ring is not None and len(ring) > 2:
        poly = pd.DataFrame({"polygon": [[list(p) for p in ring]]})
        pts = nb.dropna(subset=["lat", "lon"]).assign(tip=lambda d: "<b>" + d["bank"] + "</b><br/>$"
                                                       + (d["deposits_usd"] / 1e6).round(1).astype(str) + "M")
        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer("PolygonLayer", data=poly, get_polygon="polygon", get_fill_color=[14, 116, 144, 45],
                              get_line_color=[14, 116, 144, 220], line_width_min_pixels=2),
                    pdk.Layer("ScatterplotLayer", data=pts[["lon", "lat", "tip"]], get_position="[lon, lat]",
                              get_fill_color=[245, 158, 11, 230], get_radius=90, radius_min_pixels=4, pickable=True)],
            initial_view_state=pdk.ViewState(latitude=float(row["latitude"]), longitude=float(row["longitude"]), zoom=10.5),
            tooltip={"html": "{tip}", "style": {"backgroundColor": NAVY, "color": "white"}}), height=360)
    if nb.empty:
        st.info("No FDIC-insured branches found in this area.")
    else:
        show = nb.assign(deposits_m=nb["deposits_usd"] / 1e6)[["bank", "address", "city", "zip5", "distance_mi", "deposits_m"]]
        st.dataframe(show, hide_index=True, width="stretch", height=min(80 + 36 * len(show), 420),
                     column_config={"bank": cc.TextColumn("Bank"), "address": cc.TextColumn("Address"),
                                    "city": cc.TextColumn("City"), "zip5": cc.TextColumn("ZIP"),
                                    "distance_mi": cc.NumberColumn("Miles", format="%.1f"),
                                    "deposits_m": cc.NumberColumn("Branch deposits", format="$%.1fM")})
        out = branch_outliers(nb)
        flagged = out[out["flag"] != "In line"].assign(deposits_m=lambda d: d["deposits_usd"] / 1e6,
                                                      peer_m=lambda d: d["peer_median_usd"] / 1e6)
        if not flagged.empty:
            with st.expander(f"Branch outliers vs the median branch in this area ({len(flagged)})", expanded=False):
                st.caption("Underperforming = under half the median branch's deposits (possible weak branch, "
                           "acquisition or share-gain target). Overperforming = over twice the median; often a "
                           "headquarters or online-booked branch, so verify before drawing conclusions.")
                st.dataframe(flagged.sort_values("vs_peer")[["flag", "bank", "address", "city", "deposits_m", "peer_m", "vs_peer"]],
                             hide_index=True, width="stretch",
                             column_config={"flag": cc.TextColumn("Flag"), "bank": cc.TextColumn("Bank"),
                                            "address": cc.TextColumn("Address"), "city": cc.TextColumn("City"),
                                            "deposits_m": cc.NumberColumn("Branch deposits", format="$%.1fM"),
                                            "peer_m": cc.NumberColumn("Median branch", format="$%.1fM"),
                                            "vs_peer": cc.NumberColumn("× median", format="%.2f")})
        share = bank_share(nb).head(8)
        st.altair_chart(
            alt.Chart(share).mark_bar(cornerRadiusEnd=5, color=TEAL, height=20).encode(
                x=alt.X("share_pct:Q", title=f"Share of deposits within {scope} (%)"),
                y=alt.Y("bank:N", sort="-x", title=None),
                tooltip=["bank", alt.Tooltip("share_pct:Q", format=".1f"), alt.Tooltip("branches:Q")],
            ).properties(height=max(160, 30 * len(share))),
            width="stretch")

    st.caption("How to read this: counts are employer establishments (locations), not companies, and exclude sole "
               "proprietors with no employees. Branch deposits are as booked at each branch and can be distorted by "
               "online or headquarters branches. Coverage and saturation are relative to the ZIPs shown. Business and "
               "branch data only: do not use this to include or exclude communities from lending or services on the "
               "basis of race or other protected characteristics.")
