"""Batch tab: rank a pipeline of candidate sites (ZIPs or addresses) with pass/fail thresholds."""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import streamlit as st

from banking import DEFAULT_WEIGHTS, SMALL_DEFS, build_banking_table
from data_sources import MARKET_COLS, MARKETS, load_banking

MAX_SITES = 60
FL_ZIP = re.compile(r"(?<!\d)(3[234]\d{3})(?!\d)")
ANY_ZIP = re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)")
ZIP_COLS = ("zip", "zipcode", "zip_code", "zip code", "postal", "postal_code", "postcode", "postal code")
LABEL_COLS = ("name", "site", "site_name", "location", "label", "address", "street", "id")
TEMPLATE = "name,address,zip\nBrickell branch,801 Brickell Ave Miami FL,33131\nTampa site,,33602\nOrlando site,400 S Orange Ave Orlando FL 32801,\n"


def parse_sites(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (sites[label, zip], skipped_messages). A ZIP column is used when present; otherwise the last
    Florida ZIP found in the row's text (so a full address works). Street addresses are NOT geocoded."""
    cols = {str(c).strip().lower(): c for c in df.columns}
    zip_col = next((cols[c] for c in ZIP_COLS if c in cols), None)
    label_col = next((cols[c] for c in LABEL_COLS if c in cols), None)
    sites, skipped = [], []
    for i, row in df.iterrows():
        text_all = " ".join(str(v) for v in row.values if pd.notna(v))
        z = None
        if zip_col is not None and pd.notna(row[zip_col]):
            m = ANY_ZIP.search(str(row[zip_col]))
            z = m.group(1) if m else None
        if z is None:
            found = FL_ZIP.findall(text_all)
            z = found[-1] if found else None
        label = str(row[label_col]).strip() if label_col is not None and pd.notna(row[label_col]) else ""
        if not z:
            skipped.append(f"Row {i + 1} ({label or text_all[:40] or 'blank'}): no ZIP code found")
        elif not FL_ZIP.fullmatch(z):
            skipped.append(f"Row {i + 1} ({label or z}): ZIP {z} is outside Florida")
        else:
            sites.append({"site": label or f"ZIP {z}", "zip_code": z})
    return pd.DataFrame(sites, columns=["site", "zip_code"]), skipped


def _lines_to_df(text: str) -> pd.DataFrame:
    rows = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return pd.DataFrame({"address": rows})


def render() -> None:
    st.markdown('<div class="section-title">Batch ranking: score a pipeline of sites</div>'
                '<div class="section-sub">Upload a CSV (or paste a list) of up to 60 Florida ZIP codes or addresses. '
                'Every site is scored together and checked against your thresholds.</div>', unsafe_allow_html=True)

    ctx = st.session_state.get("banking_ctx") or {}
    radius = ctx.get("radius", 3)
    small_def = ctx.get("small_def", SMALL_DEFS[0])
    catchments = ctx.get("catchments") or {}

    c1, c2 = st.columns([3, 2], gap="large")
    up = c1.file_uploader("CSV file", type=["csv"], key="batch_file",
                          help="Needs a ZIP column, or an address column that includes the ZIP code.")
    pasted = c2.text_area("…or paste one ZIP or address per line", height=110, key="batch_text",
                          placeholder="33131\n33602\n400 S Orange Ave, Orlando, FL 32801")
    st.download_button("⬇️ Download a CSV template", TEMPLATE.encode(), file_name="synoptic_bi_sites_template.csv",
                       mime="text/csv")

    raw = None
    if up is not None:
        try:
            raw = pd.read_csv(io.BytesIO(up.getvalue()), dtype=str, keep_default_na=False).replace("", np.nan)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read that file as a CSV ({type(exc).__name__}). Save it as CSV (comma separated) and retry.")
            return
    elif pasted.strip():
        raw = _lines_to_df(pasted)
    if raw is None or raw.empty:
        st.info("Upload a CSV or paste a list to begin. Only the ZIP code is used for scoring: street addresses are "
                "not geocoded, so two sites in the same ZIP get the same market metrics.")
        return

    sites, skipped = parse_sites(raw.head(MAX_SITES * 3))
    if len(sites) > MAX_SITES:
        skipped.append(f"Only the first {MAX_SITES} sites were used.")
        sites = sites.head(MAX_SITES)
    if skipped:
        with st.expander(f"{len(skipped)} row(s) skipped", expanded=False):
            for m in skipped:
                st.write("• " + m)
    if sites.empty:
        st.warning("No Florida ZIP codes were found. Add a ZIP column, or include the 5-digit ZIP in each address.")
        return

    zips = sorted(set(sites["zip_code"]))
    known = pd.DataFrame(MARKETS, columns=MARKET_COLS)[["zip_code", "city", "county", "latitude", "longitude"]]
    universe = pd.DataFrame({"zip_code": zips}).merge(known, on="zip_code", how="left")
    data = load_banking(tuple(zips))
    if data["cbp"] is None or data["branches"] is None:
        st.warning("Banking data isn't available yet, so sites can't be scored. Open the Banking tab or the "
                   "Data Sources tab to see which source is failing.")
        st.dataframe(data["status"], hide_index=True, width="stretch")
        return

    branches = data["branches"]
    pop = data["population"]
    if pop is not None:
        universe = universe.merge(pop[["zip_code", "population"]], on="zip_code", how="left")
    city_by_zip = branches.groupby("zip5")["city"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "")
    universe["city"] = universe["city"].fillna(universe["zip_code"].map(city_by_zip)).fillna("")
    universe["city"] = [c if c else f"ZIP {z}" for c, z in zip(universe["city"], universe["zip_code"])]
    weights = {"volume": st.session_state.get("bk_w1", DEFAULT_WEIGHTS["volume"]),
               "underserved": st.session_state.get("bk_w2", DEFAULT_WEIGHTS["underserved"]),
               "density": st.session_state.get("bk_w3", DEFAULT_WEIGHTS["density"])}
    tbl = build_banking_table(universe, data["cbp"], branches, radius, small_def, weights,
                              catchments={z: r for z, r in catchments.items() if z in zips})
    if pop is not None:
        extra = pop.assign(bachelors_plus_pct=pop["edu_bachelor_pct"] + pop["edu_graduate_pct"])[
            ["zip_code", "median_income", "bachelors_plus_pct"]]
        tbl = tbl.merge(extra, on="zip_code", how="left")

    # ---- thresholds
    st.markdown("**Pass / fail thresholds** (a site passes when it meets every enabled test)")
    t1, t2, t3, t4 = st.columns(4)
    min_opp = t1.slider("Opportunity at least", 0, 100, 50, key="bt_opp")
    min_biz = t2.number_input("Small businesses at least", 0, 100000, 300, step=50, key="bt_biz")
    min_per = t3.number_input("Businesses per branch at least", 0, 5000, 0, step=25, key="bt_per",
                              help="0 turns this test off.")
    skip_sat = t4.checkbox("Exclude saturated ZIPs", value=True, key="bt_sat")
    skip_hhi = t4.checkbox("Exclude highly concentrated deposits", value=False, key="bt_hhi")

    res = sites.merge(tbl, on="zip_code", how="left")
    fails = []
    for _, r in res.iterrows():
        f = []
        if r["opportunity"] < min_opp:
            f.append("opportunity")
        if r["small_biz"] < min_biz:
            f.append("small businesses")
        if min_per and r["small_biz_per_branch"] < min_per:
            f.append("per branch")
        if skip_sat and r["coverage"] == "Saturated":
            f.append("saturated")
        if skip_hhi and r["hhi_label"] == "Highly concentrated":
            f.append("concentrated deposits")
        fails.append(", ".join(f))
    res["fails"] = fails
    res["passed"] = res["fails"] == ""
    res["result"] = np.where(res["passed"], "✅ Pass", "❌ Fail")
    res = res.sort_values(["passed", "opportunity"], ascending=[False, False]).reset_index(drop=True)
    res.insert(0, "site_rank", res.index + 1)

    n_pass = int((res["result"] == "✅ Pass").sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("Sites scored", len(res))
    k2.metric("Pass all tests", n_pass)
    k3.metric("Best site", res.iloc[0]["site"], f'{res.iloc[0]["opportunity"]:.0f} / 100', delta_color="off")

    cc = st.column_config
    cols = ["site_rank", "result", "site", "zip_code", "city", "opportunity", "coverage", "small_biz",
            "branches_in_zip", "small_biz_per_branch", "capture_index", "capture_label", "hhi_label", "median_income",
            "bachelors_plus_pct", "fails"]
    cols = [c for c in cols if c in res.columns]
    st.dataframe(res[cols], hide_index=True, width="stretch", height=min(80 + 36 * len(res), 640), column_config={
        "site_rank": cc.NumberColumn("#", width="small"), "result": cc.TextColumn("Result"),
        "site": cc.TextColumn("Site"), "zip_code": cc.TextColumn("ZIP"), "city": cc.TextColumn("Market"),
        "opportunity": cc.ProgressColumn("Opportunity", min_value=0, max_value=100, format="%.0f"),
        "coverage": cc.TextColumn("Coverage"), "small_biz": cc.NumberColumn("Small businesses", format="%d"),
        "branches_in_zip": cc.NumberColumn("Branches in ZIP", format="%d"),
        "small_biz_per_branch": cc.NumberColumn("Businesses per branch", format="%.0f"),
        "capture_index": cc.NumberColumn("Deposit capture", format="%.0f"), "capture_label": cc.TextColumn("Capture"),
        "hhi_label": cc.TextColumn("Deposit concentration"),
        "median_income": cc.NumberColumn("Median income", format="$%d"),
        "bachelors_plus_pct": cc.NumberColumn("Bachelor's+", format="%.0f%%"),
        "fails": cc.TextColumn("Failed tests")})
    st.download_button("⬇️ Download ranked sites (CSV)", res[cols].to_csv(index=False).encode(),
                       file_name="synoptic_bi_ranked_sites.csv", mime="text/csv")
    st.caption("Opportunity, coverage and capture are relative to the ZIPs in this upload, so adding or removing sites "
               f"changes them. Competitor area: {ctx.get('catch_label', f'{radius} mi')} (set in the Banking tab; "
               "ZIPs without a drive-time area use the radius). Business and branch data only: do not use this to "
               "include or exclude communities on the basis of race or other protected characteristics.")
