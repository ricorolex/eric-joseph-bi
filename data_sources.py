"""Live data connectors for Synoptic BI.

Sources (all public):
  * U.S. Census Bureau ACS 5-year  -> income, education, race/ethnicity by ZIP (ZCTA)
  * BLS QCEW (annual)              -> sector Location Quotients by county
  * BEA Regional (CAGDP9)          -> county real GDP growth (needs a free API key)

Every connector degrades gracefully: if a source is unreachable, that component
falls back to clearly-labelled sample data and the UI says so.
"""
from __future__ import annotations

import io
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests
import streamlit as st

HEADERS = {"User-Agent": "SynopticBI/1.0 (public economic data dashboard)"}
TIMEOUT = 25

ALL_SECTORS = [
    "Agriculture (NAICS 11)", "Mining (NAICS 21)", "Utilities (NAICS 22)", "Construction (NAICS 23)",
    "Manufacturing (NAICS 31-33)", "Wholesale Trade (NAICS 42)", "Retail Trade (NAICS 44-45)",
    "Transportation & Warehousing (NAICS 48-49)", "Information/Media (NAICS 51)", "Finance & Insurance (NAICS 52)",
    "Real Estate (NAICS 53)", "Professional & Tech Services (NAICS 54)", "Management of Companies (NAICS 55)",
    "Administrative Support (NAICS 56)", "Educational Services (NAICS 61)", "Healthcare (NAICS 62)",
    "Arts & Entertainment (NAICS 71)", "Accommodation & Food Services (NAICS 72)", "Other Services (NAICS 81)",
    "Public Administration (NAICS 92)",
]
SECTOR_CODES = {s: re.search(r"NAICS ([\d-]+)", s).group(1) for s in ALL_SECTORS}

# Market universe: ZIP (ZCTA) + county FIPS + map centroid. Add rows to expand coverage.
MARKETS = [
    ("33131", "Miami (Brickell)", "Miami-Dade", "12086", 25.7686, -80.1923),
    ("33139", "Miami Beach", "Miami-Dade", "12086", 25.7793, -80.1300),
    ("33301", "Fort Lauderdale", "Broward", "12011", 26.1242, -80.1436),
    ("33480", "Palm Beach", "Palm Beach", "12099", 26.7056, -80.0364),
    ("33602", "Tampa (Downtown)", "Hillsborough", "12057", 27.9478, -82.4584),
    ("33701", "St. Petersburg", "Pinellas", "12103", 27.7731, -82.6400),
    ("32801", "Orlando (Downtown)", "Orange", "12095", 28.5419, -81.3792),
    ("32202", "Jacksonville", "Duval", "12031", 30.3322, -81.6557),
    ("32301", "Tallahassee", "Leon", "12073", 30.4383, -84.2807),
    ("32601", "Gainesville", "Alachua", "12001", 29.6516, -82.3248),
    ("33801", "Lakeland", "Polk", "12105", 28.0395, -81.9498),
    ("34102", "Naples", "Collier", "12021", 26.1420, -81.7948),
    ("34236", "Sarasota", "Sarasota", "12115", 27.3364, -82.5307),
    ("32502", "Pensacola", "Escambia", "12033", 30.4213, -87.2169),
    ("33901", "Fort Myers", "Lee", "12071", 26.6406, -81.8723),
    ("32114", "Daytona Beach", "Volusia", "12127", 29.2108, -81.0228),
    ("32901", "Melbourne", "Brevard", "12009", 28.0836, -80.6081),
]
MARKET_COLS = ["zip_code", "city", "county", "fips", "latitude", "longitude"]

# ILLUSTRATIVE fallback values (NOT real statistics) used only when a live source is unavailable.
_SAMPLE = {
    "zip_code": ["33131", "33139", "33301", "33480", "33602", "33701", "32801", "32202", "32301", "32601", "33801"],
    "median_income": [98000, 85000, 78000, 135000, 72000, 66000, 64000, 55000, 48000, 47000, 52000],
    "gdp_growth_pct": [5.1, 3.8, 4.2, 5.8, 4.8, 4.1, 5.5, 3.4, 1.5, 2.1, 3.1],
    "edu_associate_pct": [8.2, 7.1, 9.4, 5.1, 10.2, 9.8, 11.0, 12.1, 8.5, 7.2, 13.4],
    "edu_bachelor_pct": [42.5, 38.4, 39.1, 48.0, 36.4, 35.2, 34.1, 22.4, 28.1, 31.4, 18.2],
    "edu_graduate_pct": [28.4, 22.1, 20.4, 32.5, 18.1, 17.5, 15.2, 9.8, 24.3, 33.2, 6.1],
    "race_white_pct": [62.4, 74.2, 68.5, 88.1, 65.2, 71.4, 58.1, 51.2, 53.4, 57.2, 61.5],
    "race_black_pct": [14.1, 4.5, 18.2, 3.2, 16.4, 15.1, 21.4, 34.8, 33.1, 16.4, 22.8],
    "race_asian_pct": [3.5, 2.1, 2.8, 1.5, 4.2, 3.9, 5.1, 4.8, 3.2, 8.5, 1.9],
    "race_hispanic_latino_pct": [68.2, 55.4, 18.5, 6.4, 22.1, 11.5, 28.4, 12.1, 8.4, 11.2, 19.4],
    "lq_data": [1.3, 1.1, 1.2, 0.8, 1.8, 1.5, 1.9, 1.1, 0.9, 1.4, 0.5],
}
ACS_COLS = ["median_income", "edu_associate_pct", "edu_bachelor_pct", "edu_graduate_pct",
            "race_white_pct", "race_black_pct", "race_asian_pct", "race_hispanic_latino_pct"]


def secret(name: str) -> str | None:
    """Read a setting from Streamlit secrets, falling back to an environment variable."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.environ.get(name)


def _headers() -> dict:
    """User-Agent for public APIs. BLS asks automated clients to include a contact: set BLS_CONTACT_EMAIL."""
    contact = secret("BLS_CONTACT_EMAIL")
    ua = "SynopticBI/1.0 (public economic data dashboard" + (f"; contact: {contact}" if contact else "") + ")"
    return {"User-Agent": ua}


NEEDS_CENSUS_KEY = ("Census rejects requests that have no key. Get a free key (instant, by email) at "
                    "api.census.gov/data/key_signup.html and set CENSUS_API_KEY.")


def offline() -> bool:
    return (os.environ.get("SYNOPTIC_OFFLINE") or os.environ.get("EJBI_OFFLINE") or "").lower() in ("1", "true", "yes")


def _scrub(msg) -> str:
    """Hide API keys that HTTP libraries echo back inside error messages (key=..., UserID=...)."""
    return re.sub(r"(key|UserID)=[^&\s'\")]+", r"\1=***", str(msg), flags=re.I)


# ---------------------------------------------------------------- Census ACS
ACS_VARS = [
    "B19013_001E",                                                    # median household income
    "B01003_001E",                                                    # total population
    "B15003_001E", "B15003_021E", "B15003_022E", "B15003_023E",       # edu: total, assoc, bach, masters
    "B15003_024E", "B15003_025E",                                     # edu: professional, doctorate
    "B03002_001E", "B03002_003E", "B03002_004E", "B03002_006E", "B03002_012E",  # race/ethnicity
]


def _pct(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den * 100).where(den > 0)


def parse_acs(rows: list[list[str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows[1:], columns=rows[0])
    for v in ACS_VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce")
        df[v] = df[v].where(df[v] > -1e8)  # Census uses -666666666 etc. for "not available"
    edu_total = df["B15003_001E"]
    race_total = df["B03002_001E"]
    return pd.DataFrame({
        "zip_code": df["zip code tabulation area"].astype(str),
        "median_income": df["B19013_001E"],
        "population": df["B01003_001E"],
        "edu_associate_pct": _pct(df["B15003_021E"], edu_total),
        "edu_bachelor_pct": _pct(df["B15003_022E"], edu_total),
        "edu_graduate_pct": _pct(df["B15003_023E"] + df["B15003_024E"] + df["B15003_025E"], edu_total),
        "race_white_pct": _pct(df["B03002_003E"], race_total),
        "race_black_pct": _pct(df["B03002_004E"], race_total),
        "race_asian_pct": _pct(df["B03002_006E"], race_total),
        "race_hispanic_latino_pct": _pct(df["B03002_012E"], race_total),
    })


def fetch_acs(zips: list[str], api_key: str | None) -> tuple[pd.DataFrame, int]:
    if not api_key:
        raise RuntimeError(NEEDS_CENSUS_KEY)
    last_err: Exception | None = None
    for year in (2024, 2023, 2022):  # newest release that responds wins
        query = {"get": "NAME," + ",".join(ACS_VARS),
                 "for": "zip code tabulation area:" + ",".join(zips)}
        if api_key:
            query["key"] = api_key
        url = f"https://api.census.gov/data/{year}/acs/acs5?" + urlencode(query, quote_via=quote, safe=",:")
        try:
            r = requests.get(url, headers=_headers(), timeout=TIMEOUT)
            r.raise_for_status()
            return parse_acs(r.json()), year
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"Census ACS request failed ({last_err})")


# ---------------------------------------------------------------- BLS QCEW
def parse_qcew(text: str) -> pd.Series:
    """Return Series: NAICS sector code -> employment location quotient for one county."""
    df = pd.read_csv(io.StringIO(text), dtype={"industry_code": str, "own_code": str, "agglvl_code": str})
    df = df[df["industry_code"].isin(set(SECTOR_CODES.values()))].copy()
    df["pri"] = (df["own_code"] != "5").astype(int)  # prefer private ownership; public admin has none
    df = (df.sort_values(["industry_code", "pri", "annual_avg_emplvl"], ascending=[True, True, False])
            .drop_duplicates("industry_code"))
    lq = pd.to_numeric(df["lq_annual_avg_emplvl"], errors="coerce")
    if "disclosure_code" in df:
        lq = lq.where(df["disclosure_code"].fillna("") != "N")  # suppressed -> unknown
    return pd.Series(lq.values, index=df["industry_code"].values)


def _fetch_qcew_county(fips: str) -> tuple[str, pd.Series, int]:
    for year in (2025, 2024, 2023):
        url = f"https://data.bls.gov/cew/data/api/{year}/a/area/{fips}.csv"
        r = requests.get(url, headers=_headers(), timeout=TIMEOUT)
        if r.status_code == 200 and r.text.strip():
            return fips, parse_qcew(r.text), year
    raise RuntimeError(f"no QCEW annual file for county {fips}")


def fetch_qcew(fips_list: list[str]) -> tuple[dict[str, pd.Series], int]:
    results, years = {}, []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_fetch_qcew_county, f) for f in fips_list]
        for fut in futures:
            try:
                fips, series, year = fut.result()
                results[fips] = series
                years.append(year)
            except Exception:  # noqa: BLE001
                continue
    if not results:
        raise RuntimeError("BLS QCEW returned no county data")
    return results, max(years)


# ---------------------------------------------------------------- BEA
def fetch_bea_gdp_growth(api_key: str) -> tuple[pd.Series, str]:
    params = {"UserID": api_key, "method": "GetData", "datasetname": "Regional", "TableName": "CAGDP9",
              "LineCode": "1", "GeoFips": "COUNTY", "Year": "LAST5", "ResultFormat": "JSON"}
    r = requests.get("https://apps.bea.gov/api/data", params=params, headers=_headers(), timeout=TIMEOUT)
    r.raise_for_status()
    try:
        data = r.json()["BEAAPI"]["Results"]["Data"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"BEA returned an error response ({exc})") from exc
    df = pd.DataFrame(data)
    df = df[df["GeoFips"].str.startswith("12") & (df["GeoFips"] != "12000")].copy()
    df["v"] = pd.to_numeric(df["DataValue"].astype(str).str.replace(",", ""), errors="coerce")
    wide = df.pivot_table(index="GeoFips", columns="TimePeriod", values="v")
    periods = [c for c in sorted(wide.columns) if wide[c].notna().sum() > 5]
    if len(periods) < 2:
        raise RuntimeError("BEA returned fewer than two usable years")
    old, new = periods[-2], periods[-1]
    return (wide[new] / wide[old] - 1) * 100, f"{old}-{new}"


# ---------------------------------------------------------------- assembly
def _synthetic_sector_lq(zips: list[str], base_lq: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for sector in ALL_SECTORS:
        noise = rng.normal(1.0, 0.35, len(zips)).clip(0.2, None)
        for z, n in zip(zips, noise):
            rows.append({"zip_code": z, "sector": sector, "lq": round(float(base_lq.get(z, 1.0)) * n, 2)})
    return pd.DataFrame(rows)


@st.cache_data(ttl=6 * 3600, show_spinner="Pulling live economic data…")
def load_all():
    """Return (markets, sector_lq, status, loaded_at). Cached for 6 hours."""
    markets = pd.DataFrame(MARKETS, columns=MARKET_COLS)
    sample = pd.DataFrame(_SAMPLE)
    status = []

    # --- Census ACS
    acs, acs_year, acs_err = None, None, None
    if offline():
        acs_err = "offline mode (SYNOPTIC_OFFLINE is set)"
    else:
        try:
            acs, acs_year = fetch_acs(list(markets["zip_code"]), secret("CENSUS_API_KEY"))
        except Exception as exc:  # noqa: BLE001
            acs_err = _scrub(exc)
    if acs is not None:
        df = markets.merge(acs, on="zip_code", how="left")
        status.append({"Source": "Census ACS 5-year", "Feeds": "Income, education, race/ethnicity",
                       "State": "Live", "Detail": f"{acs_year - 4}-{acs_year} estimates, {len(acs)} ZIPs returned"})
    else:
        df = markets[markets["zip_code"].isin(sample["zip_code"])].merge(
            sample[["zip_code"] + ACS_COLS], on="zip_code", how="left")
        status.append({"Source": "Census ACS 5-year", "Feeds": "Income, education, race/ethnicity",
                       "State": "Sample", "Detail": f"Using illustrative values. {acs_err}"})

    # --- BEA GDP
    bea_key = secret("BEA_API_KEY")
    gdp, gdp_note = None, None
    if offline():
        gdp_note = "offline mode"
    elif not bea_key:
        gdp_note = "no BEA_API_KEY set (free key at apps.bea.gov/API/signup)"
    else:
        try:
            gdp, span = fetch_bea_gdp_growth(bea_key)
        except Exception as exc:  # noqa: BLE001
            gdp_note = _scrub(exc)
    if gdp is not None:
        df["gdp_growth_pct"] = df["fips"].map(gdp)
        status.append({"Source": "BEA Regional (CAGDP9)", "Feeds": "County real GDP growth",
                       "State": "Live", "Detail": f"{span} change, county level"})
    else:
        df = df.merge(sample[["zip_code", "gdp_growth_pct"]], on="zip_code", how="left")
        status.append({"Source": "BEA Regional (CAGDP9)", "Feeds": "County real GDP growth",
                       "State": "Sample", "Detail": f"Using illustrative values. {gdp_note}"})

    # --- BLS QCEW
    qcew, qcew_year, qcew_err = None, None, None
    if offline():
        qcew_err = "offline mode"
    else:
        try:
            qcew, qcew_year = fetch_qcew(sorted(df["fips"].unique()))
        except Exception as exc:  # noqa: BLE001
            qcew_err = _scrub(exc)
    if qcew is not None:
        rows = []
        for z, fips in zip(df["zip_code"], df["fips"]):
            series = qcew.get(fips)
            for sector, code in SECTOR_CODES.items():
                val = float(series.get(code, np.nan)) if series is not None else np.nan
                rows.append({"zip_code": z, "sector": sector, "lq": val})
        sector_lq = pd.DataFrame(rows)
        status.append({"Source": "BLS QCEW (annual)", "Feeds": "Sector location quotients",
                       "State": "Live", "Detail": f"{qcew_year} data, county level ({len(qcew)} counties)"})
    else:
        base = sample.set_index("zip_code")["lq_data"]
        sector_lq = _synthetic_sector_lq(list(df["zip_code"]), base)
        status.append({"Source": "BLS QCEW (annual)", "Feeds": "Sector location quotients",
                       "State": "Sample", "Detail": f"Using synthetic values. {qcew_err}"})

    # Drop markets that came back without core live metrics.
    df = df.dropna(subset=["median_income", "edu_bachelor_pct"]).reset_index(drop=True)
    loaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return df, sector_lq, pd.DataFrame(status), loaded_at


# ---------------------------------------------------------------- Census CBP (small business by ZIP)
CBP_YEARS = (2024, 2023, 2022)  # newest release that responds wins


def _size_bucket(label: str) -> str | None:
    """Classify a CBP employment-size label without relying on numeric size codes."""
    low = str(label).lower().strip()
    if low.startswith("all"):
        return "total"
    nums = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", low)]
    if not nums:
        return None
    if "less than" in low or "fewer than" in low or nums[0] < 5:
        return "lt5"
    if nums[0] in (5, 10):
        return "5_19"
    return "20plus"


def parse_cbp(rows: list[list[str]]) -> pd.DataFrame:
    """Census CBP rows (all sectors, all legal forms, by size class) -> one row per ZIP."""
    df = pd.DataFrame(rows[1:], columns=rows[0])
    zip_col = "zip code" if "zip code" in df.columns else "ZIPCODE"
    df["zip_code"] = df[zip_col].astype(str).str.zfill(5)
    for c in ("ESTAB", "EMP", "PAYANN"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["bucket"] = df["EMPSZES_LABEL"].map(_size_bucket)
    est = df.pivot_table(index="zip_code", columns="bucket", values="ESTAB", aggfunc="sum").reindex(
        columns=["total", "lt5", "5_19", "20plus"])
    tot = df[df["bucket"] == "total"].set_index("zip_code")
    out = pd.DataFrame({
        "estab_total": est["total"], "estab_lt5": est["lt5"], "estab_5_19": est["5_19"],
        "estab_20plus": est["20plus"],
        "employees": tot["EMP"], "payroll_k": tot["PAYANN"],
    }).reset_index()
    return out


def fetch_cbp(zips: list[str], api_key: str | None) -> tuple[pd.DataFrame, int]:
    if not api_key:
        raise RuntimeError(NEEDS_CENSUS_KEY)
    last_err: Exception | None = None
    for year in CBP_YEARS:
        query = {"get": "ESTAB,EMP,PAYANN,EMPSZES,EMPSZES_LABEL",
                 "for": "zip code:" + ",".join(zips), "NAICS2017": "00", "LFO": "001", "key": api_key}
        url = f"https://api.census.gov/data/{year}/cbp?" + urlencode(query, quote_via=quote, safe=",:")
        try:
            r = requests.get(url, headers=_headers(), timeout=TIMEOUT)
            r.raise_for_status()
            return parse_cbp(r.json()), year
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"Census CBP request failed ({last_err})")


# ---------------------------------------------------------------- FDIC Summary of Deposits (branches)
FDIC_SOD_URL = "https://api.fdic.gov/banks/sod"


def _first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((n for n in names if n in df.columns), None)


def parse_fdic(rows: list[dict]) -> pd.DataFrame:
    """Normalize raw FDIC SOD records into one tidy branch table."""
    df = pd.DataFrame(rows)
    zip_c = _first_col(df, ("ZIPBR", "ZIP"))
    dep_c = _first_col(df, ("DEPSUMBR",))
    bank_c = _first_col(df, ("NAMEFULL", "NAME", "NAMEBR"))
    if not (zip_c and dep_c and bank_c):
        raise ValueError("FDIC response is missing expected fields (need a ZIP, DEPSUMBR and a bank name). "
                         f"Columns returned: {sorted(df.columns)[:25]}")
    lat_c = _first_col(df, ("SIMS_LATITUDE", "LATITUDE"))
    lon_c = _first_col(df, ("SIMS_LONGITUDE", "LONGITUDE"))
    addr_c = _first_col(df, ("ADDRESBR", "ADDRESS"))
    city_c = _first_col(df, ("CITYBR", "CITY"))
    cert_c = _first_col(df, ("CERT",))
    out = pd.DataFrame({
        "bank": df[bank_c].astype(str).str.strip(),
        "cert": df[cert_c] if cert_c else None,
        "address": df[addr_c] if addr_c else "",
        "city": df[city_c] if city_c else "",
        "zip5": df[zip_c].astype(str).str.extract(r"(\d{5})")[0],
        "lat": pd.to_numeric(df[lat_c], errors="coerce") if lat_c else np.nan,
        "lon": pd.to_numeric(df[lon_c], errors="coerce") if lon_c else np.nan,
        "deposits_usd": pd.to_numeric(df[dep_c], errors="coerce").fillna(0) * 1000,  # DEPSUMBR is in $ thousands
    })
    dedupe = [c for c in ("ID",) if c in df.columns]
    if dedupe:
        out = out.loc[~df.duplicated(subset=dedupe)]
    return out.dropna(subset=["zip5"]).reset_index(drop=True)


def fetch_fdic_sod(state: str = "FL", api_key: str | None = None) -> tuple[pd.DataFrame, int]:
    """All branches in a state for the newest Summary of Deposits year (June 30 survey)."""
    this_year = datetime.now(timezone.utc).year
    last_err: Exception | None = None
    for year in (this_year, this_year - 1, this_year - 2):
        try:
            rows, offset = [], 0
            while True:
                params = {"filters": f"YEAR:{year} AND STALPBR:{state}", "limit": 10000, "offset": offset}
                if api_key:
                    params["api_key"] = api_key
                r = requests.get(FDIC_SOD_URL, params=params, headers=_headers(), timeout=60)
                r.raise_for_status()
                js = r.json()
                batch = [(it.get("data", it) if isinstance(it, dict) else {}) for it in js.get("data", [])]
                rows += batch
                offset += len(batch)
                total = (js.get("meta") or {}).get("total", len(rows))
                if not batch or offset >= total:
                    break
            if rows:
                return parse_fdic(rows), year
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"FDIC Summary of Deposits request failed ({last_err})")


@st.cache_data(ttl=24 * 3600, show_spinner="Pulling small-business and bank-branch data…")
def load_banking(zips: tuple[str, ...]):
    """Returns dict(cbp, branches, population, status, loaded_at). Any failed source is None."""
    out = {"cbp": None, "branches": None, "population": None}
    status = []
    census_key = secret("CENSUS_API_KEY")

    def attempt(name, feeds, fn, store, detail_fn):
        if offline():
            status.append({"Source": name, "Feeds": feeds, "State": "Unavailable", "Detail": "Offline mode"})
            return
        try:
            value, year = fn()
            out[store] = value
            status.append({"Source": name, "Feeds": feeds, "State": "Live", "Detail": detail_fn(value, year)})
        except Exception as exc:  # noqa: BLE001
            status.append({"Source": name, "Feeds": feeds, "State": "Unavailable", "Detail": _scrub(exc)})

    attempt("Census CBP", "Small businesses by ZIP and size", lambda: fetch_cbp(list(zips), census_key), "cbp",
            lambda v, y: f"{y} data, {len(v)} ZIPs with employer businesses")
    attempt("FDIC Summary of Deposits", "Bank branches and deposits",
            lambda: fetch_fdic_sod("FL", secret("FDIC_API_KEY")), "branches",
            lambda v, y: f"June 30, {y} survey, {len(v):,} Florida branches")
    attempt("Census ACS (population)", "Residents per ZIP", lambda: fetch_acs(list(zips), census_key), "population",
            lambda v, y: f"{y - 4}-{y} estimates")
    return {**out, "status": pd.DataFrame(status),
            "loaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}


# ---------------------------------------------------------------- connection check
def _http_detail(r) -> str:
    body = re.sub(r"<[^>]+>", " ", r.text or "")
    return _scrub(f"HTTP {r.status_code}: {' '.join(body.split())[:140]}")


def run_diagnostics() -> pd.DataFrame:
    """One tiny request per source, from THIS server. Never returns key values."""
    rows = []

    def probe(source, key_name, fn, fix):
        if key_name and not secret(key_name):
            rows.append({"Source": source, "Key": f"{key_name}: missing", "Result": "❌ Not tried",
                         "Detail": "No key is set.", "Fix": fix})
            return
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, _scrub(exc)[:250]
        rows.append({"Source": source, "Key": f"{key_name}: set" if key_name else "No key needed",
                     "Result": "✅ Working" if ok else "❌ Failed", "Detail": detail, "Fix": "" if ok else fix})

    def census(path_year_template, extra):
        def run():
            for year in (2024, 2023, 2022):
                url = f"https://api.census.gov/data/{year}/{path_year_template}" + extra + f"&key={secret('CENSUS_API_KEY')}"
                r = requests.get(url, headers=_headers(), timeout=20)
                if r.status_code == 200:
                    j = r.json()
                    if isinstance(j, list) and len(j) >= 2:
                        return True, f"{year} data returned"
            return False, _http_detail(r)
        return run

    census_fix = "Check the key was activated (click the link in Census's email) and pasted without spaces or quotes inside quotes."
    probe("Census ACS (income, education, population)", "CENSUS_API_KEY",
          census("acs/acs5", "?get=NAME,B19013_001E&for=zip%20code%20tabulation%20area:33131"), census_fix)
    probe("Census CBP (small businesses)", "CENSUS_API_KEY",
          census("cbp", "?get=ESTAB,EMPSZES_LABEL&for=zip%20code:33131&NAICS2017=00&LFO=001"), census_fix)

    def bea():
        r = requests.get("https://apps.bea.gov/api/data", headers=_headers(), timeout=20, params={
            "UserID": secret("BEA_API_KEY"), "method": "GetData", "datasetname": "Regional", "TableName": "CAGDP9",
            "LineCode": "1", "GeoFips": "12086", "Year": "LAST5", "ResultFormat": "JSON"})
        try:
            res = r.json()["BEAAPI"]["Results"]
            if "Data" in res:
                return True, f"{len(res['Data'])} yearly values returned"
            err = res.get("Error") or r.json()["BEAAPI"].get("Error") or {}
            return False, _scrub(err.get("APIErrorDescription") or str(err))[:250]
        except Exception:  # noqa: BLE001
            return False, _http_detail(r)

    probe("BEA GDP growth", "BEA_API_KEY", bea,
          "Create a free key at apps.bea.gov/API/signup, click the activation link in the email, then paste it.")

    def bls():
        r = requests.get("https://data.bls.gov/cew/data/api/2024/a/area/12086.csv", headers=_headers(), timeout=30)
        if r.status_code == 200 and "area_fips" in r.text[:300]:
            return True, "Miami-Dade 2024 file returned"
        return False, _http_detail(r)

    probe("BLS QCEW (sector concentration)", None, bls,
          "BLS may block automated requests that do not identify a contact. Set BLS_CONTACT_EMAIL to an email "
          "you monitor, then retest.")

    def fdic():
        r = requests.get(FDIC_SOD_URL, headers=_headers(), timeout=30, params={
            "filters": "YEAR:2025 AND STALPBR:FL AND ZIPBR:33131", "limit": 1,
            **({"api_key": secret("FDIC_API_KEY")} if secret("FDIC_API_KEY") else {})})
        if r.status_code == 200 and (r.json().get("data") or []):
            return True, "Branch data returned"
        return False, _http_detail(r)

    probe("FDIC bank branches", None, fdic, "If it reports an authorization error, add a free api.data.gov key as FDIC_API_KEY.")

    rows.append({"Source": "Anthropic (AI Analyst)", "Key": "ANTHROPIC_API_KEY: " + ("set" if secret("ANTHROPIC_API_KEY") else "missing"),
                 "Result": "✅ Key present" if secret("ANTHROPIC_API_KEY") else "⚪ Optional",
                 "Detail": "Not tested here to avoid usage charges.", "Fix": ""})
    return pd.DataFrame(rows)
