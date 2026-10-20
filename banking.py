"""Small-business and bank-branch analytics for ZIP codes. Pure functions: no Streamlit, easy to test.

Inputs
  universe : DataFrame[zip_code, city, latitude, longitude, (population)]
  cbp      : DataFrame[zip_code, estab_total, estab_lt5, estab_5_19, estab_20plus, employees, payroll_k]
  branches : DataFrame[bank, cert, address, city, zip5, lat, lon, deposits_usd]   (FDIC Summary of Deposits)

Notes on meaning
  * CBP counts establishments (locations with employees), not firms, and excludes sole proprietors.
  * Branch deposits are what each branch books; online/HQ branches can distort a single ZIP.
  * The saturation label compares each ZIP to the median of the ZIPs being analyzed, so it is relative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_MILES = 3958.8
DEFAULT_WEIGHTS = {"volume": 40, "underserved": 40, "density": 20}
SMALL_DEFS = ["Under 20 employees", "Under 5 employees"]
STAT_COLS = ["zip_code", "branches_in_zip", "banks_in_zip", "deposits_in_zip", "top_bank",
             "top_bank_share_pct", "deposit_hhi"]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_MILES * np.arcsin(np.sqrt(a))


def _norm(s: pd.Series) -> pd.Series:
    filled = s.fillna(s.median()).fillna(0)
    rng = filled.max() - filled.min()
    return (filled - filled.min()) / rng if rng else pd.Series(0.5, index=s.index)


def zip_branch_stats(branches: pd.DataFrame) -> pd.DataFrame:
    """Per-ZIP branch count, bank count, deposits, largest bank share and deposit HHI (0-10,000)."""
    rows = []
    for z, g in branches.groupby("zip5"):
        dep = g.groupby("bank")["deposits_usd"].sum()
        total = float(dep.sum())
        shares = dep / total * 100 if total > 0 else dep * 0
        rows.append({
            "zip_code": z, "branches_in_zip": len(g), "banks_in_zip": int(g["bank"].nunique()),
            "deposits_in_zip": total,
            "top_bank": dep.idxmax() if total > 0 else None,
            "top_bank_share_pct": float(shares.max()) if total > 0 else np.nan,
            "deposit_hhi": float((shares ** 2).sum()) if total > 0 else np.nan,
        })
    return pd.DataFrame(rows, columns=STAT_COLS)


def nearby_branches(branches: pd.DataFrame, lat: float, lon: float, radius_mi: float) -> pd.DataFrame:
    """Branches within radius of a point, with distance, largest deposits first."""
    valid = branches.dropna(subset=["lat", "lon"])
    if valid.empty or pd.isna(lat) or pd.isna(lon):
        return valid.iloc[0:0].assign(distance_mi=pd.Series(dtype=float))
    dist = haversine_miles(lat, lon, valid["lat"].values, valid["lon"].values)
    out = valid.assign(distance_mi=dist)
    return out[out["distance_mi"] <= radius_mi].sort_values("deposits_usd", ascending=False)


def point_in_ring(lons, lats, ring) -> np.ndarray:
    """Vectorized ray-casting test. ring = [(lon, lat), ...] (outer ring of a polygon)."""
    x, y = np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    pts = list(ring)
    n = len(pts)
    if n < 3:
        return inside
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if yi != yj:
            cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            inside ^= cross
        j = i
    return inside


def catchment_branches(branches: pd.DataFrame, lat: float, lon: float, radius_mi: float,
                       ring=None) -> pd.DataFrame:
    """Branches inside a drive-time polygon (ring) when given, else within a straight-line radius.
    distance_mi is always the straight-line distance from the ZIP point, for reference."""
    if ring is None or len(ring) < 3:
        return nearby_branches(branches, lat, lon, radius_mi)
    valid = branches.dropna(subset=["lat", "lon"])
    if valid.empty:
        return valid.iloc[0:0].assign(distance_mi=pd.Series(dtype=float))
    mask = point_in_ring(valid["lon"].values, valid["lat"].values, ring)
    out = valid[mask]
    if pd.notna(lat) and pd.notna(lon):
        out = out.assign(distance_mi=haversine_miles(lat, lon, out["lat"].values, out["lon"].values))
    else:
        out = out.assign(distance_mi=np.nan)
    return out.sort_values("deposits_usd", ascending=False)


def hhi_of(branch_df: pd.DataFrame) -> float:
    """Deposit HHI (0-10,000) across banks for a set of branches; NaN if there are no deposits."""
    if branch_df is None or branch_df.empty:
        return float("nan")
    dep = branch_df.groupby("bank")["deposits_usd"].sum()
    total = float(dep.sum())
    return float(((dep / total * 100) ** 2).sum()) if total > 0 else float("nan")


HHI_LABELS = ["Highly concentrated", "Moderately concentrated", "Fragmented", "n/a"]


def hhi_label(h: float) -> str:
    """Bands from the 2023 U.S. merger guidelines (>1800 highly, 1000-1800 moderately concentrated).
    Used here only as a plain-English reading of deposit concentration, not as an antitrust finding."""
    if h is None or pd.isna(h):
        return "n/a"
    if h > 1800:
        return "Highly concentrated"
    if h >= 1000:
        return "Moderately concentrated"
    return "Fragmented"


CAPTURE_LABELS = ["Under-captured", "Typical", "Deposit-rich", "n/a"]


def capture_label(idx: float) -> str:
    if idx is None or pd.isna(idx):
        return "n/a"
    if idx < 60:
        return "Under-captured"
    if idx > 140:
        return "Deposit-rich"
    return "Typical"


OUTLIER_LABELS = ["Underperforming", "In line", "Overperforming"]


def branch_outliers(branch_df: pd.DataFrame, low: float = 0.5, high: float = 2.0) -> pd.DataFrame:
    """Compare each branch's deposits with the median branch in the same set (the catchment).
    ratio < low  -> Underperforming (possible weak branch / acquisition or share-gain target)
    ratio > high -> Overperforming (often a headquarters or online-booked branch, so verify)."""
    if branch_df is None or branch_df.empty:
        return pd.DataFrame(columns=list(getattr(branch_df, "columns", [])) + ["peer_median_usd", "vs_peer", "flag"])
    d = branch_df.copy()
    med = float(d.loc[d["deposits_usd"] > 0, "deposits_usd"].median()) if (d["deposits_usd"] > 0).any() else float("nan")
    d["peer_median_usd"] = med
    d["vs_peer"] = d["deposits_usd"] / med if med and not pd.isna(med) else np.nan
    d["flag"] = np.where(d["vs_peer"].isna(), "In line",
                         np.where(d["vs_peer"] < low, "Underperforming",
                                  np.where(d["vs_peer"] > high, "Overperforming", "In line")))
    return d


def bank_share(branch_df: pd.DataFrame) -> pd.DataFrame:
    """Deposit market share by bank for any set of branches."""
    if branch_df.empty:
        return pd.DataFrame(columns=["bank", "branches", "deposits_usd", "share_pct"])
    g = branch_df.groupby("bank").agg(branches=("bank", "size"), deposits_usd=("deposits_usd", "sum")).reset_index()
    total = g["deposits_usd"].sum()
    g["share_pct"] = g["deposits_usd"] / total * 100 if total > 0 else 0.0
    return g.sort_values("deposits_usd", ascending=False).reset_index(drop=True)


def build_banking_table(universe: pd.DataFrame, cbp: pd.DataFrame | None, branches: pd.DataFrame,
                        radius_mi: float = 3.0, small_def: str = SMALL_DEFS[0],
                        weights: dict | None = None, catchments: dict | None = None) -> pd.DataFrame:
    """catchments: optional {zip_code: ring} of drive-time polygons; ZIPs without one use the radius."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    catchments = catchments or {}
    t = universe.copy()
    t["zip_code"] = t["zip_code"].astype(str)
    for col in ("latitude", "longitude", "population"):
        if col not in t:
            t[col] = np.nan

    if cbp is not None:
        t = t.merge(cbp, on="zip_code", how="left")
    for col in ("estab_total", "estab_lt5", "estab_5_19", "estab_20plus", "employees", "payroll_k"):
        if col not in t:
            t[col] = 0.0
        t[col] = t[col].fillna(0)  # a ZIP missing from CBP has no employer establishments
    t["small_biz"] = t["estab_lt5"] + (t["estab_5_19"] if small_def == SMALL_DEFS[0] else 0)

    t = t.merge(zip_branch_stats(branches), on="zip_code", how="left")
    for col in ("branches_in_zip", "banks_in_zip", "deposits_in_zip"):
        t[col] = t[col].fillna(0)

    # Centroid: the market's own point, else the average location of branches in that ZIP.
    centroid = branches.dropna(subset=["lat", "lon"]).groupby("zip5")[["lat", "lon"]].mean()
    t["latitude"] = t["latitude"].fillna(t["zip_code"].map(centroid["lat"]))
    t["longitude"] = t["longitude"].fillna(t["zip_code"].map(centroid["lon"]))

    near_n, near_dep, near_banks, near_hhi, near_under = [], [], [], [], []
    has_coords = not branches.dropna(subset=["lat", "lon"]).empty
    for z, lat, lon in zip(t["zip_code"], t["latitude"], t["longitude"]):
        if pd.isna(lat) or pd.isna(lon) or not has_coords:
            near_n.append(np.nan); near_dep.append(np.nan); near_banks.append(np.nan)
            near_hhi.append(np.nan); near_under.append(np.nan)
            continue
        nb = catchment_branches(branches, lat, lon, radius_mi, catchments.get(z))
        near_n.append(len(nb)); near_dep.append(float(nb["deposits_usd"].sum())); near_banks.append(nb["bank"].nunique())
        near_hhi.append(hhi_of(nb))
        near_under.append(int((branch_outliers(nb)["flag"] == "Underperforming").sum()) if len(nb) else 0)
    t["branches_nearby"], t["deposits_nearby"], t["banks_nearby"] = near_n, near_dep, near_banks
    t["hhi_nearby"], t["underperforming_branches"] = near_hhi, near_under
    # Prefer the catchment HHI; fall back to the in-ZIP HHI when there are no coordinates.
    t["hhi_used"] = t["hhi_nearby"].fillna(t["deposit_hhi"])
    t["hhi_label"] = t["hhi_used"].map(hhi_label)

    t["small_biz_per_branch"] = t["small_biz"] / t["branches_in_zip"].clip(lower=1)  # no-branch ZIP counts as 1
    t["small_biz_per_10k_pop"] = (t["small_biz"] / t["population"] * 10000).where(t["population"] > 0)
    t["deposits_per_small_biz"] = (t["deposits_in_zip"] / t["small_biz"]).where(t["small_biz"] > 0)
    # Deposit capture index: deposits booked per small business, 100 = median ZIP that has branches.
    med_dep = t.loc[t["branches_in_zip"] > 0, "deposits_per_small_biz"].median()
    t["capture_index"] = (t["deposits_per_small_biz"] / med_dep * 100) if pd.notna(med_dep) and med_dep > 0 else np.nan
    t.loc[(t["branches_in_zip"] == 0) & (t["small_biz"] > 0), "capture_index"] = 0.0
    t["capture_label"] = t["capture_index"].map(capture_label)

    parts = {"volume": _norm(t["small_biz"]), "underserved": _norm(t["small_biz_per_branch"]),
             "density": _norm(t["small_biz_per_10k_pop"])}
    total_w = sum(w[k] for k in parts) or 1
    t["opportunity"] = sum(parts[k] * w[k] for k in parts) / total_w * 100

    med = t.loc[t["branches_in_zip"] > 0, "small_biz_per_branch"].median()

    def coverage(row) -> str:
        if row["branches_in_zip"] == 0:
            return "No branch in ZIP"
        if pd.isna(med) or med == 0:
            return "n/a"
        if row["small_biz_per_branch"] >= 1.25 * med:
            return "Underserved"
        if row["small_biz_per_branch"] <= 0.75 * med:
            return "Saturated"
        return "Balanced"

    t["coverage"] = t.apply(coverage, axis=1)
    t = t.sort_values("opportunity", ascending=False).reset_index(drop=True)
    t.insert(0, "rank", t.index + 1)
    return t
