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
                        weights: dict | None = None) -> pd.DataFrame:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
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

    near_n, near_dep, near_banks = [], [], []
    for lat, lon in zip(t["latitude"], t["longitude"]):
        nb = nearby_branches(branches, lat, lon, radius_mi)
        if pd.isna(lat) or pd.isna(lon) or branches.dropna(subset=["lat", "lon"]).empty:
            near_n.append(np.nan); near_dep.append(np.nan); near_banks.append(np.nan)
        else:
            near_n.append(len(nb)); near_dep.append(float(nb["deposits_usd"].sum())); near_banks.append(nb["bank"].nunique())
    t["branches_nearby"], t["deposits_nearby"], t["banks_nearby"] = near_n, near_dep, near_banks

    t["small_biz_per_branch"] = t["small_biz"] / t["branches_in_zip"].clip(lower=1)  # no-branch ZIP counts as 1
    t["small_biz_per_10k_pop"] = (t["small_biz"] / t["population"] * 10000).where(t["population"] > 0)
    t["deposits_per_small_biz"] = (t["deposits_in_zip"] / t["small_biz"]).where(t["small_biz"] > 0)

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
