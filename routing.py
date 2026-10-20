"""Drive-time catchments (isochrones) from OpenRouteService.

Optional: add a free ORS_API_KEY (openrouteservice.org) to your secrets. Without it the app keeps using
straight-line radius. Free keys allow roughly 20 isochrone requests per minute and 500 per day, so results are
cached for a week and requests are spaced out.
"""
from __future__ import annotations

import time

import requests
import streamlit as st

from data_sources import _scrub, secret

ORS_URL = "https://api.openrouteservice.org/v2/isochrones/driving-car"
DRIVE_MINUTES = (5, 10, 15)
_BURST = 15          # requests we send back to back before spacing them out
_SPACING_S = 3.2     # keeps us under ~20 requests/minute afterwards


def ors_key() -> str | None:
    return secret("ORS_API_KEY") or None


def parse_isochrone(js: dict) -> list[tuple[float, float]]:
    """Outer ring [(lon, lat), ...] of the largest polygon in an ORS isochrone GeoJSON response."""
    feats = (js or {}).get("features") or []
    if not feats:
        raise ValueError("OpenRouteService returned no polygon for this point.")
    geom = feats[0].get("geometry") or {}
    kind, coords = geom.get("type"), geom.get("coordinates") or []
    polys = [coords] if kind == "Polygon" else coords if kind == "MultiPolygon" else []
    if not polys:
        raise ValueError(f"Unexpected geometry type: {kind}")
    rings = [p[0] for p in polys if p and p[0]]

    def area(r):  # shoelace; the largest ring is the real catchment, the rest are islands
        return abs(sum(r[i][0] * r[(i + 1) % len(r)][1] - r[(i + 1) % len(r)][0] * r[i][1] for i in range(len(r)))) / 2

    ring = max(rings, key=area)
    return [(float(x), float(y)) for x, y, *_ in ring]


def fetch_isochrone(lat: float, lon: float, minutes: int, key: str) -> list[tuple[float, float]]:
    body = {"locations": [[lon, lat]], "range": [int(minutes) * 60], "range_type": "time"}
    headers = {"Authorization": key, "Content-Type": "application/json; charset=utf-8",
               "Accept": "application/geo+json, application/json"}
    r = requests.post(ORS_URL, json=body, headers=headers, timeout=30)
    if r.status_code == 429:  # rate limited: wait once and retry
        time.sleep(25)
        r = requests.post(ORS_URL, json=body, headers=headers, timeout=30)
    r.raise_for_status()
    return parse_isochrone(r.json())


@st.cache_data(ttl=7 * 24 * 3600, show_spinner="Drawing drive-time areas…")
def load_catchments(points: tuple, minutes: int, key: str):
    """points = ((zip, lat, lon), ...). Returns (rings {zip: ring}, errors {zip: message})."""
    rings, errors, sent = {}, {}, 0
    for z, lat, lon in points:
        if lat is None or lon is None or lat != lat or lon != lon:
            errors[z] = "no coordinates for this ZIP"
            continue
        if sent >= _BURST:
            time.sleep(_SPACING_S)
        try:
            rings[z] = fetch_isochrone(float(lat), float(lon), minutes, key)
        except Exception as exc:  # noqa: BLE001
            errors[z] = _scrub(exc)
        sent += 1
    return rings, errors
