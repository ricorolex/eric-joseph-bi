"""Publication-ready chart images (PNG, 1200x675) for articles and slides. Uses matplotlib, no browser needed."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

NAVY, TEAL, LIGHT, GRAY = "#0b1f3a", "#0e7490", "#a5d8d4", "#64748b"
CMAP = LinearSegmentedColormap.from_list("synoptic", [LIGHT, TEAL, NAVY])
SOURCES = "Sources: U.S. Census Bureau, Bureau of Labor Statistics, Bureau of Economic Analysis, FDIC"


def _frame(title: str, subtitle: str, footer: str):
    fig = plt.figure(figsize=(12, 6.75), dpi=100, facecolor="white")
    ax = fig.add_axes([0.27, 0.16, 0.68, 0.64])
    fig.text(0.04, 0.93, title, fontsize=21, fontweight="bold", color=NAVY, va="center")
    fig.text(0.04, 0.865, subtitle, fontsize=12.5, color=GRAY, va="center")
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.1, 0.1], color="#e5eaf1", lw=1.2))
    fig.text(0.04, 0.062, footer, fontsize=9, color=GRAY, va="center")
    fig.text(0.96, 0.062, "Synoptic BI", fontsize=11, fontweight="bold", color=TEAL, va="center", ha="right")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="y", length=0, labelsize=12, labelcolor=NAVY)
    ax.tick_params(axis="x", labelsize=10, labelcolor=GRAY, colors="#cbd5e1")
    ax.xaxis.grid(True, color="#eef2f6")
    ax.set_axisbelow(True)
    return fig, ax


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _barh(ax, labels, values, vmax, fmt, color_vals=None, extra=None):
    y = np.arange(len(labels))[::-1]
    colors = [CMAP(v / 100) for v in (color_vals if color_vals is not None else values)]
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    right = vmax * (1.38 if extra else 1.0)  # leave a clear column on the right for the grey detail text
    ax.set_xlim(0, right)
    ticks = [t for t in ax.get_xticks() if 0 <= t <= vmax]
    ax.set_xticks(ticks)
    for yi, v, lab in zip(y, values, [fmt(v) for v in values]):
        ax.text(v + vmax * 0.012, yi, lab, va="center", fontsize=11.5, fontweight="bold", color=NAVY)
    if extra:
        for yi, t in zip(y, extra):
            ax.text(right, yi, t, va="center", ha="right", fontsize=9.5, color=GRAY)


def market_scores_png(view: pd.DataFrame, sector: str, loaded_at: str, top_n: int = 10) -> bytes:
    v = view.head(top_n).sort_values("BI_Score", ascending=False)
    fig, ax = _frame("Florida markets ranked by Synoptic BI score",
                     f"Sector: {sector.split(' (')[0]} · scores are relative to the markets shown (0-100)",
                     f"{SOURCES}. Data as of {loaded_at}. Not investment advice.")
    _barh(ax, list(v["city"]), list(v["BI_Score"]), 100, lambda x: f"{x:.0f}",
          extra=[f"${i / 1000:,.0f}k income" for i in v["median_income"]])
    ax.set_xlabel("")
    return _png(fig)


def banking_opportunity_png(table: pd.DataFrame, small_def: str, catch_label: str, loaded_at: str, top_n: int = 12) -> bytes:
    t = table.head(top_n)
    labels = [f"{c} ({z})" for c, z in zip(t["city"], t["zip_code"])]
    fig, ax = _frame("Where small businesses outnumber bank branches",
                     f"Banking opportunity score by ZIP (0-100) · small business = {small_def.lower()} · area: {catch_label}",
                     f"Sources: Census Business Patterns, FDIC Summary of Deposits. Data as of {loaded_at}. Employer locations only.")
    _barh(ax, labels, list(t["opportunity"]), 100, lambda x: f"{x:.0f}",
          extra=[f"{s:,.0f} biz · {int(b)} branch" + ("" if int(b) == 1 else "es") for s, b in zip(t["small_biz"], t["branches_in_zip"])])
    return _png(fig)


def deposit_share_png(share: pd.DataFrame, place: str, scope: str, hhi_text: str, loaded_at: str) -> bytes:
    s = share.head(8)
    fig, ax = _frame(f"Who holds the deposits around {place}",
                     f"Share of branch deposits within {scope} · {hhi_text}",
                     f"Source: FDIC Summary of Deposits. Data as of {loaded_at}. Deposits are as booked at each branch "
                     "and can reflect headquarters or online banking.")
    vmax = max(float(s["share_pct"].max()) * 1.25, 10)
    _barh(ax, [b.replace(", National Association", " N.A.") for b in s["bank"]], list(s["share_pct"]), vmax,
          lambda x: f"{x:.1f}%", color_vals=[min(100, v * 2.2) for v in s["share_pct"]],
          extra=[f"{int(n)} branch" + ("" if int(n) == 1 else "es") for n in s["branches"]])
    return _png(fig)
