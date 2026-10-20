"""Plain-English interpretation of scores, shared by every tab and by the memo."""
from __future__ import annotations

import pandas as pd

# (minimum score, label, colour, what it means)
BANDS = [
    (80, "Exceptional", "#0b1f3a", "Leads this set on nearly every factor you weighted."),
    (65, "Strong", "#0e7490", "Clearly above most markets; a priority to look at first."),
    (50, "Solid", "#3a8fa3", "Mid-to-upper pack; attractive if it fits your specific needs."),
    (35, "Moderate", "#b7791f", "Mixed picture; strong on some factors, weak on others."),
    (0, "Weak", "#9b2c2c", "Trails most of the set on the factors you weighted."),
]
SCORE_NOTE = (
    "How to read the score: it compares markets with each other on the factors and weights shown, from 0 (weakest "
    "in the set) to 100 (best in the set on every factor). 80+ Exceptional, 65-79 Strong, 50-64 Solid, 35-49 "
    "Moderate, under 35 Weak. It is relative, not an absolute grade: differences under about 5 points are not "
    "meaningful, and a high score does not replace local due diligence."
)


def band(score) -> tuple[str, str, str]:
    """(label, colour, meaning) for a 0-100 score."""
    if score is None or pd.isna(score):
        return "n/a", "#64748b", "Not enough data to rate."
    for lo, label, color, meaning in BANDS:
        if score >= lo:
            return label, color, meaning
    return BANDS[-1][1], BANDS[-1][2], BANDS[-1][3]


def rating(score) -> str:
    return band(score)[0]


def driver_sentence(row, parts: dict, total_w: float) -> str:
    """What is pushing one market's score up or down, in one sentence."""
    items = []
    for name, (_, w) in parts.items():
        max_pts = w / total_w * 100 if total_w else 0
        if max_pts > 0:
            pts = float(row[f"c_{name}"])
            items.append((name, pts, max_pts, pts / max_pts))
    if not items:
        return ""
    best = max(items, key=lambda x: x[1])
    worst = min(items, key=lambda x: x[3])
    s = f"Biggest boost: {best[0].lower()} ({best[1]:.0f} of {best[2]:.0f} possible points)."
    if worst[0] != best[0]:
        s += f" Weakest: {worst[0].lower()} ({worst[1]:.0f} of {worst[2]:.0f})."
    return s


def band_legend_html() -> str:
    chips = "".join(f'<span style="display:inline-block;margin:2px 6px 2px 0;padding:2px 10px;border-radius:999px;'
                    f'background:{c};color:#fff;font-size:.74rem;font-weight:600">{lo}+ {lab}</span>'
                    for lo, lab, c, _ in BANDS)
    return chips


def guide_market(st) -> None:
    with st.expander("📏 How to read the scores", expanded=False):
        st.markdown(
            "**What the score is.** A 0 to 100 comparison of the markets in this dashboard with each other, built from "
            "the factors and weights in the sidebar. 100 means best in the set on every factor you weighted; 0 means "
            "weakest. It is **relative**: it ranks these markets, it does not grade them against the whole country.")
        rows = "\n".join(f"| **{lo}+** | {lab} | {m} |" for lo, lab, _, m in BANDS)
        st.markdown("| Score | Rating | Meaning |\n|---|---|---|\n" + rows)
        st.markdown(
            "**Using it well**\n"
            "- Differences under about 5 points are not meaningful. Treat close markets as a tie and decide on the "
            "components.\n"
            "- Open **Market spotlight** to see which factor is carrying a market and which is dragging it.\n"
            "- Changing the weights or choosing a **scoring preset** changes the ranking by design.\n"
            "- LQ (sector concentration) and GDP growth are county-level, so ZIPs in the same county share them.\n"
            "- If the header says some sources are on sample data, treat scores as illustrative.")


def guide_banking(st) -> None:
    with st.expander("📏 How to read the banking numbers", expanded=False):
        st.markdown(
            "**Opportunity (0 to 100)** blends how many small businesses a ZIP has, how many it has per bank branch "
            "(the under-served signal) and how dense they are per 10,000 residents, using the weights you set. It is "
            "relative to the ZIPs being analyzed and uses the same ratings as the market score (80+ Exceptional, "
            "65+ Strong, 50+ Solid, 35+ Moderate).")
        st.markdown(
            "| Measure | How to read it |\n|---|---|\n"
            "| **Coverage** | Businesses per branch against the median ZIP. *Underserved* = 25% or more above the median; "
            "*Saturated* = 25% or more below; *No branch* = no FDIC branch in the ZIP. |\n"
            "| **Deposit capture index** | Deposits per small business vs the median ZIP (100 = median). Under 60 = "
            "*Under-captured* (deposits look thin for the number of businesses); over 140 = *Deposit-rich*. |\n"
            "| **Deposit HHI** | How concentrated deposits are among banks in the area, 0 to 10,000. Over 1,800 = highly "
            "concentrated, 1,000 to 1,800 = moderate, under 1,000 = fragmented. Informational, not an antitrust finding. |\n"
            "| **Weak branches** | Branches booking under half the median branch's deposits in the area. Overperformers "
            "(over 2x) are often headquarters or online-booked branches, so verify them. |")
        st.caption("Counts are employer establishments (locations), not companies, and exclude sole proprietors. "
                   "Branch deposits are as booked at each branch.")
