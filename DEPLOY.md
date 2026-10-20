# Synoptic BI: put it on the web

## Run locally first
```
pip install -r requirements.txt
streamlit run app.py
```

## Option A: Streamlit Community Cloud (free, easiest)
1. Create a GitHub repo and upload everything in this folder (keep `app.py`, `data_sources.py`,
   `requirements.txt` and the `.streamlit/` folder at the repo root). A private repo works.
2. Go to https://share.streamlit.io, sign in with GitHub, click **New app**, pick the repo,
   branch `main`, main file `app.py`, then **Deploy**.
3. Open **Settings > Secrets** for the app and paste the contents of `.streamlit/secrets.toml.example`
   with your real values.
4. You get a public URL like `https://your-app.streamlit.app` that works from any device or network.

## Option B: any container host (Render, Fly.io, Railway, Google Cloud Run, etc.)
The included `Dockerfile` listens on `$PORT`. Create a web service from the repo, choose Docker,
and add the secrets as environment variables (`APP_PASSWORD`, `CENSUS_API_KEY`, `BEA_API_KEY`, `ANTHROPIC_API_KEY`).

## Before you share the link
- **Set `APP_PASSWORD`** if the dashboard should not be public. Without it, anyone with the URL can view it.
- **Get the two free keys** or those parts show sample values: `CENSUS_API_KEY` (required for income, education,
  population and the Banking tab) and `BEA_API_KEY` (GDP growth). Click the activation link in each key email.
- After deploying, open **Data Sources > Run connection check** to see which sources work from your host.
- Data is cached for 6 hours. Use the sidebar **Refresh live data** button to force a reload.
- Add markets by appending rows to `MARKETS` in `data_sources.py` (ZIP, city, county, county FIPS, lat, lon).

## AI Analyst (optional)
- Create an API key in the Claude Console and add it as `ANTHROPIC_API_KEY`. Without it the AI tab simply shows
  "switched off" and the rest of the dashboard works normally.
- API usage is billed to your account. Always set `APP_PASSWORD` on a public deployment so strangers cannot run up
  your bill, and consider setting a monthly spend limit in the Console.
- Chat uses a smaller, cheaper model and the brief uses a stronger one. Change them with `AI_CHAT_MODEL` and
  `AI_BRIEF_MODEL` (see docs.claude.com for current model names).
- The AI never sees race/ethnicity columns, and every number it states comes from the dashboard's own data.

## Banking tab (small businesses + FDIC branches)
- Needs `CENSUS_API_KEY` (free). Census rejects requests without a key.
- Bank branches and deposits come from the FDIC Summary of Deposits (annual June 30 survey). No key is normally
  needed; if the tab reports an FDIC error, add a free api.data.gov key as `FDIC_API_KEY`.
- You can type extra Florida ZIP codes into the tab to analyze any area.

## New business features
- **Presets** (sidebar): one click sets filters and weights for Banking expansion, Tech/professional hiring or Wealth
  management. Population growth is not in the data, so GDP growth stands in for it.
- **Drive-time areas** (Banking tab > Competitor area): needs a free `ORS_API_KEY` (openrouteservice.org). Free keys allow
  about 20 isochrone requests a minute and 500 a day, so results are cached for a week; the first load of 17+ ZIPs can take a
  minute. Without a key the app keeps using straight-line radius.
- **Competitor vulnerability**: deposit capture index (100 = median ZIP), deposit HHI with plain-English bands, and branch
  outliers (under half or over twice the median branch in the area). HHI is informational, not an antitrust finding.
- **Batch tab**: upload a CSV (or paste a list) of up to 60 Florida ZIPs or addresses. Only the ZIP is used; street addresses
  are not geocoded. Opportunity is relative to the ZIPs in the upload.
- **Memo tab**: builds a branded Word and PDF memo (uses your AI brief as the executive summary when one exists).
  Needs `python-docx` and `reportlab` (already in requirements.txt).
