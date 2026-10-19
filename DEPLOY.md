# Eric Joseph BI: put it on the web

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
and add the secrets as environment variables (`APP_PASSWORD`, `CENSUS_API_KEY`, `BEA_API_KEY`).

## Before you share the link
- **Set `APP_PASSWORD`** if the dashboard should not be public. Without it, anyone with the URL can view it.
- **Get the free BEA key** or GDP growth will show sample values. Census and BLS work without keys.
- Data is cached for 6 hours. Use the sidebar **Refresh live data** button to force a reload.
- Add markets by appending rows to `MARKETS` in `data_sources.py` (ZIP, city, county, county FIPS, lat, lon).
