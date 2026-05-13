# Deploying M/V Al-bazm II Maritime Fuel Optimization

This app is a **single-container** FastAPI service that **serves both** the React frontend (built into static files) and the `/api/*` endpoints. That means **one service, one URL, one cost** on any cloud host that supports Docker.

---

## TL;DR

```bash
docker build -t mv-albazm-ii .
docker run -p 8001:8001 mv-albazm-ii
# → http://localhost:8001 serves the full app (UI + API)
```

That container is what you push to Render / Railway / Fly.io / DigitalOcean. No code changes needed between hosts.

---

## Option 1: Render (recommended — free tier works)

1. Push this repo to GitHub.
2. Go to https://render.com → **New** → **Blueprint**.
3. Connect your GitHub repo. Render auto-detects `render.yaml`.
4. Click **Apply**. First build takes ~5 minutes (frontend build + Python deps).
5. When it's live, Render gives you a URL like `https://mv-albazm-ii.onrender.com`.
6. Open it → the full app works.

**Free tier notes:**
- Free instances sleep after 15 min inactivity (first request after sleep takes ~30s)
- Upgrade to **Starter** ($7/mo) for always-on
- 512 MB RAM is enough; if you ever upgrade the ML model, jump to 1 GB

---

## Option 2: Railway

1. https://railway.app → **New Project** → **Deploy from GitHub repo**.
2. Railway reads `railway.json` and `Dockerfile` automatically.
3. In **Variables** tab, add:
   - `PORT=8080` (Railway injects its own, this is fallback)
   - `CORS_ORIGINS=*`
4. Click **Deploy**. You'll get a `*.up.railway.app` URL.

**Cost:** Railway has a $5/mo free credit; this app uses ~$2-3/mo.

---

## Option 3: Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Launch (reads fly.toml; --no-deploy so we can inspect first)
fly launch --no-deploy --copy-config --name mv-albazm-ii

# Deploy
fly deploy
```

**Cost:** Free tier (3 small VMs, 1 GB RAM each). This app fits comfortably.

---

## Option 4: DigitalOcean App Platform

1. https://cloud.digitalocean.com/apps → **Create App**
2. Connect GitHub repo → DO auto-detects the `Dockerfile`
3. Plan: **Basic** ($5/mo). 512 MB RAM is enough.
4. Set env vars:
   - `CORS_ORIGINS=*`
5. Deploy.

---

## What gets deployed

Layer in the image:
```
/app/
├── backend/                  ← FastAPI app + ML model + data files
│   ├── server.py
│   ├── ship_ml.py
│   ├── live_weather.py
│   ├── route_manager.py
│   ├── variable_speed.py
│   ├── physics_corrections.py
│   ├── engine_data.csv       ← 372+ voyages, used for ML training
│   ├── waypoints_real.json   ← Khalifa ↔ Ruwais waypoints
│   ├── model_files/          ← pre-trained joblibs
│   └── model_cache/          ← auto-created on first start
├── frontend/
│   └── build/                ← React production build (static HTML/JS/CSS)
└── start.sh                  ← Uvicorn entrypoint, honours $PORT
```

---

## Environment variables (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8001` | Port the server listens on. Render/Railway/Fly set this automatically. |
| `CORS_ORIGINS` | `*` | Comma-separated origins allowed to call the API. |
| `FRONTEND_BUILD_DIR` | autodetect | Override path to the React build. |
| `WEB_CONCURRENCY` | `1` | Uvicorn workers. Keep at 1 unless you have ≥1 GB RAM. |

There are **no required secrets** — Open-Meteo (weather) is free and key-less.

---

## Verifying the deployment

After deploy, hit these URLs (replace `your-app.onrender.com`):

```bash
curl https://your-app.onrender.com/api/health
# → {"status":"healthy","timestamp":"...","model_loaded":true,"routes_loaded":true}

curl 'https://your-app.onrender.com/api/weather?departure_port=Khalifa%20Port&arrival_port=Ruwais%20Port'
# → live Open-Meteo weather for both ports

curl https://your-app.onrender.com/
# → serves the React UI (HTML)
```

---

## Troubleshooting

**"Application failed to respond" on first request (Render free)**
→ The instance was sleeping; first request takes ~30s. Upgrade to Starter for always-on.

**Build fails with "ENOSPC" on frontend stage**
→ The free build runner ran out of disk. Use Render Starter or Railway/Fly.io.

**`/api/weather` returns fallback data**
→ Open-Meteo was unreachable from the deploy region. The app keeps working with sensible defaults; retry or check Open-Meteo's status page.

**ML model retrains on every restart**
→ The container's filesystem is ephemeral on most hosts. To persist `model_cache/`, attach a 100 MB volume (Render: "Disk"; Fly: "fly volumes"; Railway: "Volume").
