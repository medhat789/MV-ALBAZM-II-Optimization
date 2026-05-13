============================================================
M/V Al-bazm II Maritime Fuel Optimization System
Multi-stage Dockerfile — builds React, then runs FastAPI
Works on Render / Railway / Fly.io / DigitalOcean App Platform
============================================================
---------- Stage 1: build the React frontend ----------
FROM node:18-alpine AS frontend-build WORKDIR /frontend

Install dependencies first (cache layer).
We copy package.json (required) plus any lockfile if present.
The wildcard in yarn.lock* means the build won't fail when the lockfile
is missing from the repo — yarn will simply regenerate it.
COPY frontend/package.json ./ COPY frontend/yarn.lock* ./ RUN if [ -f yarn.lock ]; then
yarn install --frozen-lockfile --network-timeout 600000;
else
echo "⚠️ No yarn.lock found — generating a fresh one (build will still be reproducible per package.json)";
yarn install --network-timeout 600000;
fi

Copy source and build
COPY frontend/ ./

REACT_APP_BACKEND_URL is empty in prod → same-origin (the FastAPI service serves
both the built static files AND the /api routes), so the React app calls /api/...
ENV REACT_APP_BACKEND_URL="" RUN yarn build

---------- Stage 2: Python backend serving React build ----------
FROM python:3.11-slim

System deps required by pandas/scipy/sklearn (BLAS, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends
build-essential libgomp1 curl &&
apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
PIP_NO_CACHE_DIR=1
PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

Install Python deps first (cache layer)
COPY backend/requirements.txt ./backend/requirements.txt RUN pip install --no-cache-dir -r backend/requirements.txt

Copy backend source + data files + pretrained model files
COPY backend/ ./backend/

Bring in the built React app produced in stage 1
COPY --from=frontend-build /frontend/build ./frontend/build

Hand off to start.sh — it honours $PORT (Render/Railway/Fly inject this)
COPY start.sh ./start.sh RUN chmod +x ./start.sh

Most hosts inject PORT; default to 8001 for local docker run
ENV PORT=8001 EXPOSE 8001

Healthcheck the API endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3
CMD curl -fsS http://127.0.0.1:${PORT}/api/health || exit 1

CMD ["./start.sh"] ===== END OF FILE ===== Exit code: 0
