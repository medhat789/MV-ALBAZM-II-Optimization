FROM node:20-alpine AS frontend-build
WORKDIR /frontend

COPY frontend/package.json ./
COPY frontend/yarn.lock* ./

RUN if [ -f yarn.lock ]; then yarn install --frozen-lockfile --network-timeout 600000; else yarn install --network-timeout 600000; fi

COPY frontend/ ./
ENV REACT_APP_BACKEND_URL=""
RUN yarn build

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 curl && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/

COPY --from=frontend-build /frontend/build ./frontend/build

COPY start.sh ./start.sh
RUN chmod +x ./start.sh

ENV PORT=8001
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 CMD curl -fsS http://127.0.0.1:${PORT}/api/health || exit 1

CMD ["./start.sh"]
