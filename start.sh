#!/bin/sh
# Production entrypoint — used by Dockerfile (Render / Railway / Fly.io / DO)
#
#  • $PORT is provided by the platform (defaults to 8001 locally)
#  • The backend serves BOTH the React build AND the /api routes from the same port
#  • CORS is wide-open by default; restrict via CORS_ORIGINS env var if you want

set -e

cd /app/backend
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8001}" --workers "${WEB_CONCURRENCY:-1}" --log-level info
