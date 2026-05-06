# syntax=docker/dockerfile:1
# Single image: build React, then run FastAPI + serve frontend/dist (same origin).
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY btc-usd-max.csv bitcoin_data.csv ./

COPY --from=frontend-build /app/frontend/dist ./frontend/dist

ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
