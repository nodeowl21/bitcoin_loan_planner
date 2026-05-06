#!/usr/bin/env python3
"""Refresh ``btc-usd-max.csv`` and ``bitcoin_data.csv`` from daily BTC/USD history.

The app only uses the **price** column from ``btc-usd-max.csv``. ``market_cap``
is set to ``0`` for compatibility with the old export shape; ``total_volume``
uses CryptoCompare's ``volumeto`` (approx. USD volume).

**Data source:** CryptoCompare public endpoint (no key). Respect fair use;
see https://www.cryptocompare.com/

Usage (from repo root)::

    python3 scripts/update_price_data.py

Or with the project venv::

    .venv/bin/python scripts/update_price_data.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
HIST_CSV = ROOT / "btc-usd-max.csv"
POWER_CSV = ROOT / "bitcoin_data.csv"

CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/histoday"
HEADERS = {
    "User-Agent": "BitcoinLoanPlanner-data-refresh/1.0 (+https://github.com/nodeowl21/bitcoin_loan_planner)",
}


def fetch_histoday() -> list[dict]:
    response = requests.get(
        CRYPTOCOMPARE_URL,
        params={"fsym": "BTC", "tsym": "USD", "allData": "true"},
        timeout=180,
        headers=HEADERS,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("Response") != "Success" or not payload.get("Data", {}).get("Data"):
        raise RuntimeError(f"CryptoCompare error: {json.dumps(payload)[:500]}")
    return payload["Data"]["Data"]


def _utc_day_rows(rows: Iterable[dict]) -> tuple[list[dict], dict[str, float]]:
    """Return hist rows + last Value per calendar day (dedupe)."""
    hist: list[dict] = []
    by_date: dict[str, float] = {}
    for row in rows:
        ts = int(row["time"])
        dt = datetime.fromtimestamp(ts, tz=UTC)
        snapped = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        close = float(row["close"])
        vol_usd = float(row.get("volumeto") or 0.0)
        hist.append(
            {
                "snapped_at": snapped,
                "price": close,
                "market_cap": 0.0,
                "total_volume": vol_usd,
            }
        )
        by_date[dt.strftime("%Y-%m-%d")] = close
    return hist, by_date


def main() -> int:
    raw = fetch_histoday()
    hist_rows, by_date = _utc_day_rows(raw)
    hist_rows.sort(key=lambda r: r["snapped_at"])
    power_rows = [{"Date": d, "Value": v} for d, v in sorted(by_date.items())]

    with HIST_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["snapped_at", "price", "market_cap", "total_volume"])
        w.writeheader()
        w.writerows(hist_rows)

    with POWER_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Date", "Value"])
        w.writeheader()
        w.writerows(power_rows)

    print(f"Wrote {len(hist_rows)} rows → {HIST_CSV.relative_to(ROOT)}")
    print(f"Wrote {len(power_rows)} rows → {POWER_CSV.relative_to(ROOT)}")
    last = power_rows[-1]
    print(f"Latest date: {last['Date']}  close: {last['Value']:,.2f} USD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
