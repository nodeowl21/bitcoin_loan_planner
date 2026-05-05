from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .engine import get_live_btc_price, optimize_strategy, simulate
from .models import Currency, OptimizationResponse, SimulationRequest, SimulationResponse


app = FastAPI(title="Bitcoin Loan Planner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/btc-price")
def btc_price(currency: Currency = Query(default="USD")) -> dict[str, float | str]:
    price = get_live_btc_price(currency)
    if price is None:
        raise HTTPException(status_code=502, detail="Could not fetch BTC price")
    return {"currency": currency, "price": price}


@app.post("/simulate", response_model=SimulationResponse)
def run_simulation(request: SimulationRequest) -> dict:
    live_price = get_live_btc_price(request.portfolio.currency) if request.portfolio.btc_price is None else None
    current_btc_price = request.portfolio.btc_price or live_price or 100_000.0
    return simulate(request, current_btc_price=current_btc_price)


@app.post("/optimize", response_model=OptimizationResponse)
def run_optimization(request: SimulationRequest) -> dict:
    live_price = get_live_btc_price(request.portfolio.currency) if request.portfolio.btc_price is None else None
    current_btc_price = request.portfolio.btc_price or live_price or 100_000.0
    return optimize_strategy(request, current_btc_price=current_btc_price)
