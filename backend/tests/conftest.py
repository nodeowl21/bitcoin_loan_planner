"""Shared fixtures and helpers for backend tests."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from backend.app.models import Loan, Portfolio, SimulationConfig, StrategyConfig


def make_price_df(prices: list[float], start_date: dt.date | None = None) -> pd.DataFrame:
    """Build a deterministic daily-frequency price frame for simulation tests."""
    start = start_date or dt.date(2024, 1, 1)
    dates = pd.date_range(start=start, periods=len(prices), freq="D")
    return pd.DataFrame({"price": prices}, index=dates)


def make_loan(**overrides) -> Loan:
    defaults: dict = dict(
        id="test-loan",
        platform="Test Platform",
        amount=50_000.0,
        interest=10.0,
        start_date=dt.date(2024, 1, 1),
        term_months=None,
        liquidation_ltv=85.0,
        btc_bought=0.0,
    )
    defaults.update(overrides)
    return Loan(**defaults)


def make_portfolio(loans: list[Loan] | None = None, **overrides) -> Portfolio:
    defaults: dict = dict(
        btc_owned=1.0,
        currency="USD",
        btc_price=100_000.0,
        income_per_year=0.0,
        btc_saving_rate_percent=0.0,
        other_assets=0.0,
        loans=loans or [],
    )
    defaults.update(overrides)
    return Portfolio(**defaults)


def make_strategy(**overrides) -> StrategyConfig:
    defaults: dict = dict(
        ltv=50.0,
        ltv_relative_to_ath=False,
        enable_buy=False,
        rebalance_buy=10.0,
        rebalance_buy_factor=100.0,
        enable_sell=False,
        rebalance_sell=10.0,
        rebalance_sell_factor=100.0,
    )
    defaults.update(overrides)
    return StrategyConfig(**defaults)


def make_config(**overrides) -> SimulationConfig:
    defaults: dict = dict(
        sim_mode="Generated",
        sim_years=1,
        exp_return=0.0,
        volatility=0.0,
        interval="Daily",
        interest=10.0,
        liquidation_ltv=85.0,
        enable_btc_saving=False,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


@pytest.fixture
def price_factory():
    return make_price_df


@pytest.fixture
def loan_factory():
    return make_loan


@pytest.fixture
def portfolio_factory():
    return make_portfolio


@pytest.fixture
def strategy_factory():
    return make_strategy


@pytest.fixture
def config_factory():
    return make_config
