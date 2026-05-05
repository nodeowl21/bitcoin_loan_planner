"""Tests for the price-frame generation utilities.

The price modelling helpers are deliberately pinned with smoke tests rather
than precise value assertions because they depend on real CSV data and on
random number generation. We do verify shape, monotonicity of the date index,
positivity of prices and reproducibility of the deterministic generator.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from backend.app.engine import (
    HISTORICAL_PRICE_FILE,
    POWER_LAW_PRICE_FILE,
    build_price_frame,
    generate_random_walk,
    get_btc_ath,
    load_historical_prices,
)

from .conftest import make_config


@pytest.fixture(autouse=True)
def _skip_if_data_missing():
    """The Historical and Power-Law modes require CSV files in the project root."""
    if not HISTORICAL_PRICE_FILE.exists() or not POWER_LAW_PRICE_FILE.exists():
        pytest.skip("Reference CSV data files are missing")


class TestGenerateRandomWalk:
    def test_returns_dataframe_with_price_column(self):
        df = generate_random_walk(start_price=100_000.0, years=1, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert "price" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 365 + 1

    def test_first_price_matches_start(self):
        df = generate_random_walk(start_price=50_000.0, years=1, seed=42)
        assert df["price"].iloc[0] == pytest.approx(50_000.0)

    def test_prices_are_strictly_positive(self):
        df = generate_random_walk(start_price=100_000.0, years=2, seed=42)
        assert (df["price"] > 0).all()

    def test_seed_yields_deterministic_output(self):
        a = generate_random_walk(start_price=100_000.0, years=1, seed=7)
        b = generate_random_walk(start_price=100_000.0, years=1, seed=7)
        pd.testing.assert_series_equal(a["price"], b["price"])

    def test_different_seeds_diverge(self):
        a = generate_random_walk(start_price=100_000.0, years=1, seed=1)
        b = generate_random_walk(start_price=100_000.0, years=1, seed=2)
        assert not a["price"].equals(b["price"])


class TestBuildPriceFrame:
    def test_generated_mode(self):
        config = make_config(sim_mode="Generated", sim_years=1)
        df = build_price_frame(current_btc_price=100_000.0, config=config)
        assert "price" in df.columns
        assert (df["price"] > 0).all()
        assert df["price"].iloc[0] == pytest.approx(100_000.0)

    def test_historical_mode_uses_csv_and_anchors_to_current_price(self):
        config = make_config(sim_mode="Historical", sim_years=2)
        df = build_price_frame(current_btc_price=100_000.0, config=config)
        assert "price" in df.columns
        assert len(df) > 0
        assert (df["price"] > 0).all()
        # The Historical path normalizes the slice so the first value matches
        # the current price.
        assert df["price"].iloc[0] == pytest.approx(100_000.0, rel=1e-6)
        assert df.index.is_monotonic_increasing

    def test_power_law_mode_returns_positive_series(self):
        config = make_config(sim_mode="Power-Law", sim_years=2)
        df = build_price_frame(current_btc_price=100_000.0, config=config)
        assert "price" in df.columns
        assert len(df) == 2 * 365 + 1
        assert (df["price"] > 0).all()
        assert df.index.is_monotonic_increasing


class TestHistoricalDataAccessors:
    def test_load_historical_prices_returns_sorted_frame(self):
        df = load_historical_prices()
        assert "price" in df.columns
        assert "snapped_at" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["snapped_at"])
        assert (df["price"] > 0).all()
        assert df["snapped_at"].is_monotonic_increasing

    def test_get_btc_ath_is_positive_and_finite(self):
        ath = get_btc_ath()
        assert ath > 0
        assert ath < 10_000_000  # sanity upper bound
        assert isinstance(ath, float)
