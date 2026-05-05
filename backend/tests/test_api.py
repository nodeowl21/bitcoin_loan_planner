"""Tests for the FastAPI endpoints.

These tests use the FastAPI TestClient and provide an explicit BTC price in the
request payload so that the endpoints do not need to reach out to CoinGecko.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def _base_payload() -> dict:
    return {
        "portfolio": {
            "btc_owned": 1.0,
            "currency": "USD",
            "btc_price": 100_000.0,
            "income_per_year": 0.0,
            "btc_saving_rate_percent": 0.0,
            "other_assets": 0.0,
            "loans": [],
        },
        "strategy": {
            "ltv": 50,
            "ltv_relative_to_ath": False,
            "enable_buy": False,
            "rebalance_buy": 10,
            "rebalance_buy_factor": 100,
            "enable_sell": False,
            "rebalance_sell": 10,
            "rebalance_sell_factor": 100,
        },
        "simulation": {
            "sim_mode": "Generated",
            "sim_years": 1,
            "exp_return": 0,
            "volatility": 0,
            "interval": "Daily",
            "interest": 10,
            "liquidation_ltv": 85,
            "enable_btc_saving": False,
        },
    }


class TestHealth:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestSimulateEndpoint:
    def test_simulate_returns_full_response(self):
        response = client.post("/simulate", json=_base_payload())
        assert response.status_code == 200

        data = response.json()
        assert {"series", "rebalancing_log", "summary"} <= data.keys()
        assert isinstance(data["series"], list)
        assert len(data["series"]) > 0

        first_point = data["series"][0]
        assert {"date", "price", "btc", "total_debt", "ltv", "real_ltv"} <= first_point.keys()

    def test_simulate_summary_fields_present(self):
        response = client.post("/simulate", json=_base_payload())
        summary = response.json()["summary"]

        for field in (
            "total_btc",
            "net_btc",
            "total_debt",
            "total_interest",
            "total_value",
            "net_value",
            "btc_delta",
            "net_btc_delta",
            "net_value_delta",
            "max_ltv",
            "liquidation_risk",
        ):
            assert field in summary

        assert summary["liquidation_risk"] in {"Low", "Medium", "High"}

    def test_simulate_with_loan_includes_debt(self):
        payload = _base_payload()
        payload["portfolio"]["loans"] = [
            {
                "id": "loan-1",
                "platform": "Test",
                "amount": 20_000.0,
                "interest": 10.0,
                "start_date": "2024-01-01",
                "term_months": None,
                "liquidation_ltv": 85.0,
                "btc_bought": 0.0,
            }
        ]

        response = client.post("/simulate", json=payload)
        assert response.status_code == 200
        summary = response.json()["summary"]
        assert summary["total_debt"] > 0

    def test_simulate_rejects_invalid_strategy(self):
        payload = _base_payload()
        payload["strategy"]["ltv"] = 200  # > 100, should fail validation
        response = client.post("/simulate", json=payload)
        assert response.status_code == 422


class TestOptimizeEndpoint:
    def test_optimize_returns_strategy_and_delta(self):
        response = client.post("/optimize", json=_base_payload())
        assert response.status_code == 200

        data = response.json()
        assert {"strategy", "net_btc_delta"} == data.keys()

        strategy = data["strategy"]
        for field in (
            "ltv",
            "ltv_relative_to_ath",
            "enable_buy",
            "rebalance_buy",
            "rebalance_buy_factor",
            "enable_sell",
            "rebalance_sell",
            "rebalance_sell_factor",
        ):
            assert field in strategy

        assert 5 <= strategy["ltv"] <= 80
        assert 0 <= strategy["rebalance_buy"] <= 30
        assert isinstance(data["net_btc_delta"], (int, float))
