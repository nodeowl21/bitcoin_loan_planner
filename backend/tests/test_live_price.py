"""Tests for live BTC spot price fetching (multi-provider fallbacks)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.app.engine import get_live_btc_price


def test_get_live_btc_price_usd_uses_binance_when_available() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"price": "95000.5"}

    with patch("backend.app.engine.requests.get", return_value=mock_resp) as mock_get:
        price = get_live_btc_price("USD")

    assert price == 95000.5
    mock_get.assert_called_once()
    assert "binance.com" in mock_get.call_args[0][0]


def test_get_live_btc_price_usd_falls_back_to_coinbase() -> None:
    binance_fail = MagicMock()
    binance_fail.raise_for_status.side_effect = OSError("blocked")

    coinbase_ok = MagicMock()
    coinbase_ok.raise_for_status.return_value = None
    coinbase_ok.json.return_value = {"data": {"amount": "88000.0"}}

    with patch("backend.app.engine.requests.get", side_effect=[binance_fail, coinbase_ok]) as mock_get:
        price = get_live_btc_price("USD")

    assert price == 88000.0
    assert mock_get.call_count == 2
    assert "coinbase.com" in mock_get.call_args_list[1][0][0]


def test_get_live_btc_price_usd_falls_back_to_coingecko() -> None:
    fail = MagicMock()
    fail.raise_for_status.side_effect = OSError("nope")

    cg_ok = MagicMock()
    cg_ok.raise_for_status.return_value = None
    cg_ok.json.return_value = {"bitcoin": {"usd": 70000.0}}

    with patch("backend.app.engine.requests.get", side_effect=[fail, fail, cg_ok]) as mock_get:
        price = get_live_btc_price("USD")

    assert price == 70000.0
    assert mock_get.call_count == 3
    assert "coingecko.com" in mock_get.call_args_list[2][0][0]


def test_get_live_btc_price_eur_uses_binance_btceur() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"price": "88000.0"}

    with patch("backend.app.engine.requests.get", return_value=mock_resp) as mock_get:
        price = get_live_btc_price("EUR")

    assert price == 88000.0
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert kwargs.get("params", {}).get("symbol") == "BTCEUR"
