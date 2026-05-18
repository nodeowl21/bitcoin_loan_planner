from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution

from .models import Loan, Portfolio, SimulationConfig, StrategyConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_PRICE_FILE = PROJECT_ROOT / "btc-usd-max.csv"
POWER_LAW_PRICE_FILE = PROJECT_ROOT / "bitcoin_data.csv"

# Identifiable UA: some APIs throttle or block the default Python-requests user agent.
_HTTP_HEADERS = {
    "User-Agent": "BitcoinLoanPlanner/1.0 (+https://github.com/nodeowl21/bitcoin_loan_planner)",
}


def _price_binance(symbol: str) -> float | None:
    try:
        response = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=12,
            headers=_HTTP_HEADERS,
        )
        response.raise_for_status()
        return float(response.json()["price"])
    except Exception:
        return None


def _price_coinbase(vs: str) -> float | None:
    """vs is the quote currency for Coinbase, e.g. USD or EUR."""
    try:
        response = requests.get(
            f"https://api.coinbase.com/v2/prices/BTC-{vs.upper()}/spot",
            timeout=12,
            headers=_HTTP_HEADERS,
        )
        response.raise_for_status()
        return float(response.json()["data"]["amount"])
    except Exception:
        return None


def _price_coingecko(currency: str) -> float | None:
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        response = requests.get(
            url,
            params={"ids": "bitcoin", "vs_currencies": currency.lower()},
            timeout=12,
            headers=_HTTP_HEADERS,
        )
        response.raise_for_status()
        data = response.json()
        return float(data["bitcoin"][currency.lower()])
    except Exception:
        return None


def get_live_btc_price(currency: str) -> float | None:
    """Spot price for BTC. Uses several providers: CoinGecko often rate-limits or
    blocks datacenter IPs (e.g. Render), so Binance/Coinbase are tried first."""
    cur = (currency or "USD").upper()
    if cur == "USD":
        price = _price_binance("BTCUSDT")
        if price is not None:
            return price
        price = _price_coinbase("USD")
        if price is not None:
            return price
        return _price_coingecko("USD")
    if cur == "EUR":
        price = _price_binance("BTCEUR")
        if price is not None:
            return price
        price = _price_coinbase("EUR")
        if price is not None:
            return price
        return _price_coingecko("EUR")
    return _price_coingecko(cur)


def generate_random_walk(
    start_price: float,
    years: int = 5,
    annual_return: float = 0.5,
    daily_volatility: float = 0.05,
    seed: int | None = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = years * 365
    dt_step = 1 / 365
    prices = [start_price]

    for _ in range(days):
        shock = rng.normal(loc=annual_return * dt_step, scale=daily_volatility * math.sqrt(dt_step))
        prices.append(prices[-1] * math.exp(shock))

    dates = pd.date_range(start=dt.date.today(), periods=days + 1, freq="D")
    return pd.DataFrame({"price": prices}, index=dates)


def btc_price_model_power_law(start_year: int, years: int) -> pd.Series:
    df = pd.read_csv(POWER_LAW_PRICE_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Value"] > 0].copy()
    df["Days"] = (df["Date"] - pd.Timestamp("2009-01-09")).dt.days

    def log_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a + b * np.log(x)

    x_data = df["Days"].values
    y_data = np.log(df["Value"].values)
    popt, _ = curve_fit(log_func, x_data, y_data)
    a_fit, b_fit = popt

    start_date = pd.Timestamp(f"{start_year}-01-01")
    future_dates = pd.date_range(start=start_date, periods=years * 365 + 1, freq="D")
    x_future = (future_dates - pd.Timestamp("2009-01-09")).days
    x_future = np.where(x_future > 0, x_future, 1)
    prices = np.exp(log_func(x_future, a_fit, b_fit))
    return pd.Series(prices, index=future_dates)


def load_historical_prices() -> pd.DataFrame:
    df = pd.read_csv(HISTORICAL_PRICE_FILE)
    df["snapped_at"] = pd.to_datetime(df["snapped_at"])
    return df.sort_values("snapped_at")


def get_btc_ath() -> float:
    return float(load_historical_prices()["price"].max())


def build_price_frame(current_btc_price: float, config: SimulationConfig) -> pd.DataFrame:
    if config.sim_mode == "Generated":
        return generate_random_walk(
            start_price=current_btc_price,
            years=config.sim_years,
            annual_return=config.exp_return / 100,
            daily_volatility=config.volatility / 100,
        )

    if config.sim_mode == "Power-Law":
        price_series = btc_price_model_power_law(start_year=dt.date.today().year, years=config.sim_years)
        return pd.DataFrame({"price": price_series})

    df_raw = load_historical_prices()
    price_series = df_raw.set_index("snapped_at")["price"].sort_index()
    end_date = price_series.index.max()
    start_date = end_date - pd.DateOffset(years=config.sim_years)
    sliced = price_series.loc[start_date:end_date]
    relative_prices = sliced / sliced.iloc[0]
    simulated_prices = relative_prices * current_btc_price
    future_dates = pd.date_range(start=dt.date.today(), periods=len(simulated_prices), freq="D")
    return pd.DataFrame({"price": simulated_prices.values}, index=future_dates)


def _add_months(value: dt.date, months: int) -> dt.date:
    timestamp = pd.Timestamp(value) + pd.DateOffset(months=months)
    return timestamp.date()


def _json_float(value: float) -> float:
    if math.isfinite(value):
        return float(value)
    return 1_000_000_000.0


def run_simulation(
    strategy: StrategyConfig,
    portfolio: Portfolio,
    config: SimulationConfig,
    current_btc: float,
    price_df: pd.DataFrame,
    reference_value: float,
    loans: list[Loan],
) -> tuple[pd.DataFrame, list[dict]]:
    target_ltv = strategy.ltv / 100
    rebalance_buy = strategy.rebalance_buy / 100
    rebalance_buy_factor = strategy.rebalance_buy_factor / 100
    rebalance_sell = strategy.rebalance_sell / 100
    rebalance_sell_factor = strategy.rebalance_sell_factor / 100
    rebalance_days = {"Daily": 1, "Weekly": 7, "Monthly": 30, "Yearly": 365}[config.interval]

    fixed_interest = 0.0
    rows: list[dict] = []
    rebalancing_log: list[dict] = []
    liquidated = False
    active_loans: list[dict] = []
    sim_start_date = price_df.index[0].date()

    for loan in loans:
        end_date = _add_months(loan.start_date, loan.term_months) if loan.term_months else None
        active_loans.append(
            {
                "platform": loan.platform or "Loan",
                "amount": loan.amount,
                "interest": loan.interest / 100,
                "start_date": loan.start_date,
                "term_months": loan.term_months,
                "liquidation_ltv": loan.liquidation_ltv / 100,
                "end_date": end_date,
                "paid": False,
                "accrued_interest": 0.0,
            }
        )

    total_debt = 0.0
    for loan in active_loans:
        if loan["end_date"] and loan["end_date"] < sim_start_date:
            loan["paid"] = True
            continue

        if loan["start_date"] < sim_start_date:
            days_running = (sim_start_date - loan["start_date"]).days
            accrued = loan["amount"] * loan["interest"] * days_running / 365
            loan["accrued_interest"] += accrued
            total_debt += loan["amount"] + accrued
        else:
            total_debt += loan["amount"]

    for i, (date_index, row) in enumerate(price_df.iterrows()):
        price = float(row["price"])
        date = date_index.date()

        if config.enable_btc_saving:
            daily_income = portfolio.income_per_year / 365
            daily_saving_fiat = (portfolio.btc_saving_rate_percent / 100) * daily_income
            if daily_saving_fiat > 0 and price > 0:
                current_btc += daily_saving_fiat / price

        real_collateral_before_repayments = current_btc * price
        real_ltv_before_repayments = (
            total_debt / real_collateral_before_repayments if real_collateral_before_repayments > 0 else float("inf")
        )

        repay_log_ltv_before = real_ltv_before_repayments
        for loan in active_loans:
            if loan["paid"] or not loan["end_date"]:
                continue
            if date >= loan["end_date"]:
                repayment_amount = loan["amount"] + loan["accrued_interest"]
                btc_to_sell = repayment_amount / price if price > 0 else 0
                current_btc = max(0.0, current_btc - btc_to_sell)
                loan["paid"] = True
                total_debt = max(0.0, total_debt - repayment_amount)
                real_collateral_after = current_btc * price
                real_ltv_after = (
                    total_debt / real_collateral_after if real_collateral_after > 0 else float("inf")
                )
                rebalancing_log.append(
                    {
                        "date": date.isoformat(),
                        "action": f"Repay: {loan['platform']}",
                        "btc_delta": _json_float(-btc_to_sell),
                        "price": _json_float(price),
                        "fiat_delta": _json_float(-repayment_amount),
                        "new_total_btc": _json_float(current_btc),
                        "new_total_debt": _json_float(total_debt),
                        "ltv_before": _json_float(repay_log_ltv_before),
                        "ltv_after": _json_float(real_ltv_after),
                    }
                )
                repay_log_ltv_before = real_ltv_after

        accrued_interest = 0.0
        for loan in active_loans:
            if loan["paid"]:
                continue
            daily_interest = loan["amount"] * loan["interest"] / 365
            loan["accrued_interest"] += daily_interest
            accrued_interest += daily_interest
            total_debt += daily_interest

        fixed_interest += accrued_interest
        real_collateral = current_btc * price
        real_ltv = total_debt / real_collateral if real_collateral > 0 else float("inf")

        if strategy.ltv_relative_to_ath:
            reference_value = max(reference_value, price)
            rebalance_collateral = current_btc * reference_value
            rebalance_ltv = total_debt / rebalance_collateral if rebalance_collateral > 0 else float("inf")
        else:
            reference_value = price
            rebalance_collateral = real_collateral
            rebalance_ltv = real_ltv

        rebalanced = False
        action = ""
        delta_btc = 0.0
        fiat_delta = 0.0

        if not liquidated:
            unpaid_loans = [loan for loan in active_loans if not loan["paid"]]
            if unpaid_loans:
                weighted_liquidation_ltv = sum(
                    loan["liquidation_ltv"] * loan["amount"] for loan in unpaid_loans
                ) / sum(loan["amount"] for loan in unpaid_loans)
            else:
                weighted_liquidation_ltv = config.liquidation_ltv / 100

            if real_ltv > weighted_liquidation_ltv:
                liquidation_value = current_btc * price
                remaining_value = max(liquidation_value - total_debt, 0.0)
                new_btc = remaining_value / price if price > 0 else 0.0
                delta_btc = new_btc - current_btc
                fiat_delta = -total_debt
                current_btc = new_btc
                total_debt = 0.0
                for loan in active_loans:
                    loan["paid"] = True
                action = "Liquidation"
                liquidated = True
                rebalanced = True
            elif i % rebalance_days == 0:
                deviation = rebalance_ltv - target_ltv

                if strategy.enable_sell and deviation > rebalance_sell:
                    adjusted_ltv = target_ltv + rebalance_sell * (1 - rebalance_sell_factor)
                    btc_to_sell = (total_debt - adjusted_ltv * current_btc * reference_value) / (
                        reference_value * (1 - adjusted_ltv)
                    )
                    btc_to_sell = max(0.0, btc_to_sell)

                    if btc_to_sell > 0:
                        usd_available = btc_to_sell * reference_value
                        sorted_loans = sorted(
                            [loan for loan in active_loans if not loan["paid"]],
                            key=lambda item: (
                                -item["interest"],
                                item["term_months"] if item["term_months"] is not None else float("inf"),
                            ),
                        )

                        repaid_amount = 0.0
                        for loan in sorted_loans:
                            if usd_available <= 0:
                                break
                            outstanding = loan["amount"] + loan["accrued_interest"]
                            repay_amount = min(usd_available, outstanding)
                            usd_available -= repay_amount
                            repaid_amount += repay_amount
                            loan["amount"] -= repay_amount
                            if loan["amount"] <= 0.01:
                                loan["paid"] = True

                        current_btc = max(0.0, current_btc - btc_to_sell)
                        total_debt = max(0.0, total_debt - repaid_amount)
                        rebalanced = True
                        action = "Sell"
                        delta_btc = -btc_to_sell
                        fiat_delta = repaid_amount

                elif strategy.enable_buy and deviation < -rebalance_buy:
                    adjusted_ltv = target_ltv - rebalance_buy * (1 - rebalance_buy_factor)
                    new_credit = (adjusted_ltv * rebalance_collateral - total_debt) / (1 - adjusted_ltv)
                    new_credit = max(0.0, new_credit)
                    btc_to_buy = new_credit / reference_value if reference_value > 0 else 0

                    if btc_to_buy > 0:
                        active_loans.append(
                            {
                                "platform": "Simulated",
                                "amount": new_credit,
                                "interest": config.interest / 100,
                                "start_date": date,
                                "term_months": None,
                                "liquidation_ltv": config.liquidation_ltv / 100,
                                "paid": False,
                                "accrued_interest": 0.0,
                                "end_date": None,
                            }
                        )
                        current_btc += btc_to_buy
                        total_debt += new_credit
                        rebalanced = True
                        action = "Buy"
                        delta_btc = btc_to_buy
                        fiat_delta = -new_credit

        if rebalanced:
            ltv_after = total_debt / (current_btc * reference_value) if current_btc > 0 else float("inf")
            rebalancing_log.append(
                {
                    "date": date.isoformat(),
                    "action": action,
                    "ltv_before": _json_float(rebalance_ltv),
                    "ltv_after": _json_float(ltv_after),
                    "btc_delta": _json_float(delta_btc),
                    "price": _json_float(price),
                    "fiat_delta": _json_float(fiat_delta),
                    "new_total_btc": _json_float(current_btc),
                    "new_total_debt": _json_float(total_debt),
                }
            )

        rows.append(
            {
                "date": date_index,
                "price": price,
                "btc": current_btc,
                "total_debt": total_debt,
                "total_interest": fixed_interest,
                "ltv": rebalance_ltv,
                "real_ltv": real_ltv,
                "accrued_interest": accrued_interest,
            }
        )

    results = pd.DataFrame(rows).set_index("date")
    results["net_worth"] = results["btc"] * results["price"] - results["total_debt"]
    results["net_btc"] = results["btc"] - (results["total_debt"] / results["price"])
    return results, rebalancing_log


def build_summary(
    results: pd.DataFrame,
    rebalancing_log: list[dict],
    portfolio: Portfolio,
    config: SimulationConfig,
    initial_btc: float,
) -> dict:
    liquidated = any(entry["action"] == "Liquidation" for entry in rebalancing_log)
    last = results.iloc[-1]
    first = results.iloc[0]

    total_btc = float(last["btc"])
    total_debt = float(last["total_debt"])
    end_price = float(last["price"])
    start_price = float(first["price"])
    net_btc = float(last["net_btc"])
    initial_net_btc = float(first["net_btc"])
    total_value = total_btc * end_price
    start_value = initial_btc * start_price
    net_value = float(last["net_worth"])
    max_ltv = float(results["real_ltv"].max())

    ltv_buffer = (config.liquidation_ltv / 100 - max_ltv) / (config.liquidation_ltv / 100)
    if liquidated:
        liquidation_risk = "High"
    elif ltv_buffer < 0.20:
        liquidation_risk = "Medium"
    else:
        liquidation_risk = "Low"

    if abs(total_debt) < 1e-6:
        debt_coverage_ratio = None
    else:
        debt_coverage_ratio = (portfolio.income_per_year + portfolio.other_assets) / total_debt

    return {
        "total_btc": _json_float(total_btc),
        "net_btc": _json_float(net_btc),
        "total_debt": _json_float(total_debt),
        "total_interest": _json_float(last["total_interest"]),
        "total_value": _json_float(total_value),
        "net_value": _json_float(net_value),
        "btc_delta": _json_float(total_btc - initial_btc),
        "net_btc_delta": _json_float(net_btc - initial_net_btc),
        "net_value_delta": _json_float(net_value - start_value),
        "max_ltv": _json_float(max_ltv),
        "liquidation_risk": liquidation_risk,
        "debt_coverage_ratio": _json_float(debt_coverage_ratio) if debt_coverage_ratio is not None else None,
    }


def simulate(request, current_btc_price: float) -> dict:
    price_df = build_price_frame(current_btc_price=current_btc_price, config=request.simulation)
    btc_from_loans = sum(loan.btc_bought for loan in request.portfolio.loans)
    initial_btc = request.portfolio.btc_owned + btc_from_loans
    results, rebalancing_log = run_simulation(
        strategy=request.strategy,
        portfolio=request.portfolio,
        config=request.simulation,
        current_btc=initial_btc,
        price_df=price_df,
        reference_value=get_btc_ath(),
        loans=request.portfolio.loans,
    )

    series = []
    for timestamp, row in results.iterrows():
        series.append(
            {
                "date": timestamp.date().isoformat(),
                "price": _json_float(row["price"]),
                "btc": _json_float(row["btc"]),
                "total_debt": _json_float(row["total_debt"]),
                "total_interest": _json_float(row["total_interest"]),
                "ltv": _json_float(row["ltv"]),
                "real_ltv": _json_float(row["real_ltv"]),
                "net_worth": _json_float(row["net_worth"]),
                "net_btc": _json_float(row["net_btc"]),
            }
        )

    return {
        "series": series,
        "rebalancing_log": rebalancing_log,
        "summary": build_summary(results, rebalancing_log, request.portfolio, request.simulation, initial_btc),
    }


def optimize_strategy(request, current_btc_price: float) -> dict:
    price_df = build_price_frame(current_btc_price=current_btc_price, config=request.simulation)
    btc_from_loans = sum(loan.btc_bought for loan in request.portfolio.loans)
    initial_btc = request.portfolio.btc_owned + btc_from_loans
    reference_value = get_btc_ath()
    best: dict | None = None

    def objective(params: np.ndarray) -> float:
        nonlocal best
        target_ltv, buy_threshold = params
        candidate = StrategyConfig(
            ltv=round(float(target_ltv)),
            ltv_relative_to_ath=False,
            enable_buy=True,
            rebalance_buy=max(0.0, round(float(buy_threshold))),
            rebalance_buy_factor=100,
            enable_sell=False,
            rebalance_sell=10,
            rebalance_sell_factor=100,
        )

        results, rebalancing_log = run_simulation(
            strategy=candidate,
            portfolio=request.portfolio,
            config=request.simulation,
            current_btc=initial_btc,
            price_df=price_df.copy(),
            reference_value=reference_value,
            loans=request.portfolio.loans,
        )
        initial_net_btc = float(results["net_btc"].iloc[0])
        net_btc_delta = float(results["net_btc"].iloc[-1] - initial_net_btc)
        if any(entry["action"] == "Liquidation" for entry in rebalancing_log):
            net_btc_delta = 0.0

        if best is None or net_btc_delta > best["net_btc_delta"]:
            best = {"strategy": candidate, "net_btc_delta": net_btc_delta}

        return -net_btc_delta

    differential_evolution(
        objective,
        bounds=[(5, 80), (0, 30)],
        strategy="best1bin",
        maxiter=12,
        popsize=8,
        tol=0.01,
        seed=42,
        polish=False,
    )

    if best is None:
        best = {"strategy": request.strategy, "net_btc_delta": 0.0}

    return {
        "strategy": best["strategy"].model_dump(),
        "net_btc_delta": _json_float(best["net_btc_delta"]),
    }
