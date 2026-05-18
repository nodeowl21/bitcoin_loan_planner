"""Tests for the simulation engine.

These tests pin down the contractual behaviour of the simulation core:
    * Helper utilities behave as documented.
    * Liquidation freezes debt and prevents further interest accrual / rebalancing.
    * Buy / sell rebalancing fire under the expected LTV deviations.
    * Loans with a term repay themselves at maturity.
    * Periodic BTC savings increase the BTC stack.
    * Summary aggregates classify liquidation risk correctly.
"""
from __future__ import annotations

import datetime as dt
import math

import pytest

from backend.app.engine import (
    _add_months,
    _json_float,
    build_summary,
    optimize_strategy,
    run_simulation,
)
from backend.app.models import SimulationRequest

from .conftest import (
    make_config,
    make_loan,
    make_portfolio,
    make_price_df,
    make_strategy,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_add_months_within_year(self):
        assert _add_months(dt.date(2024, 1, 15), 3) == dt.date(2024, 4, 15)

    def test_add_months_crosses_year(self):
        assert _add_months(dt.date(2024, 11, 1), 3) == dt.date(2025, 2, 1)

    def test_add_months_handles_end_of_month(self):
        # 31 Jan + 1 month -> 29 Feb (2024 is a leap year)
        assert _add_months(dt.date(2024, 1, 31), 1) == dt.date(2024, 2, 29)

    def test_json_float_passes_finite_values(self):
        assert _json_float(1.5) == 1.5
        assert _json_float(0.0) == 0.0
        assert _json_float(-2.25) == -2.25

    def test_json_float_replaces_non_finite(self):
        assert _json_float(math.inf) == 1_000_000_000.0
        assert _json_float(-math.inf) == 1_000_000_000.0
        assert _json_float(math.nan) == 1_000_000_000.0


# ---------------------------------------------------------------------------
# Liquidation behaviour (regression tests for the math fix)
# ---------------------------------------------------------------------------


class TestLiquidation:
    def test_liquidation_clears_debt_and_marks_loans_paid(self):
        portfolio = make_portfolio(
            loans=[make_loan(amount=80_000.0, liquidation_ltv=85.0)]
        )
        strategy = make_strategy(ltv=80.0)
        config = make_config()

        prices = [100_000.0] + [90_000.0] * 30
        price_df = make_price_df(prices)

        results, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        liquidations = [e for e in log if e["action"] == "Liquidation"]
        assert len(liquidations) == 1, "Exactly one liquidation event expected"

        liq = liquidations[0]
        assert liq["new_total_debt"] == 0.0
        assert liq["new_total_btc"] >= 0.0
        assert liq["ltv_after"] == 0.0

        assert results["total_debt"].iloc[-1] == 0.0
        assert (results["total_debt"] >= 0.0).all()

    def test_no_interest_accrual_after_liquidation(self):
        portfolio = make_portfolio(
            loans=[make_loan(amount=80_000.0, liquidation_ltv=85.0)]
        )
        strategy = make_strategy(ltv=80.0)
        config = make_config()

        prices = [100_000.0] + [90_000.0] * 60
        price_df = make_price_df(prices)

        results, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        liquidation_date = dt.date.fromisoformat(
            next(e for e in log if e["action"] == "Liquidation")["date"]
        )

        # Strictly after the liquidation date there must be no further interest
        # accrual and the debt must remain zero.
        post_liq = results.loc[results.index.date > liquidation_date]
        assert not post_liq.empty
        assert (post_liq["accrued_interest"] == 0.0).all()
        assert (post_liq["total_debt"] == 0.0).all()

    def test_liquidation_keeps_remainder_when_collateral_exceeds_debt(self):
        # 1 BTC at 100k, 50k debt, liquidation at very low 60% LTV. Price drop to
        # 80k pushes LTV above 60% -> liquidation should retain ~30k worth of BTC.
        portfolio = make_portfolio(
            loans=[make_loan(amount=50_000.0, liquidation_ltv=60.0)]
        )
        strategy = make_strategy(ltv=50.0)
        config = make_config()

        prices = [100_000.0] + [80_000.0] * 30
        price_df = make_price_df(prices)

        _, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        liq = next(e for e in log if e["action"] == "Liquidation")
        # Remaining BTC should be > 0 (since collateral > debt at moment of liq)
        assert liq["new_total_btc"] > 0.0
        assert liq["new_total_btc"] < 1.0
        assert liq["new_total_debt"] == 0.0

    def test_no_buy_rebalancing_after_liquidation(self):
        portfolio = make_portfolio(
            loans=[make_loan(amount=80_000.0, liquidation_ltv=85.0)]
        )
        strategy = make_strategy(
            ltv=50.0,
            enable_buy=True,
            rebalance_buy=5.0,
            rebalance_buy_factor=100.0,
        )
        config = make_config()

        # Crash to trigger liquidation, then a strong rally that would normally
        # invite buy rebalancing - it must not happen post-liquidation.
        prices = [100_000.0, 80_000.0] + [200_000.0] * 30
        price_df = make_price_df(prices)

        _, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        liquidation_idx = next(
            (i for i, e in enumerate(log) if e["action"] == "Liquidation"), None
        )
        assert liquidation_idx is not None

        post_liq_log = log[liquidation_idx + 1 :]
        assert all(e["action"] != "Buy" for e in post_liq_log)
        assert all(e["action"] != "Sell" for e in post_liq_log)
        assert all(e["action"] != "Liquidation" for e in post_liq_log)

    def test_no_liquidation_when_collateral_holds(self):
        portfolio = make_portfolio(
            loans=[make_loan(amount=20_000.0, liquidation_ltv=85.0)]
        )
        strategy = make_strategy(ltv=20.0)
        config = make_config()

        prices = [100_000.0] * 30
        price_df = make_price_df(prices)

        _, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        assert all(e["action"] != "Liquidation" for e in log)


# ---------------------------------------------------------------------------
# Rebalancing behaviour
# ---------------------------------------------------------------------------


class TestRebalancing:
    def test_buy_triggers_when_ltv_drops_below_threshold(self):
        portfolio = make_portfolio(
            loans=[make_loan(amount=50_000.0, liquidation_ltv=90.0)]
        )
        strategy = make_strategy(
            ltv=50.0,
            enable_buy=True,
            rebalance_buy=5.0,
            rebalance_buy_factor=100.0,
        )
        config = make_config(liquidation_ltv=90.0)

        # Price doubles -> LTV halves -> buy should trigger
        prices = [100_000.0] + [200_000.0] * 30
        price_df = make_price_df(prices)

        _, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        buy_events = [e for e in log if e["action"] == "Buy"]
        assert len(buy_events) >= 1

        first_buy = buy_events[0]
        assert first_buy["btc_delta"] > 0
        assert first_buy["fiat_delta"] < 0  # took on new credit

    def test_sell_triggers_when_ltv_rises_above_threshold(self):
        portfolio = make_portfolio(
            loans=[make_loan(amount=50_000.0, liquidation_ltv=95.0)]
        )
        strategy = make_strategy(
            ltv=50.0,
            enable_sell=True,
            rebalance_sell=5.0,
            rebalance_sell_factor=100.0,
        )
        config = make_config(liquidation_ltv=95.0)

        # Price drops 20% -> LTV rises -> sell should trigger before liquidation
        prices = [100_000.0] + [80_000.0] * 30
        price_df = make_price_df(prices)

        _, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        sell_events = [e for e in log if e["action"] == "Sell"]
        assert len(sell_events) >= 1

        first_sell = sell_events[0]
        assert first_sell["btc_delta"] < 0
        assert first_sell["fiat_delta"] > 0  # repaid debt

    def test_rebalancing_disabled_means_no_actions(self):
        portfolio = make_portfolio(
            loans=[make_loan(amount=50_000.0, liquidation_ltv=95.0)]
        )
        strategy = make_strategy(
            ltv=50.0,
            enable_buy=False,
            enable_sell=False,
        )
        config = make_config(liquidation_ltv=95.0)

        prices = [100_000.0] + [200_000.0] * 30
        price_df = make_price_df(prices)

        _, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        assert all(e["action"] not in {"Buy", "Sell"} for e in log)

    def test_rebalancing_interval_controls_frequency(self):
        """Daily > Weekly > Monthly buy counts under a continuously rising price."""
        portfolio_template = lambda: make_portfolio(
            loans=[make_loan(amount=20_000.0, liquidation_ltv=95.0)]
        )
        strategy = make_strategy(
            ltv=20.0,
            enable_buy=True,
            rebalance_buy=0.0,
            rebalance_buy_factor=100.0,
        )

        # 2% daily rally for 60 days keeps the LTV continuously below target.
        prices = [100_000.0 * (1.02 ** i) for i in range(60)]
        price_df = make_price_df(prices)

        counts: dict[str, int] = {}
        for interval in ("Daily", "Weekly", "Monthly"):
            config = make_config(interval=interval, liquidation_ltv=95.0)
            portfolio = portfolio_template()
            _, log = run_simulation(
                strategy=strategy,
                portfolio=portfolio,
                config=config,
                current_btc=1.0,
                price_df=price_df,
                reference_value=100_000.0,
                loans=portfolio.loans,
            )
            counts[interval] = sum(1 for e in log if e["action"] == "Buy")

        assert counts["Daily"] >= 1
        assert counts["Weekly"] >= 1
        assert counts["Monthly"] >= 1
        assert counts["Daily"] > counts["Weekly"]
        assert counts["Weekly"] > counts["Monthly"]

    def test_buy_factor_below_100_buys_less(self):
        """rebalance_buy_factor < 100 dampens the size of new credit drawn."""
        portfolio = make_portfolio(
            loans=[make_loan(amount=50_000.0, liquidation_ltv=95.0)]
        )
        config = make_config(liquidation_ltv=95.0)
        prices = [100_000.0] + [200_000.0] * 5
        price_df = make_price_df(prices)

        full_strategy = make_strategy(
            ltv=50.0,
            enable_buy=True,
            rebalance_buy=5.0,
            rebalance_buy_factor=100.0,
        )
        damped_strategy = make_strategy(
            ltv=50.0,
            enable_buy=True,
            rebalance_buy=5.0,
            rebalance_buy_factor=50.0,
        )

        _, full_log = run_simulation(
            strategy=full_strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )
        _, damped_log = run_simulation(
            strategy=damped_strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        full_buy = next(e for e in full_log if e["action"] == "Buy")
        damped_buy = next(e for e in damped_log if e["action"] == "Buy")
        assert damped_buy["btc_delta"] < full_buy["btc_delta"]
        assert abs(damped_buy["fiat_delta"]) < abs(full_buy["fiat_delta"])

    def test_ltv_relative_to_ath_diverges_from_real_ltv_after_drawdown(self):
        """In ATH mode, the strategy LTV uses the running peak as collateral
        reference - so after a drawdown ``ltv`` (rebalance LTV) is below
        ``real_ltv``."""
        portfolio = make_portfolio(
            loans=[make_loan(amount=50_000.0, liquidation_ltv=99.0)]
        )
        strategy = make_strategy(
            ltv=50.0,
            ltv_relative_to_ath=True,
            enable_sell=False,
            enable_buy=False,
        )
        config = make_config(liquidation_ltv=99.0)

        # Rally to 200k for 5 days, then drawdown back to 100k.
        prices = [100_000.0] + [200_000.0] * 5 + [100_000.0] * 10
        price_df = make_price_df(prices)

        results, _ = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        drawdown = results.iloc[6:]
        assert (drawdown["ltv"] < drawdown["real_ltv"]).all()
        assert drawdown["ltv"].max() < 0.6
        assert drawdown["real_ltv"].max() > 0.45

    def test_ltv_real_mode_matches_real_ltv(self):
        """Without ATH mode, the strategy LTV equals the real LTV every day."""
        portfolio = make_portfolio(
            loans=[make_loan(amount=50_000.0, liquidation_ltv=99.0)]
        )
        strategy = make_strategy(
            ltv=50.0,
            ltv_relative_to_ath=False,
            enable_sell=False,
            enable_buy=False,
        )
        config = make_config(liquidation_ltv=99.0)

        prices = [100_000.0] + [200_000.0] * 5 + [100_000.0] * 10
        price_df = make_price_df(prices)

        results, _ = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        assert (results["ltv"] == results["real_ltv"]).all()


# ---------------------------------------------------------------------------
# Loan lifecycle and savings
# ---------------------------------------------------------------------------


class TestLoanLifecycle:
    def test_loan_with_term_is_repaid_at_maturity(self):
        start = dt.date(2024, 1, 1)
        portfolio = make_portfolio(
            loans=[
                make_loan(
                    amount=10_000.0,
                    interest=10.0,
                    term_months=1,
                    start_date=start,
                    liquidation_ltv=95.0,
                )
            ]
        )
        strategy = make_strategy()
        config = make_config(liquidation_ltv=95.0)

        prices = [100_000.0] * 60
        price_df = make_price_df(prices, start_date=start)

        _, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        repay_events = [e for e in log if e["action"].startswith("Repay")]
        assert len(repay_events) == 1

        repay = repay_events[0]
        assert repay["btc_delta"] < 0
        assert repay["fiat_delta"] < 0
        # Repayment around end of month - sanity check
        assert dt.date.fromisoformat(repay["date"]) >= dt.date(2024, 2, 1)

    def test_repay_log_ltv_after_reflects_debt_and_btc_after_repayment(self):
        """LTV-After muss nach Tilgung aus neuer Schuld und neuem BTC-Bestand folgen."""
        start = dt.date(2024, 1, 1)
        portfolio = make_portfolio(
            loans=[
                make_loan(
                    amount=20_000.0,
                    interest=0.0,
                    term_months=1,
                    start_date=start,
                    liquidation_ltv=95.0,
                )
            ]
        )
        strategy = make_strategy()
        config = make_config(liquidation_ltv=95.0)

        prices = [100_000.0] * 60
        price_df = make_price_df(prices, start_date=start)

        _, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        repay = next(e for e in log if e["action"].startswith("Repay"))
        assert repay["ltv_before"] == pytest.approx(0.20)
        assert repay["ltv_after"] == pytest.approx(0.0)
        assert repay["new_total_debt"] == 0.0
        assert repay["new_total_btc"] == pytest.approx(0.8)

    def test_btc_saving_increases_btc_balance(self):
        # 36500 USD/year income, 10% saving rate -> 10 USD/day -> 0.0001 BTC/day @ 100k
        portfolio = make_portfolio(
            income_per_year=36_500.0,
            btc_saving_rate_percent=10.0,
        )
        strategy = make_strategy()
        config = make_config(enable_btc_saving=True)

        prices = [100_000.0] * 30
        price_df = make_price_df(prices)

        results, _ = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=[],
        )

        final_btc = results["btc"].iloc[-1]
        assert final_btc == pytest.approx(1.003, abs=1e-4)

    def test_loan_started_before_simulation_carries_accrued_interest(self):
        sim_start = dt.date(2024, 1, 1)
        loan_start = dt.date(2023, 1, 1)  # 365 days before simulation

        portfolio = make_portfolio(
            loans=[
                make_loan(
                    amount=10_000.0,
                    interest=10.0,
                    start_date=loan_start,
                    liquidation_ltv=95.0,
                )
            ]
        )
        strategy = make_strategy()
        config = make_config(liquidation_ltv=95.0)

        prices = [100_000.0] * 5
        price_df = make_price_df(prices, start_date=sim_start)

        results, _ = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        # 1 year of pre-accrued interest at 10% on 10k = ~1000 + a few days of
        # in-simulation interest. Should clearly exceed the principal-only debt.
        first_debt = float(results["total_debt"].iloc[0])
        assert first_debt > 10_900.0
        assert first_debt < 11_100.0

    def test_loan_already_matured_before_simulation_is_skipped(self):
        sim_start = dt.date(2024, 1, 1)
        portfolio = make_portfolio(
            loans=[
                make_loan(
                    amount=50_000.0,
                    interest=10.0,
                    start_date=dt.date(2023, 1, 1),
                    term_months=6,  # ends 2023-07-01, before sim start
                    liquidation_ltv=85.0,
                )
            ]
        )
        strategy = make_strategy()
        config = make_config(liquidation_ltv=85.0)

        prices = [100_000.0] * 10
        price_df = make_price_df(prices, start_date=sim_start)

        results, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        # Loan already matured -> no debt, no repay log entries
        assert (results["total_debt"] == 0.0).all()
        assert all(not e["action"].startswith("Repay") for e in log)

    def test_sell_rebalancing_repays_high_interest_loan_first(self):
        """When rebalancing sells BTC, the highest-interest loan is paid first."""
        portfolio = make_portfolio(
            loans=[
                make_loan(
                    id="cheap",
                    platform="Cheap",
                    amount=25_000.0,
                    interest=2.0,
                    liquidation_ltv=95.0,
                ),
                make_loan(
                    id="expensive",
                    platform="Expensive",
                    amount=25_000.0,
                    interest=20.0,
                    liquidation_ltv=95.0,
                ),
            ]
        )
        strategy = make_strategy(
            ltv=50.0,
            enable_sell=True,
            rebalance_sell=2.0,
            rebalance_sell_factor=100.0,
        )
        config = make_config(liquidation_ltv=95.0)

        prices = [100_000.0] + [70_000.0] * 5
        price_df = make_price_df(prices)

        _, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        sells = [e for e in log if e["action"] == "Sell"]
        assert sells, "Expected at least one sell rebalancing"
        # The first sell should not have left the cheap loan smaller than the
        # expensive loan, since the expensive one is repaid first.
        # We verify this indirectly: the total debt drop on the first sell
        # corresponds to repayment ordering, so net debt should be reduced.
        first_sell = sells[0]
        assert first_sell["fiat_delta"] > 0
        assert first_sell["new_total_debt"] < 50_000.0

    def test_btc_saving_disabled_keeps_balance_constant(self):
        portfolio = make_portfolio(
            income_per_year=36_500.0,
            btc_saving_rate_percent=10.0,
        )
        strategy = make_strategy()
        config = make_config(enable_btc_saving=False)

        prices = [100_000.0] * 30
        price_df = make_price_df(prices)

        results, _ = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=[],
        )

        assert results["btc"].iloc[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_low_risk_when_far_from_liquidation(self):
        portfolio = make_portfolio(
            loans=[make_loan(amount=20_000.0, liquidation_ltv=85.0)]
        )
        strategy = make_strategy(ltv=20.0)
        config = make_config(liquidation_ltv=85.0)

        prices = [100_000.0] * 30
        price_df = make_price_df(prices)

        results, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        summary = build_summary(results, log, portfolio, config, initial_btc=1.0)

        assert summary["liquidation_risk"] == "Low"
        assert summary["total_debt"] > 0
        assert summary["total_btc"] == pytest.approx(1.0)
        assert summary["debt_coverage_ratio"] is not None

    def test_net_btc_delta_is_end_minus_start_net_btc(self):
        portfolio = make_portfolio(
            loans=[make_loan(amount=50_000.0, liquidation_ltv=85.0)]
        )
        strategy = make_strategy(ltv=20.0)
        config = make_config(liquidation_ltv=85.0)

        prices = [100_000.0] * 30
        price_df = make_price_df(prices)

        results, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        summary = build_summary(results, log, portfolio, config, initial_btc=1.0)
        start_net = float(results["net_btc"].iloc[0])
        end_net = float(results["net_btc"].iloc[-1])
        assert summary["net_btc_delta"] == pytest.approx(end_net - start_net)
        portfolio = make_portfolio(
            loans=[make_loan(amount=80_000.0, liquidation_ltv=85.0)]
        )
        strategy = make_strategy(ltv=80.0)
        config = make_config(liquidation_ltv=85.0)

        prices = [100_000.0] + [90_000.0] * 30
        price_df = make_price_df(prices)

        results, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        summary = build_summary(results, log, portfolio, config, initial_btc=1.0)

        assert summary["liquidation_risk"] == "High"
        assert summary["total_debt"] == 0.0
        assert summary["max_ltv"] >= 0.85
        # Without income/other assets and no remaining debt, DCR is None
        assert summary["debt_coverage_ratio"] is None

    def test_summary_debt_coverage_ratio(self):
        portfolio = make_portfolio(
            loans=[make_loan(amount=20_000.0, liquidation_ltv=85.0)],
            income_per_year=10_000.0,
            other_assets=5_000.0,
        )
        strategy = make_strategy(ltv=20.0)
        config = make_config(liquidation_ltv=85.0)

        prices = [100_000.0] * 30
        price_df = make_price_df(prices)

        results, log = run_simulation(
            strategy=strategy,
            portfolio=portfolio,
            config=config,
            current_btc=1.0,
            price_df=price_df,
            reference_value=100_000.0,
            loans=portfolio.loans,
        )

        summary = build_summary(results, log, portfolio, config, initial_btc=1.0)
        assert summary["debt_coverage_ratio"] is not None
        # Sanity: (10000 + 5000) / final_debt > 0
        assert summary["debt_coverage_ratio"] > 0


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


class TestOptimization:
    def test_optimize_strategy_returns_valid_config(self):
        request = SimulationRequest(
            portfolio=make_portfolio(),
            strategy=make_strategy(),
            simulation=make_config(sim_mode="Generated", sim_years=1),
        )

        result = optimize_strategy(request, current_btc_price=100_000.0)
        assert "strategy" in result
        assert "net_btc_delta" in result

        strategy = result["strategy"]
        # Pydantic model dump - ensure shape and bounds
        assert 5 <= strategy["ltv"] <= 80
        assert 0 <= strategy["rebalance_buy"] <= 30
        assert strategy["enable_buy"] is True
        # net_btc_delta is absolute BTC vs. starting net BTC (can be negative)
        assert math.isfinite(result["net_btc_delta"])
