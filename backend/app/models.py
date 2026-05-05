from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field


Currency = Literal["USD", "EUR"]
SimulationMode = Literal["Historical", "Generated", "Power-Law"]
RebalancingInterval = Literal["Daily", "Weekly", "Monthly", "Yearly"]


class Loan(BaseModel):
    id: str | None = None
    platform: str = ""
    amount: float = Field(default=0.0, ge=0)
    interest: float = Field(default=5.0, ge=0, le=50)
    start_date: dt.date = Field(default_factory=dt.date.today)
    term_months: int | None = Field(default=None, ge=1, le=360)
    liquidation_ltv: float = Field(default=100.0, ge=1, le=100)
    btc_bought: float = Field(default=0.0, ge=0)


class Portfolio(BaseModel):
    btc_owned: float = Field(default=1.0, ge=0)
    currency: Currency = "USD"
    btc_price: float | None = Field(default=None, gt=0)
    income_per_year: float = Field(default=0.0, ge=0)
    btc_saving_rate_percent: float = Field(default=0.0, ge=0, le=100)
    other_assets: float = Field(default=0.0, ge=0)
    loans: list[Loan] = Field(default_factory=list)


class StrategyConfig(BaseModel):
    ltv: float = Field(default=20.0, ge=1, le=100)
    ltv_relative_to_ath: bool = False
    enable_buy: bool = True
    rebalance_buy: float = Field(default=10.0, ge=0, le=100)
    rebalance_buy_factor: float = Field(default=100.0, ge=1, le=100)
    enable_sell: bool = True
    rebalance_sell: float = Field(default=10.0, ge=0, le=100)
    rebalance_sell_factor: float = Field(default=100.0, ge=1, le=100)


class SimulationConfig(BaseModel):
    sim_mode: SimulationMode = "Historical"
    sim_years: int = Field(default=5, ge=1, le=20)
    exp_return: float = Field(default=50.0, ge=-100, le=200)
    volatility: float = Field(default=5.0, ge=0, le=100)
    interval: RebalancingInterval = "Weekly"
    interest: float = Field(default=12.5, ge=0, le=50)
    liquidation_ltv: float = Field(default=100.0, ge=1, le=100)
    enable_btc_saving: bool = True


class SimulationRequest(BaseModel):
    portfolio: Portfolio = Field(default_factory=Portfolio)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)


class Summary(BaseModel):
    total_btc: float
    net_btc: float
    total_debt: float
    total_interest: float
    total_value: float
    net_value: float
    btc_delta: float
    net_btc_delta: float
    net_value_delta: float
    max_ltv: float
    liquidation_risk: str
    debt_coverage_ratio: float | None


class SimulationResponse(BaseModel):
    series: list[dict]
    rebalancing_log: list[dict]
    summary: Summary


class OptimizationResponse(BaseModel):
    strategy: StrategyConfig
    net_btc_delta: float
