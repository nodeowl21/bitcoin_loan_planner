import type { Loan, Portfolio, SimulationConfig, StrategyPresets } from "./types";

export const todayIso = (): string => new Date().toISOString().slice(0, 10);

export const defaultPortfolio: Portfolio = {
  btc_owned: 1,
  currency: "USD",
  btc_price: 100000,
  income_per_year: 0,
  btc_saving_rate_percent: 0,
  other_assets: 0,
  loans: [],
};

export const defaultPresets: StrategyPresets = {
  Custom: {
    ltv: 20,
    ltv_relative_to_ath: false,
    enable_buy: true,
    rebalance_buy: 10,
    rebalance_buy_factor: 100,
    enable_sell: true,
    rebalance_sell: 10,
    rebalance_sell_factor: 100,
  },
  "Defensive HODL": {
    ltv: 10,
    ltv_relative_to_ath: false,
    enable_buy: false,
    rebalance_buy: 10,
    rebalance_buy_factor: 100,
    enable_sell: false,
    rebalance_sell: 10,
    rebalance_sell_factor: 100,
  },
  "Balanced Rebalancer": {
    ltv: 20,
    ltv_relative_to_ath: false,
    enable_buy: true,
    rebalance_buy: 10,
    rebalance_buy_factor: 100,
    enable_sell: true,
    rebalance_sell: 10,
    rebalance_sell_factor: 100,
  },
  "Aggressive Stacker": {
    ltv: 35,
    ltv_relative_to_ath: false,
    enable_buy: true,
    rebalance_buy: 5,
    rebalance_buy_factor: 100,
    enable_sell: false,
    rebalance_sell: 10,
    rebalance_sell_factor: 100,
  },
  "Crash Resilient": {
    ltv: 15,
    ltv_relative_to_ath: false,
    enable_buy: false,
    rebalance_buy: 10,
    rebalance_buy_factor: 100,
    enable_sell: true,
    rebalance_sell: 10,
    rebalance_sell_factor: 100,
  },
};

export const presetDescriptions: Record<string, string> = {
  "Defensive HODL": "Minimal risk, no rebalancing. Loan is taken once and held. Ideal for conservative holders.",
  "Balanced Rebalancer": "Moderate LTV, active buy & sell rebalancing. Grows BTC stack with balanced risk.",
  "Aggressive Stacker": "High LTV with aggressive buy-ins and active rebalancing. Maximum exposure to upside.",
  "Crash Resilient":
    "Start with low leverage. Sell if LTV drifts too high. Designed to survive downturns by staying conservative and reducing risk.",
};

export const defaultSimulation: SimulationConfig = {
  sim_mode: "Historical",
  sim_years: 5,
  exp_return: 50,
  volatility: 5,
  interval: "Weekly",
  interest: 12.5,
  liquidation_ltv: 100,
  enable_btc_saving: true,
};

export const emptyLoan = (): Loan => ({
  id: crypto.randomUUID(),
  platform: "",
  amount: 0,
  interest: 5,
  start_date: todayIso(),
  term_months: null,
  liquidation_ltv: 100,
  btc_bought: 0,
});
