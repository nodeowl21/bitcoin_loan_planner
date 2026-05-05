export type Currency = "USD" | "EUR";
export type SimulationMode = "Historical" | "Generated" | "Power-Law";
export type RebalancingInterval = "Daily" | "Weekly" | "Monthly" | "Yearly";
export type ComparisonView = "Net Worth" | "LTV" | "Net BTC Delta (%)";

export type Loan = {
  id: string;
  platform: string;
  amount: number;
  interest: number;
  start_date: string;
  term_months: number | null;
  liquidation_ltv: number;
  btc_bought: number;
};

export type Portfolio = {
  btc_owned: number;
  currency: Currency;
  btc_price: number;
  income_per_year: number;
  btc_saving_rate_percent: number;
  other_assets: number;
  loans: Loan[];
};

export type StrategyConfig = {
  ltv: number;
  ltv_relative_to_ath: boolean;
  enable_buy: boolean;
  rebalance_buy: number;
  rebalance_buy_factor: number;
  enable_sell: boolean;
  rebalance_sell: number;
  rebalance_sell_factor: number;
};

export type StrategyPresets = Record<string, StrategyConfig>;

export type SimulationConfig = {
  sim_mode: SimulationMode;
  sim_years: number;
  exp_return: number;
  volatility: number;
  interval: RebalancingInterval;
  interest: number;
  liquidation_ltv: number;
  enable_btc_saving: boolean;
};

export type SeriesPoint = {
  date: string;
  price: number;
  btc: number;
  total_debt: number;
  total_interest: number;
  ltv: number;
  real_ltv: number;
  net_worth: number;
  net_btc: number;
};

export type RebalancingEntry = {
  date: string;
  action: string;
  ltv_before: number;
  ltv_after: number;
  btc_delta: number;
  price: number;
  fiat_delta: number;
  new_total_btc: number;
  new_total_debt: number;
};

export type Summary = {
  total_btc: number;
  net_btc: number;
  total_debt: number;
  total_interest: number;
  total_value: number;
  net_value: number;
  btc_delta: number;
  net_btc_delta: number;
  net_value_delta: number;
  max_ltv: number;
  liquidation_risk: "Low" | "Medium" | "High";
  debt_coverage_ratio: number | null;
};

export type SimulationResponse = {
  series: SeriesPoint[];
  rebalancing_log: RebalancingEntry[];
  summary: Summary;
};

export type OptimizationResponse = {
  strategy: StrategyConfig;
  net_btc_delta: number;
};

export type ExportData = {
  portfolio: Omit<Portfolio, "loans">;
  loans: Loan[];
  strategies: {
    presets: StrategyPresets;
    default: string;
  };
  simulation: SimulationConfig & {
    selected_sim_strategy: string;
  };
};
