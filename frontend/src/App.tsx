import { useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import Plot from "react-plotly.js";

type Currency = "USD" | "EUR";
type SimulationMode = "Historical" | "Generated" | "Power-Law";
type RebalancingInterval = "Daily" | "Weekly" | "Monthly" | "Yearly";
type ComparisonView = "Net Worth" | "LTV" | "Net BTC Delta (%)";

type Loan = {
  id: string;
  platform: string;
  amount: number;
  interest: number;
  start_date: string;
  term_months: number | null;
  liquidation_ltv: number;
  btc_bought: number;
};

type Portfolio = {
  btc_owned: number;
  currency: Currency;
  btc_price: number;
  income_per_year: number;
  btc_saving_rate_percent: number;
  other_assets: number;
  loans: Loan[];
};

type StrategyConfig = {
  ltv: number;
  ltv_relative_to_ath: boolean;
  enable_buy: boolean;
  rebalance_buy: number;
  rebalance_buy_factor: number;
  enable_sell: boolean;
  rebalance_sell: number;
  rebalance_sell_factor: number;
};

type StrategyPresets = Record<string, StrategyConfig>;

type SimulationConfig = {
  sim_mode: SimulationMode;
  sim_years: number;
  exp_return: number;
  volatility: number;
  interval: RebalancingInterval;
  interest: number;
  liquidation_ltv: number;
  enable_btc_saving: boolean;
};

type SeriesPoint = {
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

type RebalancingEntry = {
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

type Summary = {
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

type SimulationResponse = {
  series: SeriesPoint[];
  rebalancing_log: RebalancingEntry[];
  summary: Summary;
};

type OptimizationResponse = {
  strategy: StrategyConfig;
  net_btc_delta: number;
};

type ExportData = {
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

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const today = new Date().toISOString().slice(0, 10);

const defaultPortfolio: Portfolio = {
  btc_owned: 1,
  currency: "USD",
  btc_price: 100000,
  income_per_year: 0,
  btc_saving_rate_percent: 0,
  other_assets: 0,
  loans: [],
};

const defaultPresets: StrategyPresets = {
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

const presetDescriptions: Record<string, string> = {
  "Defensive HODL": "Minimal risk, no rebalancing. Loan is taken once and held. Ideal for conservative holders.",
  "Balanced Rebalancer": "Moderate LTV, active buy & sell rebalancing. Grows BTC stack with balanced risk.",
  "Aggressive Stacker": "High LTV with aggressive buy-ins and active rebalancing. Maximum exposure to upside.",
  "Crash Resilient":
    "Start with low leverage. Sell if LTV drifts too high. Designed to survive downturns by staying conservative and reducing risk.",
};

const defaultSimulation: SimulationConfig = {
  sim_mode: "Historical",
  sim_years: 5,
  exp_return: 50,
  volatility: 5,
  interval: "Weekly",
  interest: 12.5,
  liquidation_ltv: 100,
  enable_btc_saving: true,
};

const emptyLoan = (): Loan => ({
  id: crypto.randomUUID(),
  platform: "",
  amount: 0,
  interest: 5,
  start_date: today,
  term_months: null,
  liquidation_ltv: 100,
  btc_bought: 0,
});

function numberValue(event: ChangeEvent<HTMLInputElement>): number {
  return Number(event.target.value);
}

function formatCurrency(value: number, currency: Currency, digits = 0): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: digits,
  }).format(value);
}

function formatNumber(value: number, maximumFractionDigits = 2): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(value);
}

function formatPercent(value: number): string {
  return `${formatNumber(value * 100, 2)}%`;
}

function App() {
  const [portfolio, setPortfolio] = useState<Portfolio>(defaultPortfolio);
  const [strategyPresets, setStrategyPresets] = useState<StrategyPresets>(defaultPresets);
  const [defaultStrategy, setDefaultStrategy] = useState("Custom");
  const [selectedPreset, setSelectedPreset] = useState("Custom");
  const [strategy, setStrategy] = useState<StrategyConfig>(defaultPresets.Custom);
  const [strategyName, setStrategyName] = useState("Custom");
  const [simulation, setSimulation] = useState<SimulationConfig>(defaultSimulation);
  const [selectedSimStrategy, setSelectedSimStrategy] = useState("Custom");
  const [selectedCompareStrategies, setSelectedCompareStrategies] = useState<string[]>(["Custom"]);
  const [comparisonView, setComparisonView] = useState<ComparisonView>("Net Worth");
  const [comparisonResults, setComparisonResults] = useState<Record<string, SimulationResponse>>({});
  const [loanDraft, setLoanDraft] = useState<Loan>(emptyLoan());
  const [editingLoanId, setEditingLoanId] = useState<string | null>(null);
  const [result, setResult] = useState<SimulationResponse | null>(null);
  const [optimized, setOptimized] = useState<OptimizationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [importedFileName, setImportedFileName] = useState<string | null>(null);
  const importRef = useRef<HTMLInputElement | null>(null);

  const presetNames = Object.keys(strategyPresets);
  const currencySymbol = portfolio.currency === "USD" ? "$" : "EUR ";

  const totals = useMemo(() => {
    const btcFromLoans = portfolio.loans.reduce((sum, loan) => sum + loan.btc_bought, 0);
    const totalBtc = portfolio.btc_owned + btcFromLoans;
    const totalDebt = portfolio.loans.reduce((sum, loan) => {
      const start = new Date(loan.start_date);
      const now = new Date();
      const end = loan.term_months
        ? new Date(start.getFullYear(), start.getMonth() + loan.term_months, start.getDate())
        : now;
      const effectiveEnd = end < now ? end : now;
      const daysPassed = Math.max(0, (effectiveEnd.getTime() - start.getTime()) / 86_400_000);
      return sum + loan.amount + (loan.amount * loan.interest * daysPassed) / 36500;
    }, 0);
    const portfolioValue = totalBtc * portfolio.btc_price;
    const ltv = portfolioValue > 0 ? totalDebt / portfolioValue : 1_000_000_000;
    const totalAssets = portfolioValue + portfolio.other_assets;
    const netAssets = totalAssets - totalDebt;
    const btcExposure = totalAssets > 0 ? portfolioValue / totalAssets : 0;

    return { totalBtc, totalDebt, portfolioValue, ltv, totalAssets, netAssets, btcExposure };
  }, [portfolio]);

  const ltvChartData = useMemo(() => {
    if (!result) {
      return [];
    }
    const selectedStrategy = strategyPresets[selectedSimStrategy] ?? strategy;
    const ltvField = selectedStrategy.ltv_relative_to_ath ? "ltv" : "real_ltv";

    return [
      {
        x: result.series.map((point) => point.date),
        y: result.series.map((point) => point[ltvField]),
        name: selectedStrategy.ltv_relative_to_ath ? "LTV relative to ATH" : "LTV",
        type: "scatter",
        mode: "lines",
        yaxis: "y",
        hovertemplate: "Date: %{x}<br>LTV: %{y:.2%}<extra></extra>",
      },
      ...(selectedStrategy.ltv_relative_to_ath
        ? [
            {
              x: result.series.map((point) => point.date),
              y: result.series.map((point) => point.real_ltv),
              name: "Real LTV",
              type: "scatter",
              mode: "lines",
              yaxis: "y",
              hovertemplate: "Date: %{x}<br>Real LTV: %{y:.2%}<extra></extra>",
            },
          ]
        : []),
      {
        x: result.series.map((point) => point.date),
        y: result.series.map(() => selectedStrategy.ltv / 100),
        name: "Target LTV",
        type: "scatter",
        mode: "lines",
        yaxis: "y",
        hoverinfo: "skip",
      },
      {
        x: result.series.map((point) => point.date),
        y: result.series.map((point) => point.price),
        name: "BTC Price",
        type: "scatter",
        mode: "lines",
        yaxis: "y2",
        hovertemplate: `Date: %{x}<br>BTC Price: ${currencySymbol}%{y:,.2f}<extra></extra>`,
      },
      {
        x: result.rebalancing_log.map((entry) => entry.date),
        y: result.rebalancing_log.map((entry) => entry.ltv_before),
        text: result.rebalancing_log.map((entry) => `${entry.action}: ${entry.btc_delta.toFixed(6)} BTC`),
        name: "Rebalancing Events",
        type: "scatter",
        mode: "markers",
        yaxis: "y",
        hovertemplate: "%{text}<br>Date: %{x}<br>LTV: %{y:.2%}<extra></extra>",
      },
    ];
  }, [currencySymbol, result, selectedSimStrategy, strategy, strategyPresets]);

  const comparisonChartData = useMemo(() => {
    return selectedCompareStrategies
      .map((name) => {
        const response = comparisonResults[name];
        if (!response) {
          return null;
        }
        const initialBtc = portfolio.btc_owned + portfolio.loans.reduce((sum, loan) => sum + loan.btc_bought, 0);
        const y =
          comparisonView === "Net Worth"
            ? response.series.map((point) => point.net_worth)
            : comparisonView === "LTV"
              ? response.series.map((point) => point.real_ltv)
              : response.series.map((point) => (point.net_btc - initialBtc) / Math.max(initialBtc, 1e-9));
        return {
          x: response.series.map((point) => point.date),
          y,
          name,
          type: "scatter",
          mode: "lines",
          hovertemplate:
            comparisonView === "Net BTC Delta (%)"
              ? "Date: %{x}<br>Net BTC Delta: %{y:.2%}<extra></extra>"
              : "Date: %{x}<br>Value: %{y:,.2f}<extra></extra>",
        };
      })
      .filter(Boolean);
  }, [comparisonResults, comparisonView, portfolio.btc_owned, portfolio.loans, selectedCompareStrategies]);

  const yearlyRows = useMemo(() => {
    return selectedCompareStrategies.map((name) => {
      const response = comparisonResults[name];
      if (!response) {
        return { strategy: name, values: [] as string[] };
      }
      const initialBtc = portfolio.btc_owned + portfolio.loans.reduce((sum, loan) => sum + loan.btc_bought, 0);
      const byYear = new Map<number, SeriesPoint>();
      response.series.forEach((point) => byYear.set(new Date(point.date).getFullYear(), point));
      return {
        strategy: name,
        values: Array.from(byYear.entries()).map(([year, point]) => {
          const delta = (point.net_btc - initialBtc) / Math.max(initialBtc, 1e-9);
          return `${year}: ${formatPercent(delta)}`;
        }),
      };
    });
  }, [comparisonResults, portfolio.btc_owned, portfolio.loans, selectedCompareStrategies]);

  function requestBody(strategyForRun: StrategyConfig) {
    return {
      portfolio,
      strategy: strategyForRun,
      simulation,
    };
  }

  async function postSimulation(strategyForRun: StrategyConfig): Promise<SimulationResponse> {
    const response = await fetch(`${API_BASE_URL}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody(strategyForRun)),
    });
    if (!response.ok) {
      throw new Error((await response.text()) || "Simulation failed");
    }
    return (await response.json()) as SimulationResponse;
  }

  async function fetchLivePrice() {
    setError(null);
    const response = await fetch(`${API_BASE_URL}/btc-price?currency=${portfolio.currency}`);
    if (!response.ok) {
      setError("Could not fetch BTC price");
      return;
    }
    const body = (await response.json()) as { price: number };
    setPortfolio((current) => ({ ...current, btc_price: body.price }));
  }

  async function runSimulation() {
    setIsLoading(true);
    setError(null);
    setStatus(null);
    setOptimized(null);

    try {
      const selectedStrategy = strategyPresets[selectedSimStrategy] ?? strategy;
      const mainResult = await postSimulation(selectedStrategy);
      setResult(mainResult);

      const comparisons: Record<string, SimulationResponse> = {};
      await Promise.all(
        selectedCompareStrategies.map(async (name) => {
          comparisons[name] = await postSimulation(strategyPresets[name] ?? selectedStrategy);
        }),
      );
      setComparisonResults(comparisons);
      setStatus("Simulation complete.");
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  }

  async function runOptimization() {
    setIsOptimizing(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody(strategyPresets[selectedSimStrategy] ?? strategy)),
      });
      if (!response.ok) {
        throw new Error((await response.text()) || "Optimization failed");
      }
      setOptimized((await response.json()) as OptimizationResponse);
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Unknown error");
    } finally {
      setIsOptimizing(false);
    }
  }

  function selectPreset(name: string) {
    const preset = strategyPresets[name];
    setSelectedPreset(name);
    setStrategyName(name);
    setStrategy(preset);
  }

  function saveStrategy() {
    const name = strategyName.trim();
    if (!name) {
      return;
    }
    setStrategyPresets((current) => ({ ...current, [name]: strategy }));
    setSelectedPreset(name);
    if (!selectedCompareStrategies.includes(name)) {
      setSelectedCompareStrategies((current) => [...current, name]);
      if (result) {
        void simulateComparisonStrategy(name, strategy);
      }
    }
  }

  function deleteStrategy() {
    if (strategyName === "Custom") {
      return;
    }
    setStrategyPresets((current) => {
      const next = { ...current };
      delete next[strategyName];
      return next;
    });
    setDefaultStrategy("Custom");
    setSelectedSimStrategy("Custom");
    setSelectedCompareStrategies((current) => current.filter((name) => name !== strategyName));
    selectPreset("Custom");
  }

  function addOptimizedStrategy() {
    if (!optimized) {
      return;
    }
    const name = `Optimized ${new Date().toLocaleTimeString()}`;
    setStrategyPresets((current) => ({ ...current, [name]: optimized.strategy }));
    setSelectedSimStrategy(name);
    setSelectedCompareStrategies((current) => [...new Set([...current, name])]);
    if (result) {
      void simulateComparisonStrategy(name, optimized.strategy);
    }
  }

  function saveLoan() {
    setPortfolio((current) => {
      if (editingLoanId) {
        return {
          ...current,
          loans: current.loans.map((loan) => (loan.id === editingLoanId ? { ...loanDraft, id: editingLoanId } : loan)),
        };
      }
      return { ...current, loans: [...current.loans, loanDraft] };
    });
    setLoanDraft(emptyLoan());
    setEditingLoanId(null);
  }

  function editLoan(loan: Loan) {
    setLoanDraft({ ...loan });
    setEditingLoanId(loan.id);
  }

  function removeLoan(id: string) {
    setPortfolio((current) => ({ ...current, loans: current.loans.filter((loan) => loan.id !== id) }));
    if (editingLoanId === id) {
      setLoanDraft(emptyLoan());
      setEditingLoanId(null);
    }
  }

  function exportData(): ExportData {
    const { loans, ...portfolioWithoutLoans } = portfolio;
    return {
      portfolio: portfolioWithoutLoans,
      loans,
      strategies: { presets: strategyPresets, default: defaultStrategy },
      simulation: { ...simulation, selected_sim_strategy: selectedSimStrategy },
    };
  }

  function downloadExport() {
    const blob = new Blob([JSON.stringify(exportData(), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "loan_planner_backup.json";
    link.click();
    URL.revokeObjectURL(url);
  }

  async function importData(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      const data = JSON.parse(await file.text()) as Partial<ExportData>;
      const importedPortfolio = data.portfolio ? { ...defaultPortfolio, ...data.portfolio } : defaultPortfolio;
      setPortfolio({ ...importedPortfolio, loans: data.loans ?? [] });
      setStrategyPresets(data.strategies?.presets ?? defaultPresets);
      setDefaultStrategy(data.strategies?.default ?? "Custom");
      setSimulation({ ...defaultSimulation, ...data.simulation });
      setSelectedSimStrategy(data.simulation?.selected_sim_strategy ?? data.strategies?.default ?? "Custom");
      selectPreset(data.strategies?.default ?? "Custom");
      setImportedFileName(file.name);
      setStatus("Import complete.");
    } catch {
      setError("Import failed.");
      setImportedFileName(null);
    } finally {
      if (importRef.current) {
        importRef.current.value = "";
      }
    }
  }

  async function simulateComparisonStrategy(name: string, strategyForRun: StrategyConfig) {
    setError(null);
    setStatus(`Simulating ${name} for comparison...`);
    try {
      const comparison = await postSimulation(strategyForRun);
      setComparisonResults((current) => ({ ...current, [name]: comparison }));
      setStatus(`${name} comparison updated.`);
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Comparison simulation failed");
    }
  }

  async function toggleCompareStrategy(name: string) {
    if (selectedCompareStrategies.includes(name)) {
      setSelectedCompareStrategies((current) => current.filter((item) => item !== name));
      return;
    }

    setSelectedCompareStrategies((current) => [...current, name]);
    if (result) {
      await simulateComparisonStrategy(name, strategyPresets[name] ?? strategy);
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <h1>Bitcoin Loan Planner</h1>
          <p>
            This is a Bitcoin Loan Planner for simulating credit strategies aimed at accumulating more Bitcoin over time.
          </p>
        </div>
      </header>

      {error && <div className="error-box">{error}</div>}
      {status && <div className="status-box">{status}</div>}

      <div className="content">
        <section className="app-section" aria-labelledby="portfolio-section-title">
          <header className="section-header">
            <h2 id="portfolio-section-title">Portfolio</h2>
            <p>Current holdings, BTC price and base assumptions for the simulation.</p>
          </header>
          <div className="section-body parallel-form">
            <section className="panel input">
              <h3>Settings</h3>
              <div className="grid two">
                <label>
                  Currency
                  <select
                    value={portfolio.currency}
                    onChange={(event) => setPortfolio({ ...portfolio, currency: event.target.value as Currency })}
                  >
                    <option value="EUR">EUR</option>
                    <option value="USD">USD</option>
                  </select>
                </label>
                <label>
                  BTC Price ({portfolio.currency === "USD" ? "$" : "EUR"})
                  <div className="inline-input">
                    <input
                      min="1"
                      step="1000"
                      type="number"
                      value={portfolio.btc_price}
                      onChange={(event) => setPortfolio({ ...portfolio, btc_price: numberValue(event) })}
                    />
                    <button type="button" onClick={fetchLivePrice}>
                      Live
                    </button>
                  </div>
                </label>
                <label>
                  BTC Holdings
                  <input
                    min="0"
                    step="0.000001"
                    type="number"
                    value={portfolio.btc_owned}
                    onChange={(event) => setPortfolio({ ...portfolio, btc_owned: numberValue(event) })}
                  />
                </label>
                <label>
                  Annual Income ({portfolio.currency === "USD" ? "$" : "EUR"})
                  <input
                    min="0"
                    step="1000"
                    type="number"
                    value={portfolio.income_per_year}
                    onChange={(event) => setPortfolio({ ...portfolio, income_per_year: numberValue(event) })}
                  />
                </label>
                <label>
                  BTC Saving Rate (% of Income)
                  <input
                    min="0"
                    max="100"
                    step="0.5"
                    type="number"
                    value={portfolio.btc_saving_rate_percent}
                    onChange={(event) => setPortfolio({ ...portfolio, btc_saving_rate_percent: numberValue(event) })}
                  />
                </label>
                <label>
                  Other Assets ({portfolio.currency === "USD" ? "$" : "EUR"})
                  <input
                    min="0"
                    step="1000"
                    type="number"
                    value={portfolio.other_assets}
                    onChange={(event) => setPortfolio({ ...portfolio, other_assets: numberValue(event) })}
                  />
                </label>
              </div>
            </section>

            <section className="panel readonly">
              <h3>Summary</h3>
              <div className="summary-grid">
                <Metric label="Total BTC" value={`${formatNumber(totals.totalBtc, 6)} BTC`} />
                <Metric label="Total BTC Value" value={formatCurrency(totals.portfolioValue, portfolio.currency, 2)} />
                <Metric label="Total Debt" value={formatCurrency(totals.totalDebt, portfolio.currency, 2)} />
                <Metric label="LTV" value={formatPercent(totals.ltv)} />
                <Metric label="Annual Income" value={formatCurrency(portfolio.income_per_year, portfolio.currency, 2)} />
                <Metric label="Other Assets" value={formatCurrency(portfolio.other_assets, portfolio.currency, 2)} />
                <Metric label="BTC Saving Rate" value={`${formatNumber(portfolio.btc_saving_rate_percent, 1)}%`} />
                <Metric label="Total Value" value={formatCurrency(totals.totalAssets, portfolio.currency, 2)} />
                <Metric label="Net Value" value={formatCurrency(totals.netAssets, portfolio.currency, 2)} />
                <Metric label="BTC Exposure" value={formatPercent(totals.btcExposure)} />
              </div>
            </section>
          </div>
        </section>

        <section className="app-section" aria-labelledby="loans-section-title">
          <header className="section-header">
            <h2 id="loans-section-title">Loans</h2>
            <p>Add, edit and review existing loans with their terms and risk parameters.</p>
          </header>
          <div className="section-body parallel-form">
            <section className="panel input">
              <h3>{editingLoanId ? "Edit Loan" : "Add Loan"}</h3>
              <div className="grid two">
                <label>
                  Platform / Lender
                  <input
                    value={loanDraft.platform}
                    onChange={(event) => setLoanDraft({ ...loanDraft, platform: event.target.value })}
                  />
                </label>
                <label>
                  Loan Amount ({portfolio.currency === "USD" ? "$" : "EUR"})
                  <input
                    min="0"
                    step="1000"
                    type="number"
                    value={loanDraft.amount}
                    onChange={(event) => setLoanDraft({ ...loanDraft, amount: numberValue(event) })}
                  />
                </label>
                <label>
                  Interest Rate (% p.a.)
                  <input
                    min="0"
                    max="50"
                    step="0.1"
                    type="number"
                    value={loanDraft.interest}
                    onChange={(event) => setLoanDraft({ ...loanDraft, interest: numberValue(event) })}
                  />
                </label>
                <label>
                  BTC Bought
                  <input
                    min="0"
                    step="0.000001"
                    type="number"
                    value={loanDraft.btc_bought}
                    onChange={(event) => setLoanDraft({ ...loanDraft, btc_bought: numberValue(event) })}
                  />
                </label>
                <label>
                  Start Date
                  <input
                    type="date"
                    value={loanDraft.start_date}
                    onChange={(event) => setLoanDraft({ ...loanDraft, start_date: event.target.value })}
                  />
                </label>
                <label>
                  Loan Term
                  <select
                    value={loanDraft.term_months === null ? "Unlimited" : "Set duration"}
                    onChange={(event) =>
                      setLoanDraft({
                        ...loanDraft,
                        term_months: event.target.value === "Unlimited" ? null : 12,
                      })
                    }
                  >
                    <option>Unlimited</option>
                    <option>Set duration</option>
                  </select>
                </label>
                {loanDraft.term_months !== null && (
                  <label>
                    Duration (months)
                    <input
                      min="1"
                      max="360"
                      type="number"
                      value={loanDraft.term_months}
                      onChange={(event) => setLoanDraft({ ...loanDraft, term_months: numberValue(event) })}
                    />
                  </label>
                )}
                <label>
                  Liquidation LTV (%)
                  <input
                    min="50"
                    max="100"
                    type="number"
                    value={loanDraft.liquidation_ltv}
                    onChange={(event) => setLoanDraft({ ...loanDraft, liquidation_ltv: numberValue(event) })}
                  />
                </label>
              </div>
              <div className="button-row">
                <button type="button" onClick={saveLoan}>
                  {editingLoanId ? "Update Loan" : "Save Loan"}
                </button>
                {editingLoanId && (
                  <button
                    type="button"
                    onClick={() => {
                      setLoanDraft(emptyLoan());
                      setEditingLoanId(null);
                    }}
                  >
                    Cancel Edit
                  </button>
                )}
              </div>
            </section>

            <section className="panel readonly">
              <h3>Active Loans</h3>
              {portfolio.loans.length === 0 ? (
                <p className="muted">No loans yet.</p>
              ) : (
                <div className="loan-list">
                  {portfolio.loans.map((loan) => (
                    <div className="loan-row" key={loan.id}>
                      <span>
                        <strong>{loan.platform || "Loan"}</strong> - {formatCurrency(loan.amount, portfolio.currency)} at{" "}
                        {loan.interest.toFixed(2)}% p.a. -{" "}
                        {loan.term_months ? `${loan.term_months} months` : `Unlimited since ${loan.start_date}`} - BTC
                        Bought: {loan.btc_bought.toFixed(6)} BTC - Liquidation LTV: {loan.liquidation_ltv.toFixed(0)}%
                      </span>
                      <div className="button-row compact">
                        <button type="button" onClick={() => editLoan(loan)}>
                          Edit
                        </button>
                        <button type="button" onClick={() => removeLoan(loan.id)}>
                          Delete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </div>
        </section>

        <section className="app-section" aria-labelledby="strategies-section-title">
          <header className="section-header">
            <h2 id="strategies-section-title">Strategies</h2>
            <p>Choose, adjust, save and delete strategy presets.</p>
          </header>
          <div className="section-body">
            <section className="panel input">
              <h3>Strategy Configuration</h3>
              <label>
                Choose Preset
                <select value={selectedPreset} onChange={(event) => selectPreset(event.target.value)}>
                  {presetNames.map((name) => (
                    <option key={name}>{name}</option>
                  ))}
                </select>
              </label>
              {presetDescriptions[selectedPreset] && <p className="muted">{presetDescriptions[selectedPreset]}</p>}

              <div className="grid two">
                <label>
                  Target LTV (%)
                <input
                  min="1"
                  max="100"
                  type="number"
                  value={strategy.ltv}
                  onChange={(event) => setStrategy({ ...strategy, ltv: numberValue(event) })}
                />
              </label>
              <label>
                <span className="label-with-info">
                  LTV relative to ATH
                  <InfoTip text="When enabled, the Loan-to-Value (LTV) is calculated relative to Bitcoin's all-time high instead of the current price. This means that rebalancing actions are only triggered by significant market extremes - near global highs or lows - rather than short-term price movements." />
                </span>
                <div className="checkbox-field">
                  <input
                    type="checkbox"
                    checked={strategy.ltv_relative_to_ath}
                    onChange={(event) => setStrategy({ ...strategy, ltv_relative_to_ath: event.target.checked })}
                  />
                </div>
              </label>
              <label>
                Enable Sell-Rebalancing
                <div className="checkbox-field">
                  <input
                    type="checkbox"
                    checked={strategy.enable_sell}
                    onChange={(event) => setStrategy({ ...strategy, enable_sell: event.target.checked })}
                  />
                </div>
              </label>
              {strategy.enable_sell && (
                <>
                  <label>
                    Sell Threshold (%)
                    <input
                      min="1"
                      max="100"
                      type="number"
                      value={strategy.rebalance_sell}
                      onChange={(event) => setStrategy({ ...strategy, rebalance_sell: numberValue(event) })}
                    />
                  </label>
                  <label>
                    Sell Rebalancing Intensity (%)
                    <input
                      min="1"
                      max="100"
                      type="number"
                      value={strategy.rebalance_sell_factor}
                      onChange={(event) => setStrategy({ ...strategy, rebalance_sell_factor: numberValue(event) })}
                    />
                  </label>
                </>
              )}
              <label>
                Enable Buy-Rebalancing
                <div className="checkbox-field">
                  <input
                    type="checkbox"
                    checked={strategy.enable_buy}
                    onChange={(event) => setStrategy({ ...strategy, enable_buy: event.target.checked })}
                  />
                </div>
              </label>
              {strategy.enable_buy && (
                <>
                  <label>
                    Buy Threshold (%)
                    <input
                      min="0"
                      max="100"
                      type="number"
                      value={strategy.rebalance_buy}
                      onChange={(event) => setStrategy({ ...strategy, rebalance_buy: numberValue(event) })}
                    />
                  </label>
                  <label>
                    Buy Rebalancing Intensity (%)
                    <input
                      min="1"
                      max="100"
                      type="number"
                      value={strategy.rebalance_buy_factor}
                      onChange={(event) => setStrategy({ ...strategy, rebalance_buy_factor: numberValue(event) })}
                    />
                  </label>
                </>
              )}
              <label>
                Name
                <input value={strategyName} onChange={(event) => setStrategyName(event.target.value)} />
              </label>
              <label>
                <span className="label-with-info">
                  Default Strategy
                  <InfoTip text="Mark this strategy as the default for future selections." />
                </span>
                <div className="checkbox-field">
                  <input
                    type="checkbox"
                    checked={defaultStrategy === selectedPreset}
                    onChange={(event) => event.target.checked && setDefaultStrategy(selectedPreset)}
                  />
                </div>
              </label>
            </div>
              <div className="button-row">
                <button type="button" onClick={saveStrategy}>
                  Save Strategy
                </button>
                {strategyName !== "Custom" && (
                  <button type="button" onClick={deleteStrategy}>
                    Delete Strategy
                  </button>
                )}
              </div>
            </section>
          </div>
        </section>

        <section className="app-section" aria-labelledby="simulation-section-title">
          <header className="section-header">
            <h2 id="simulation-section-title">Simulation</h2>
            <p>Configure price source, rebalancing cadence, risk limits and run the model.</p>
          </header>
          <div className={optimized ? "section-body parallel-form" : "section-body"}>
            <section className="panel input">
              <h3>Configuration</h3>
              <div className="grid two">
                <label>
                  <span className="label-with-info">
                    Choose Price Source
                    <InfoTip text="Choose between historical, generated, or power-law based prices." />
                  </span>
                  <select
                    value={simulation.sim_mode}
                    onChange={(event) =>
                      setSimulation({ ...simulation, sim_mode: event.target.value as SimulationMode })
                    }
                  >
                    <option>Historical</option>
                    <option>Generated</option>
                    <option>Power-Law</option>
                  </select>
                </label>
                <label>
                  {simulation.sim_mode === "Historical" ? "Historical Timeframe (years)" : "Number of Simulation Years"}
                  <input
                    min="1"
                    max={simulation.sim_mode === "Historical" ? 10 : 20}
                    type="number"
                    value={simulation.sim_years}
                    onChange={(event) => setSimulation({ ...simulation, sim_years: numberValue(event) })}
                  />
                </label>
                {simulation.sim_mode === "Generated" && (
                  <>
                    <label>
                      Expected Annual Return (%)
                      <input
                        min="-100"
                        max="200"
                        type="number"
                        value={simulation.exp_return}
                        onChange={(event) => setSimulation({ ...simulation, exp_return: numberValue(event) })}
                      />
                    </label>
                    <label>
                      Daily Volatility (%)
                      <input
                        min="1"
                        max="100"
                        type="number"
                        value={simulation.volatility}
                        onChange={(event) => setSimulation({ ...simulation, volatility: numberValue(event) })}
                      />
                    </label>
                  </>
                )}
                <label>
                  Rebalancing Interval
                  <select
                    value={simulation.interval}
                    onChange={(event) =>
                      setSimulation({ ...simulation, interval: event.target.value as RebalancingInterval })
                    }
                  >
                    <option>Daily</option>
                    <option>Weekly</option>
                    <option>Monthly</option>
                    <option>Yearly</option>
                  </select>
                </label>
                <label>
                  Loan Interest Rate (% p.a.)
                  <input
                    min="0"
                    max="20"
                    step="0.1"
                    type="number"
                    value={simulation.interest}
                    onChange={(event) => setSimulation({ ...simulation, interest: numberValue(event) })}
                  />
                </label>
                <label>
                  Liquidation LTV (%)
                  <input
                    min="50"
                    max="100"
                    type="number"
                    value={simulation.liquidation_ltv}
                    onChange={(event) => setSimulation({ ...simulation, liquidation_ltv: numberValue(event) })}
                  />
                </label>
                <label>
                  Enable BTC Saving (daily)
                  <div className="checkbox-field">
                    <input
                      type="checkbox"
                      checked={simulation.enable_btc_saving}
                      onChange={(event) => setSimulation({ ...simulation, enable_btc_saving: event.target.checked })}
                    />
                  </div>
                </label>
                <label>
                  Choose strategy for simulation
                  <select value={selectedSimStrategy} onChange={(event) => setSelectedSimStrategy(event.target.value)}>
                    {presetNames.map((name) => (
                      <option key={name}>{name}</option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="button-row">
                <button disabled={isLoading} onClick={runSimulation}>
                  {isLoading ? "Running Simulation..." : "Run Simulation"}
                </button>
                <button disabled={isOptimizing} onClick={runOptimization}>
                  {isOptimizing ? "Optimizing..." : "Run Optimization"}
                </button>
              </div>
            </section>

            {optimized && (
              <section className="panel readonly">
                <h3>Optimized Strategy</h3>
                <p>
                  Net BTC Delta: <strong>{formatPercent(optimized.net_btc_delta)}</strong>
                </p>
                <pre>{JSON.stringify(optimized.strategy, null, 2)}</pre>
                <div className="button-row">
                  <button type="button" onClick={addOptimizedStrategy}>
                    Add Optimized to Presets
                  </button>
                </div>
              </section>
            )}
          </div>
        </section>

        {result && (
          <>
            <section className="app-section" aria-labelledby="results-section-title">
              <header className="section-header">
                <h2 id="results-section-title">Simulation Results</h2>
                <p>Review LTV development, rebalancing actions and the resulting summary metrics.</p>
              </header>
              <div className="section-body">
                <section className="panel readonly">
                  <h3>LTV Development</h3>
                  <Plot
                    data={ltvChartData}
                    layout={chartLayout("LTV & BTC Price with Rebalancing Events", comparisonView, currencySymbol)}
                    config={{ responsive: true, displayModeBar: false }}
                    className="plot"
                    useResizeHandler
                  />
                </section>

                <section className="panel readonly">
                  <h3>Rebalancing Log</h3>
                  <DataTable entries={result.rebalancing_log} currency={portfolio.currency} />
                </section>

                <section className="panel readonly">
                  <h3>Summary</h3>
                  <div className="summary-grid">
                    <Metric label="Total BTC" value={`${formatNumber(result.summary.total_btc, 6)} BTC`} />
                    <Metric label="Net BTC" value={`${formatNumber(result.summary.net_btc, 6)} BTC`} />
                    <Metric
                      label="Total Debt (incl. interest)"
                      value={formatCurrency(result.summary.total_debt, portfolio.currency, 2)}
                    />
                    <Metric label="Liquidation Risk" value={result.summary.liquidation_risk} />
                    <Metric
                      label="Total Value"
                      value={formatCurrency(result.summary.total_value, portfolio.currency, 2)}
                    />
                    <Metric
                      label="Net Value"
                      value={formatCurrency(result.summary.net_value, portfolio.currency, 2)}
                    />
                    <Metric
                      label="Total Interest Paid"
                      value={formatCurrency(result.summary.total_interest, portfolio.currency, 2)}
                    />
                    <Metric
                      label="Debt Coverage Ratio (DCR)"
                      value={
                        result.summary.debt_coverage_ratio === null
                          ? "Infinity"
                          : formatNumber(result.summary.debt_coverage_ratio)
                      }
                      info="Debt Coverage Ratio (DCR) = (Annual Income + Other Assets) / Total Debt. Indicates how easily you could cover your outstanding debt with your non-BTC assets and income. Higher values mean lower risk. A DCR above 1.5 is considered safe, between 1.0 and 1.5 moderate, and below 1.0 risky."
                    />
                  </div>
                </section>
              </div>
            </section>

            <section className="app-section" aria-labelledby="comparison-section-title">
              <header className="section-header">
                <h2 id="comparison-section-title">Strategy Comparison</h2>
                <p>Compare selected strategies across net worth, LTV and net BTC delta.</p>
              </header>
              <div className="section-body">
                <section className="panel input">
                  <h3>Strategies</h3>
                  <div className="checkbox-grid">
                    {presetNames.map((name) => (
                      <label key={name}>
                        {name}
                        <div className="checkbox-field">
                          <input
                            type="checkbox"
                            checked={selectedCompareStrategies.includes(name)}
                            onChange={() => void toggleCompareStrategy(name)}
                          />
                        </div>
                      </label>
                    ))}
                  </div>
                  <div className="button-row">
                    {(["Net Worth", "LTV", "Net BTC Delta (%)"] as ComparisonView[]).map((view) => (
                      <button
                        key={view}
                        type="button"
                        aria-pressed={comparisonView === view}
                        onClick={() => setComparisonView(view)}
                      >
                        {view}
                      </button>
                    ))}
                  </div>
                </section>

                <section className="panel readonly">
                  <h3>{`Comparison - ${comparisonView}`}</h3>
                  <Plot
                    data={comparisonChartData}
                    layout={chartLayout(`Strategy Comparison - ${comparisonView}`, comparisonView, currencySymbol)}
                    config={{ responsive: true, displayModeBar: false }}
                    className="plot compact-plot"
                    useResizeHandler
                  />
                </section>

                <section className="panel readonly">
                  <h3>Total Delta at Year</h3>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Strategy</th>
                          <th>Net BTC Delta (%) by Year</th>
                        </tr>
                      </thead>
                      <tbody>
                        {yearlyRows.map((row) => (
                          <tr key={row.strategy}>
                            <td>{row.strategy}</td>
                            <td>{row.values.join(" | ") || "Run simulation"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              </div>
            </section>
          </>
        )}

        <section className="app-section" aria-labelledby="data-section-title">
          <header className="section-header">
            <h2 id="data-section-title">Data</h2>
            <p>Import existing settings or export the current configuration.</p>
          </header>
          <div className="section-body parallel">
            <section className="panel input">
              <h3>Import</h3>
              <input
                ref={importRef}
                type="file"
                accept="application/json"
                onChange={importData}
                hidden
              />
              <div className="button-row">
                <button type="button" onClick={() => importRef.current?.click()}>
                  Import Settings
                </button>
              </div>
              {importedFileName && (
                <p className="muted">Imported file: {importedFileName}</p>
              )}
            </section>
            <section className="panel input">
              <h3>Export</h3>
              <div className="button-row">
                <button type="button" onClick={downloadExport}>
                  Export Settings
                </button>
              </div>
            </section>
          </div>
        </section>

        <section className="app-section" aria-labelledby="notes-section-title">
          <header className="section-header">
            <h2 id="notes-section-title">Notes</h2>
            <p>Assumptions and limitations for interpreting the simulation output.</p>
          </header>
          <div className="section-body">
            <section className="panel readonly">
              <h3>Disclaimers & Assumptions</h3>
              <ul>
                <li>Historical data is no guarantee for future performance.</li>
                <li>By taking out loans, you introduce third-party risk.</li>
                <li>The simulation excludes taxes, fees, spreads and edge conditions.</li>
                <li>This tool does not constitute financial advice.</li>
              </ul>
            </section>
          </div>
        </section>
      </div>
    </main>
  );
}

function chartLayout(title: string, view: ComparisonView | string, currencySymbol: string) {
  const isComparison = view === "Net Worth" || view === "LTV" || view === "Net BTC Delta (%)";
  return {
    autosize: true,
    title,
    margin: { l: 60, r: 60, t: 60, b: 120 },
    xaxis: {
      title: "Date",
    },
    yaxis: {
      title: isComparison ? view : "LTV",
      tickformat: view === "LTV" || view === "Net BTC Delta (%)" ? ".0%" : undefined,
    },
    yaxis2: {
      title: `BTC Price (${currencySymbol})`,
      overlaying: "y",
      side: "right",
    },
    legend: {
      orientation: "h",
      yanchor: "top",
      y: -0.25,
      xanchor: "center",
      x: 0.5,
    },
  };
}

function Metric({ label, value, info }: { label: string; value: string; info?: string }) {
  return (
    <div className="metric">
      <span>
        {label}
        {info && <InfoTip text={info} />}
      </span>
      <strong>{value}</strong>
    </div>
  );
}

function InfoTip({ text }: { text: string }) {
  return (
    <span className="info-tip" role="img" aria-label={text} title={text}>
      ?
    </span>
  );
}

function DataTable({ entries, currency }: { entries: RebalancingEntry[]; currency: Currency }) {
  if (entries.length === 0) {
    return <p className="muted">No rebalancing events in this simulation.</p>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Action</th>
            <th>BTC Delta</th>
            <th>Price</th>
            <th>New Total BTC</th>
            <th>New Total Debt</th>
            <th>LTV before</th>
            <th>LTV after</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry, index) => (
            <tr key={`${entry.date}-${entry.action}-${index}`}>
              <td>{entry.date}</td>
              <td>{entry.action}</td>
              <td>{formatNumber(entry.btc_delta, 6)} BTC</td>
              <td>{formatCurrency(entry.price, currency, 2)}</td>
              <td>{formatNumber(entry.new_total_btc, 6)} BTC</td>
              <td>{formatCurrency(entry.new_total_debt, currency, 2)}</td>
              <td>{formatPercent(entry.ltv_before)}</td>
              <td>{formatPercent(entry.ltv_after)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
