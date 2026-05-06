import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import Plot from "react-plotly.js";

import {
  defaultPortfolio,
  defaultPresets,
  defaultSimulation,
  emptyLoan,
  presetDescriptions,
} from "./defaults";
import { formatCurrency, formatNumber, formatPercent, numberValue } from "./format";
import { buildExport, parseImportJson } from "./import-export";
import { calculatePortfolioTotals } from "./portfolio";
import type {
  ComparisonView,
  Currency,
  ExportData,
  Loan,
  OptimizationResponse,
  Portfolio,
  RebalancingEntry,
  RebalancingInterval,
  SeriesPoint,
  SimulationConfig,
  SimulationMode,
  SimulationResponse,
  StrategyConfig,
  StrategyPresets,
  Summary,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

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
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    if (typeof window === "undefined") return "dark";
    const stored = window.localStorage.getItem("theme");
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  });
  const importRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem("theme", theme);
  }, [theme]);

  const presetNames = Object.keys(strategyPresets);
  const currencySymbol = portfolio.currency === "USD" ? "$" : "EUR ";

  const totals = useMemo(() => calculatePortfolioTotals(portfolio), [portfolio]);

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
        line: { color: "#f7931a" },
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

  function exportSnapshot(): ExportData {
    return buildExport({
      portfolio,
      strategyPresets,
      defaultStrategy,
      simulation,
      selectedSimStrategy,
    });
  }

  function downloadExport() {
    const blob = new Blob([JSON.stringify(exportSnapshot(), null, 2)], {
      type: "application/json",
    });
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
      const parsed = parseImportJson(await file.text());
      setPortfolio(parsed.portfolio);
      setStrategyPresets(parsed.strategyPresets);
      setDefaultStrategy(parsed.defaultStrategy);
      setSimulation(parsed.simulation);
      setSelectedSimStrategy(parsed.selectedSimStrategy);
      selectPreset(parsed.defaultStrategy);
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
        <button
          type="button"
          className="theme-toggle"
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
        >
          <span className="theme-icon" aria-hidden="true" />
          {theme === "dark" ? "Light Mode" : "Dark Mode"}
        </button>
      </header>

      <ToastStack
        toasts={[
          ...(error ? [{ id: "error", kind: "error" as const, message: error }] : []),
          ...(status ? [{ id: "status", kind: "status" as const, message: status }] : []),
        ]}
        onDismiss={(id) => {
          if (id === "error") setError(null);
          if (id === "status") setStatus(null);
        }}
      />

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
              <div className="row-span grid three">
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
                        onChange={(event) =>
                          setStrategy({ ...strategy, rebalance_sell_factor: numberValue(event) })
                        }
                      />
                    </label>
                  </>
                )}
              </div>
              <div className="row-span grid three">
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
                        onChange={(event) =>
                          setStrategy({ ...strategy, rebalance_buy_factor: numberValue(event) })
                        }
                      />
                    </label>
                  </>
                )}
              </div>
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
                    layout={chartLayout("LTV & BTC Price with Rebalancing Events", comparisonView, currencySymbol, theme)}
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
                    layout={chartLayout(`Strategy Comparison - ${comparisonView}`, comparisonView, currencySymbol, theme)}
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

        <section className="app-section" aria-labelledby="disclaimers-section-title">
          <header className="section-header">
            <h2 id="disclaimers-section-title">Disclaimers & Assumptions</h2>
            <p>Assumptions and limitations for interpreting the simulation output.</p>
          </header>
          <div className="section-body">
            <div className="disclaimer-grid">
              <article className="disclaimer">
                <h4>Historical Data</h4>
                <p>Past price performance is no guarantee for future returns.</p>
              </article>
              <article className="disclaimer">
                <h4>Third-Party Risk</h4>
                <p>Borrowing introduces counterparty risk - the lender can fail or change terms.</p>
              </article>
              <article className="disclaimer">
                <h4>Simplified Model</h4>
                <p>Taxes, fees, spreads and edge conditions are excluded from the simulation.</p>
              </article>
              <article className="disclaimer">
                <h4>Not Financial Advice</h4>
                <p>This tool is for educational purposes only and does not constitute financial advice.</p>
              </article>
            </div>
          </div>
        </section>

        <footer className="app-footer">
          <p className="footer-tip">
            <span aria-hidden="true">🧡</span> If you like this tool, you can support it with a{" "}
            <span aria-hidden="true">⚡</span> Lightning tip:{" "}
            <a href="https://strike.me/nodeowl21" target="_blank" rel="noopener noreferrer">
              <strong>strike.me/nodeowl21</strong>
            </a>
          </p>
          <p className="footer-meta">
            © {new Date().getFullYear()}{" "}
            <a href="mailto:nodeowl21@proton.me">nodeowl21</a> &middot;{" "}
            <a
              href="https://github.com/nodeowl21/bitcoin_loan_planner"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>{" "}
            &middot; Open Source &middot; No Data Collected &middot; Bitcoin only
          </p>
        </footer>
      </div>
    </main>
  );
}

function chartLayout(
  title: string,
  view: ComparisonView | string,
  currencySymbol: string,
  theme: "dark" | "light",
) {
  const isComparison = view === "Net Worth" || view === "LTV" || view === "Net BTC Delta (%)";
  const isDark = theme === "dark";
  const gridColor = isDark ? "rgba(139, 148, 158, 0.18)" : "rgba(31, 35, 40, 0.08)";
  const axisColor = isDark ? "#8b949e" : "#57606a";
  const textColor = isDark ? "#e6edf3" : "#1f2328";
  return {
    autosize: true,
    title: { text: title, font: { color: textColor, size: 14 } },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: textColor, family: "Inter, system-ui, sans-serif" },
    colorway: ["#58a6ff", "#3fb950", "#d2a8ff", "#f778ba", "#56d4dd", "#a371f7", "#ffa657"],
    margin: { l: 60, r: 60, t: 60, b: 120 },
    xaxis: {
      title: { text: "Date", font: { color: axisColor } },
      gridcolor: gridColor,
      zerolinecolor: gridColor,
      tickfont: { color: axisColor },
    },
    yaxis: {
      title: { text: isComparison ? view : "LTV", font: { color: axisColor } },
      tickformat: view === "LTV" || view === "Net BTC Delta (%)" ? ".0%" : undefined,
      gridcolor: gridColor,
      zerolinecolor: gridColor,
      tickfont: { color: axisColor },
    },
    yaxis2: {
      title: { text: `BTC Price (${currencySymbol})`, font: { color: axisColor } },
      overlaying: "y",
      side: "right",
      gridcolor: gridColor,
      tickfont: { color: axisColor },
    },
    legend: {
      orientation: "h",
      yanchor: "top",
      y: -0.25,
      xanchor: "center",
      x: 0.5,
      font: { color: textColor },
      bgcolor: "rgba(0,0,0,0)",
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

type ToastKind = "status" | "error";

type Toast = {
  id: string;
  kind: ToastKind;
  message: string;
};

function ToastStack({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: string) => void }) {
  return (
    <div className="toast-stack" aria-live="polite">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
      ))}
    </div>
  );
}

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: string) => void }) {
  useEffect(() => {
    const timeout = toast.kind === "error" ? 6000 : 3000;
    const handle = window.setTimeout(() => onDismiss(toast.id), timeout);
    return () => window.clearTimeout(handle);
  }, [toast.id, toast.kind, toast.message, onDismiss]);

  return (
    <div className={`toast ${toast.kind}`} role={toast.kind === "error" ? "alert" : "status"}>
      <span>{toast.message}</span>
      <button type="button" className="toast-dismiss" aria-label="Dismiss" onClick={() => onDismiss(toast.id)}>
        ×
      </button>
    </div>
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
