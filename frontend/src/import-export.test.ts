import { describe, expect, it } from "vitest";

import { defaultPortfolio, defaultPresets, defaultSimulation } from "./defaults";
import { buildExport, parseImport, parseImportJson } from "./import-export";
import type { Portfolio } from "./types";

const samplePortfolio: Portfolio = {
  btc_owned: 2.5,
  currency: "EUR",
  btc_price: 90_000,
  income_per_year: 50_000,
  btc_saving_rate_percent: 25,
  other_assets: 10_000,
  loans: [
    {
      id: "abc",
      platform: "Foo",
      amount: 1_000,
      interest: 5,
      start_date: "2024-05-01",
      term_months: 12,
      liquidation_ltv: 80,
      btc_bought: 0.1,
    },
  ],
};

describe("buildExport", () => {
  it("separates loans from the portfolio body and bundles all settings", () => {
    const exported = buildExport({
      portfolio: samplePortfolio,
      strategyPresets: defaultPresets,
      defaultStrategy: "Custom",
      simulation: defaultSimulation,
      selectedSimStrategy: "Aggressive Stacker",
    });

    expect(exported.loans).toHaveLength(1);
    expect("loans" in exported.portfolio).toBe(false);
    expect(exported.portfolio.btc_owned).toBe(2.5);
    expect(exported.strategies.default).toBe("Custom");
    expect(exported.simulation.selected_sim_strategy).toBe("Aggressive Stacker");
    expect(exported.simulation.sim_mode).toBe(defaultSimulation.sim_mode);
  });
});

describe("parseImport", () => {
  it("rejects non-object payloads", () => {
    expect(() => parseImport(null)).toThrow();
    expect(() => parseImport([])).toThrow();
    expect(() => parseImport("not an object")).toThrow();
  });

  it("falls back to defaults when fields are missing", () => {
    const parsed = parseImport({});
    expect(parsed.portfolio).toEqual(defaultPortfolio);
    expect(parsed.strategyPresets).toEqual(defaultPresets);
    expect(parsed.defaultStrategy).toBe("Custom");
    expect(parsed.selectedSimStrategy).toBe("Custom");
    expect(parsed.simulation).toEqual(defaultSimulation);
  });

  it("merges partial portfolios with the defaults", () => {
    const parsed = parseImport({
      portfolio: { btc_owned: 5, currency: "EUR" },
    });
    expect(parsed.portfolio.btc_owned).toBe(5);
    expect(parsed.portfolio.currency).toBe("EUR");
    expect(parsed.portfolio.btc_price).toBe(defaultPortfolio.btc_price);
    expect(parsed.portfolio.loans).toEqual([]);
  });

  it("preserves loans from the payload", () => {
    const parsed = parseImport({ loans: samplePortfolio.loans });
    expect(parsed.portfolio.loans).toEqual(samplePortfolio.loans);
  });

  it("uses the strategies.default as fallback for selectedSimStrategy", () => {
    const parsed = parseImport({
      strategies: { presets: defaultPresets, default: "Crash Resilient" },
    });
    expect(parsed.defaultStrategy).toBe("Crash Resilient");
    expect(parsed.selectedSimStrategy).toBe("Crash Resilient");
  });

  it("respects an explicit selected_sim_strategy", () => {
    const parsed = parseImport({
      strategies: { presets: defaultPresets, default: "Custom" },
      simulation: { ...defaultSimulation, selected_sim_strategy: "Aggressive Stacker" },
    });
    expect(parsed.selectedSimStrategy).toBe("Aggressive Stacker");
  });

  it("buildExport then parseImport round-trips the snapshot", () => {
    const exported = buildExport({
      portfolio: samplePortfolio,
      strategyPresets: defaultPresets,
      defaultStrategy: "Custom",
      simulation: defaultSimulation,
      selectedSimStrategy: "Custom",
    });

    const parsed = parseImport(exported);

    expect(parsed.portfolio.btc_owned).toBe(samplePortfolio.btc_owned);
    expect(parsed.portfolio.currency).toBe(samplePortfolio.currency);
    expect(parsed.portfolio.loans).toEqual(samplePortfolio.loans);
    expect(parsed.strategyPresets).toEqual(defaultPresets);
    expect(parsed.simulation.sim_mode).toBe(defaultSimulation.sim_mode);
  });
});

describe("parseImportJson", () => {
  it("parses a JSON string and produces an import payload", () => {
    const json = JSON.stringify(
      buildExport({
        portfolio: samplePortfolio,
        strategyPresets: defaultPresets,
        defaultStrategy: "Custom",
        simulation: defaultSimulation,
        selectedSimStrategy: "Custom",
      }),
    );

    const parsed = parseImportJson(json);
    expect(parsed.portfolio.btc_owned).toBe(samplePortfolio.btc_owned);
    expect(parsed.portfolio.loans).toHaveLength(1);
  });

  it("propagates JSON parse errors", () => {
    expect(() => parseImportJson("{not json")).toThrow();
  });
});
