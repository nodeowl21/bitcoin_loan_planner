import { describe, expect, it } from "vitest";

import { defaultPortfolio, defaultPresets, defaultSimulation, emptyLoan, presetDescriptions, todayIso } from "./defaults";

describe("defaultPortfolio", () => {
  it("starts with one BTC and no loans", () => {
    expect(defaultPortfolio.btc_owned).toBe(1);
    expect(defaultPortfolio.loans).toEqual([]);
  });

  it("uses USD by default", () => {
    expect(defaultPortfolio.currency).toBe("USD");
  });
});

describe("defaultPresets", () => {
  it("contains the named strategies", () => {
    expect(Object.keys(defaultPresets)).toEqual(
      expect.arrayContaining(["Custom", "Defensive HODL", "Balanced Rebalancer", "Aggressive Stacker", "Crash Resilient"]),
    );
  });

  it("Defensive HODL disables both buy and sell rebalancing", () => {
    const preset = defaultPresets["Defensive HODL"];
    expect(preset.enable_buy).toBe(false);
    expect(preset.enable_sell).toBe(false);
  });

  it("Aggressive Stacker has the highest target LTV", () => {
    const ltvs = Object.values(defaultPresets).map((preset) => preset.ltv);
    expect(defaultPresets["Aggressive Stacker"].ltv).toBe(Math.max(...ltvs));
  });

  it("Crash Resilient sells but does not buy", () => {
    const preset = defaultPresets["Crash Resilient"];
    expect(preset.enable_buy).toBe(false);
    expect(preset.enable_sell).toBe(true);
  });

  it("each preset is within the API's accepted bounds", () => {
    for (const preset of Object.values(defaultPresets)) {
      expect(preset.ltv).toBeGreaterThanOrEqual(1);
      expect(preset.ltv).toBeLessThanOrEqual(100);
      expect(preset.rebalance_buy).toBeGreaterThanOrEqual(0);
      expect(preset.rebalance_buy).toBeLessThanOrEqual(100);
      expect(preset.rebalance_sell).toBeGreaterThanOrEqual(0);
      expect(preset.rebalance_sell).toBeLessThanOrEqual(100);
      expect(preset.rebalance_buy_factor).toBeGreaterThanOrEqual(1);
      expect(preset.rebalance_buy_factor).toBeLessThanOrEqual(100);
      expect(preset.rebalance_sell_factor).toBeGreaterThanOrEqual(1);
      expect(preset.rebalance_sell_factor).toBeLessThanOrEqual(100);
    }
  });
});

describe("presetDescriptions", () => {
  it("describes every named preset (Custom is unlabeled)", () => {
    for (const name of Object.keys(defaultPresets)) {
      if (name === "Custom") continue;
      expect(presetDescriptions[name]).toBeDefined();
      expect(presetDescriptions[name].length).toBeGreaterThan(10);
    }
  });
});

describe("defaultSimulation", () => {
  it("uses Historical mode by default", () => {
    expect(defaultSimulation.sim_mode).toBe("Historical");
  });

  it("has BTC saving enabled", () => {
    expect(defaultSimulation.enable_btc_saving).toBe(true);
  });
});

describe("emptyLoan", () => {
  it("starts at zero amount and 5% interest", () => {
    const loan = emptyLoan();
    expect(loan.amount).toBe(0);
    expect(loan.interest).toBe(5);
    expect(loan.btc_bought).toBe(0);
    expect(loan.term_months).toBeNull();
  });

  it("has a unique id per call", () => {
    const a = emptyLoan();
    const b = emptyLoan();
    expect(a.id).not.toBe(b.id);
  });

  it("uses today as start_date", () => {
    expect(emptyLoan().start_date).toBe(todayIso());
  });
});

describe("todayIso", () => {
  it("returns an ISO date string of length 10", () => {
    const today = todayIso();
    expect(today).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });
});
