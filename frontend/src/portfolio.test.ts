import { describe, expect, it } from "vitest";

import { calculatePortfolioTotals } from "./portfolio";
import type { Loan, Portfolio } from "./types";

function makeLoan(overrides: Partial<Loan> = {}): Loan {
  return {
    id: "loan",
    platform: "Test",
    amount: 0,
    interest: 0,
    start_date: "2024-01-01",
    term_months: null,
    liquidation_ltv: 100,
    btc_bought: 0,
    ...overrides,
  };
}

function makePortfolio(overrides: Partial<Portfolio> = {}): Portfolio {
  return {
    btc_owned: 1,
    currency: "USD",
    btc_price: 100_000,
    income_per_year: 0,
    btc_saving_rate_percent: 0,
    other_assets: 0,
    loans: [],
    ...overrides,
  };
}

describe("calculatePortfolioTotals", () => {
  it("returns plain totals for an empty portfolio", () => {
    const totals = calculatePortfolioTotals(makePortfolio({ loans: [] }));
    expect(totals.totalBtc).toBe(1);
    expect(totals.netBtc).toBe(1);
    expect(totals.totalDebt).toBe(0);
    expect(totals.portfolioValue).toBe(100_000);
    expect(totals.ltv).toBe(0);
    expect(totals.netAssets).toBe(100_000);
  });

  it("includes BTC bought via loans in totalBtc", () => {
    const portfolio = makePortfolio({
      loans: [makeLoan({ btc_bought: 0.5, amount: 0 })],
    });
    const totals = calculatePortfolioTotals(portfolio);
    expect(totals.totalBtc).toBeCloseTo(1.5);
    expect(totals.netBtc).toBeCloseTo(1.5);
  });

  it("computes accrued interest for a still-running loan", () => {
    const start = "2024-01-01";
    const now = new Date("2025-01-01"); // exactly one year later
    const portfolio = makePortfolio({
      loans: [
        makeLoan({ amount: 10_000, interest: 10, start_date: start }),
      ],
    });
    const totals = calculatePortfolioTotals(portfolio, now);
    // 10k principal + 10% over 365 days ~ 11000
    expect(totals.totalDebt).toBeGreaterThan(10_900);
    expect(totals.totalDebt).toBeLessThan(11_100);
  });

  it("caps interest accrual at the loan's term end", () => {
    const start = "2024-01-01";
    const now = new Date("2026-01-01"); // 2 years later, beyond a 6-month loan
    const portfolio = makePortfolio({
      loans: [
        makeLoan({ amount: 10_000, interest: 10, start_date: start, term_months: 6 }),
      ],
    });
    const totals = calculatePortfolioTotals(portfolio, now);
    // Accrual stops at the end date (~6 months), so debt is much less than
    // a 2-year unbounded accrual.
    expect(totals.totalDebt).toBeLessThan(10_700);
  });

  it("handles a portfolio with zero BTC and other assets", () => {
    const portfolio = makePortfolio({
      btc_owned: 0,
      other_assets: 50_000,
    });
    const totals = calculatePortfolioTotals(portfolio);
    expect(totals.portfolioValue).toBe(0);
    expect(totals.ltv).toBe(1_000_000_000);
    expect(totals.btcExposure).toBe(0);
    expect(totals.totalAssets).toBe(50_000);
  });

  it("net BTC subtracts debt in BTC at the portfolio price", () => {
    const portfolio = makePortfolio({
      btc_owned: 1,
      btc_price: 100_000,
      loans: [makeLoan({ amount: 50_000, interest: 0, start_date: "2024-01-01" })],
    });
    const totals = calculatePortfolioTotals(portfolio, new Date("2024-01-01"));
    expect(totals.totalBtc).toBe(1);
    expect(totals.totalDebt).toBe(50_000);
    expect(totals.netBtc).toBeCloseTo(0.5);
  });

  it("computes BTC exposure as the BTC fraction of total assets", () => {
    const portfolio = makePortfolio({
      btc_owned: 1,
      btc_price: 100_000,
      other_assets: 100_000,
    });
    const totals = calculatePortfolioTotals(portfolio);
    expect(totals.btcExposure).toBeCloseTo(0.5);
  });
});
