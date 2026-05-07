import type { Loan, Portfolio } from "./types";

export type PortfolioTotals = {
  totalBtc: number;
  /** BTC stack minus debt expressed in BTC at `portfolio.btc_price` (same idea as simulation `net_btc`). */
  netBtc: number;
  totalDebt: number;
  portfolioValue: number;
  ltv: number;
  totalAssets: number;
  netAssets: number;
  btcExposure: number;
};

const MILLISECONDS_PER_DAY = 86_400_000;
const SAFE_INFINITY = 1_000_000_000;

function accruedDebtForLoan(loan: Loan, now: Date): number {
  const start = new Date(loan.start_date);
  const end = loan.term_months
    ? new Date(start.getFullYear(), start.getMonth() + loan.term_months, start.getDate())
    : now;
  const effectiveEnd = end < now ? end : now;
  const daysPassed = Math.max(0, (effectiveEnd.getTime() - start.getTime()) / MILLISECONDS_PER_DAY);
  return loan.amount + (loan.amount * loan.interest * daysPassed) / 36500;
}

export function calculatePortfolioTotals(portfolio: Portfolio, now: Date = new Date()): PortfolioTotals {
  const btcFromLoans = portfolio.loans.reduce((sum, loan) => sum + loan.btc_bought, 0);
  const totalBtc = portfolio.btc_owned + btcFromLoans;

  const totalDebt = portfolio.loans.reduce((sum, loan) => sum + accruedDebtForLoan(loan, now), 0);

  const price = portfolio.btc_price;
  const netBtc = price > 1e-12 ? totalBtc - totalDebt / price : totalBtc;

  const portfolioValue = totalBtc * portfolio.btc_price;
  const ltv = portfolioValue > 0 ? totalDebt / portfolioValue : SAFE_INFINITY;
  const totalAssets = portfolioValue + portfolio.other_assets;
  const netAssets = totalAssets - totalDebt;
  const btcExposure = totalAssets > 0 ? portfolioValue / totalAssets : 0;

  return { totalBtc, netBtc, totalDebt, portfolioValue, ltv, totalAssets, netAssets, btcExposure };
}
