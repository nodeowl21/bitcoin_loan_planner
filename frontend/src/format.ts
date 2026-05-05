import type { ChangeEvent } from "react";

import type { Currency } from "./types";

export function numberValue(event: ChangeEvent<HTMLInputElement>): number {
  return Number(event.target.value);
}

export function formatCurrency(value: number, currency: Currency, digits = 0): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatNumber(value: number, maximumFractionDigits = 2): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(value);
}

export function formatPercent(value: number): string {
  return `${formatNumber(value * 100, 2)}%`;
}
