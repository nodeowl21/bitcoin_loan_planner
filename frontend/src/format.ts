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

/** Plain number with an explicit + for positive values (except zero). */
export function formatSignedNumber(value: number, maximumFractionDigits = 2): string {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits,
    signDisplay: "exceptZero",
  }).format(value);
}

/** Currency with an explicit + for positive values (except zero). */
export function formatSignedCurrency(value: number, currency: Currency, digits = 0): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: digits,
    signDisplay: "exceptZero",
  }).format(value);
}

export function deltaTone(value: number, eps = 1e-9): "positive" | "negative" | "neutral" {
  if (value > eps) return "positive";
  if (value < -eps) return "negative";
  return "neutral";
}

export function formatPercent(value: number): string {
  return `${formatNumber(value * 100, 2)}%`;
}
