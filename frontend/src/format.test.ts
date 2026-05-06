import { describe, expect, it } from "vitest";

import {
  deltaTone,
  formatCurrency,
  formatNumber,
  formatPercent,
  formatSignedCurrency,
  formatSignedNumber,
  numberValue,
} from "./format";

describe("formatCurrency", () => {
  it("formats USD without fractional digits by default", () => {
    expect(formatCurrency(1234.56, "USD")).toBe("$1,235");
  });

  it("respects the digits argument", () => {
    expect(formatCurrency(1234.56, "USD", 2)).toBe("$1,234.56");
  });

  it("supports EUR", () => {
    expect(formatCurrency(1000, "EUR")).toMatch(/1,000/);
  });
});

describe("formatNumber", () => {
  it("uses up to 2 fraction digits by default", () => {
    expect(formatNumber(1.2345)).toBe("1.23");
  });

  it("respects custom precision", () => {
    expect(formatNumber(1.234567, 4)).toBe("1.2346");
  });

  it("formats integers without trailing zeros", () => {
    expect(formatNumber(1500)).toBe("1,500");
  });
});

describe("formatPercent", () => {
  it("multiplies the value by 100 and appends %", () => {
    expect(formatPercent(0.1234)).toBe("12.34%");
  });

  it("renders zero", () => {
    expect(formatPercent(0)).toBe("0%");
  });

  it("renders negative values", () => {
    expect(formatPercent(-0.05)).toBe("-5%");
  });
});

describe("formatSignedNumber", () => {
  it("shows a plus sign for positive values", () => {
    expect(formatSignedNumber(1.5)).toBe("+1.5");
  });

  it("shows a minus sign for negative values", () => {
    expect(formatSignedNumber(-2.25)).toBe("-2.25");
  });

  it("omits a sign for zero", () => {
    expect(formatSignedNumber(0)).toBe("0");
  });
});

describe("formatSignedCurrency", () => {
  it("shows explicit signs for non-zero USD", () => {
    expect(formatSignedCurrency(100, "USD", 2)).toBe("+$100.00");
    expect(formatSignedCurrency(-50, "USD", 2)).toBe("-$50.00");
  });
});

describe("deltaTone", () => {
  it("classifies by epsilon", () => {
    expect(deltaTone(0.001, 1e-9)).toBe("positive");
    expect(deltaTone(-0.001, 1e-9)).toBe("negative");
    expect(deltaTone(1e-12, 1e-9)).toBe("neutral");
  });
});

describe("numberValue", () => {
  it("converts a numeric input value to a number", () => {
    const event = {
      target: { value: "42" } as HTMLInputElement,
    } as unknown as Parameters<typeof numberValue>[0];
    expect(numberValue(event)).toBe(42);
  });

  it("returns NaN for non-numeric input", () => {
    const event = {
      target: { value: "abc" } as HTMLInputElement,
    } as unknown as Parameters<typeof numberValue>[0];
    expect(Number.isNaN(numberValue(event))).toBe(true);
  });
});
