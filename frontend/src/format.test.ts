import { describe, expect, it } from "vitest";

import { formatCurrency, formatNumber, formatPercent, numberValue } from "./format";

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
