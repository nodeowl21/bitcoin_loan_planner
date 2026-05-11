import { describe, expect, it } from "vitest";

import { commitNumericText } from "./NumericInput";

describe("commitNumericText", () => {
  it("treats empty input as 0 when min is not positive", () => {
    expect(commitNumericText("", { min: 0 })).toBe(0);
    expect(commitNumericText("   ", { min: 0 })).toBe(0);
  });

  it("treats empty as min when min > 0", () => {
    expect(commitNumericText("", { min: 1 })).toBe(1);
    expect(commitNumericText("-", { min: 50, max: 100 })).toBe(50);
  });

  it("parses decimals and clamps", () => {
    expect(commitNumericText("12.6", { min: 0, max: 100 })).toBe(12.6);
    expect(commitNumericText("150", { min: 0, max: 100 })).toBe(100);
  });

  it("respects allowNegative and min", () => {
    expect(commitNumericText("-5", { min: -100, max: 200, allowNegative: true })).toBe(-5);
  });

  it("floors negative to 0 when negatives not allowed", () => {
    expect(commitNumericText("-3", { min: 0, max: 100, allowNegative: false })).toBe(0);
  });

  it("rounds when integer", () => {
    expect(commitNumericText("3.7", { min: 1, max: 360, integer: true })).toBe(4);
  });
});
