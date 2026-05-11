import { useEffect, useState } from "react";
import type { InputHTMLAttributes } from "react";

export type NumericInputProps = Omit<
  InputHTMLAttributes<HTMLInputElement>,
  "type" | "value" | "onChange" | "inputMode" | "min" | "max"
> & {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  /** Round to integer on commit (e.g. years, months). */
  integer?: boolean;
  /** Allow a leading minus while typing (e.g. expected return %). */
  allowNegative?: boolean;
};

function serializeDisplay(v: number, integer: boolean): string {
  if (integer) return String(Math.round(v));
  return String(v);
}

function normalizeDecimalSeparators(raw: string): string {
  return raw.trim().replace(",", ".");
}

/** Commit rules: empty / invalid → 0, unless `min > 0` then → `min` (e.g. duration months, BTC price). */
export function commitNumericText(
  raw: string,
  opts: { min?: number; max?: number; integer?: boolean; allowNegative?: boolean },
): number {
  const t = normalizeDecimalSeparators(raw);
  const incomplete = t === "" || t === "-" || t === "+" || t === "." || t === "-." || t === "+.";
  if (incomplete) {
    if (opts.min !== undefined && opts.min > 0) return opts.min;
    return 0;
  }

  let n = Number(t);
  if (!Number.isFinite(n)) {
    if (opts.min !== undefined && opts.min > 0) return opts.min;
    return 0;
  }

  if (!opts.allowNegative && n < 0) {
    n = 0;
  }

  if (opts.integer) n = Math.round(n);
  if (opts.min !== undefined) n = Math.max(opts.min, n);
  if (opts.max !== undefined) n = Math.min(opts.max, n);
  return n;
}

function partialPattern(allowNegative: boolean): RegExp {
  return allowNegative ? /^-?\d*[.,]?\d*$/ : /^\d*[.,]?\d*$/;
}

export function NumericInput({
  value,
  onChange,
  min,
  max,
  integer = false,
  allowNegative = false,
  disabled,
  className,
  onFocus,
  onBlur,
  ...rest
}: NumericInputProps) {
  const [focused, setFocused] = useState(false);
  const [text, setText] = useState(() => serializeDisplay(value, integer));

  useEffect(() => {
    if (!focused) {
      setText(serializeDisplay(value, integer));
    }
  }, [value, focused, integer]);

  const pattern = partialPattern(allowNegative);

  return (
    <input
      {...rest}
      type="text"
      inputMode={integer ? "numeric" : "decimal"}
      autoComplete="off"
      disabled={disabled}
      className={className}
      value={focused ? text : serializeDisplay(value, integer)}
      onFocus={(e) => {
        setFocused(true);
        setText(serializeDisplay(value, integer));
        onFocus?.(e);
      }}
      onChange={(e) => {
        const next = e.target.value;
        if (next !== "" && !pattern.test(next)) return;
        if (!allowNegative && next.includes("-")) return;
        setText(next);
      }}
      onBlur={(e) => {
        const n = commitNumericText(text, { min, max, integer, allowNegative });
        onChange(n);
        setFocused(false);
        setText(serializeDisplay(n, integer));
        onBlur?.(e);
      }}
    />
  );
}
