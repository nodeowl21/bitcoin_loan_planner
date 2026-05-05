import { defaultPortfolio, defaultPresets, defaultSimulation } from "./defaults";
import type { ExportData, Portfolio, SimulationConfig, StrategyPresets } from "./types";

export type ExportInput = {
  portfolio: Portfolio;
  strategyPresets: StrategyPresets;
  defaultStrategy: string;
  simulation: SimulationConfig;
  selectedSimStrategy: string;
};

export function buildExport(input: ExportInput): ExportData {
  const { loans, ...portfolioWithoutLoans } = input.portfolio;
  return {
    portfolio: portfolioWithoutLoans,
    loans,
    strategies: {
      presets: input.strategyPresets,
      default: input.defaultStrategy,
    },
    simulation: { ...input.simulation, selected_sim_strategy: input.selectedSimStrategy },
  };
}

export type ParsedImport = {
  portfolio: Portfolio;
  strategyPresets: StrategyPresets;
  defaultStrategy: string;
  simulation: SimulationConfig;
  selectedSimStrategy: string;
};

/**
 * Parse an import payload (parsed JSON, not a JSON string) and merge it with
 * defaults so that the resulting state is always shaped correctly even when
 * fields are missing. Throws when the payload is not a JSON object.
 */
export function parseImport(payload: unknown): ParsedImport {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("Import payload must be a JSON object");
  }
  const data = payload as Partial<ExportData>;
  const importedPortfolio = data.portfolio
    ? { ...defaultPortfolio, ...data.portfolio }
    : defaultPortfolio;
  const portfolio: Portfolio = { ...importedPortfolio, loans: data.loans ?? [] };
  const strategyPresets = data.strategies?.presets ?? defaultPresets;
  const defaultStrategy = data.strategies?.default ?? "Custom";
  const simulation: SimulationConfig = { ...defaultSimulation, ...data.simulation };
  const selectedSimStrategy =
    data.simulation?.selected_sim_strategy ?? data.strategies?.default ?? "Custom";

  return {
    portfolio,
    strategyPresets,
    defaultStrategy,
    simulation,
    selectedSimStrategy,
  };
}

export function parseImportJson(text: string): ParsedImport {
  return parseImport(JSON.parse(text));
}
