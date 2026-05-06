/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_FIREFISH_AFFILIATE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module "react-plotly.js" {
  import type { ComponentType, CSSProperties } from "react";

  const Plot: ComponentType<{
    data: unknown[];
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
    useResizeHandler?: boolean;
    className?: string;
    style?: CSSProperties;
  }>;

  export default Plot;
}
