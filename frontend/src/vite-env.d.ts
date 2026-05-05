/// <reference types="vite/client" />

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
