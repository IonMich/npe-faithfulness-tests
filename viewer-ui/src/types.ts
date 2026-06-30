export type ModelOption = {
  id: string;
  label: string;
  plot_label?: string;
  color?: string;
  kind: string;
  training_scope?: string;
  training_description?: string;
  train_simulations?: number;
  local_quantile?: number | null;
  checkpoint: string;
  has_local_region: boolean;
};

export type SummaryRow = {
  parameter: string;
  median: string;
  q16: string;
  q84: string;
  q05: string;
  q95: string;
};

export type MetricValue = {
  value?: number;
};

export type GridMetrics = {
  mean_normalized_wasserstein?: MetricValue;
};

export type GridMetadata = {
  grid_size: number;
  grid_points: number;
  deterministic?: boolean;
  grid_expansions: number;
  max_edge_mass: number;
  sample_count?: number;
  elapsed_seconds: number;
};

export type NpeGridMetadata = {
  grid_size: number;
  grid_points: number;
  resolution_cap: number;
  uses_reference_ranges: boolean;
};

export type McmcMetadata = {
  chains: number;
  steps: number;
  burn_in: number;
  runtime_seconds: number;
  elapsed_seconds: number;
  acceptance_rate: number;
  convergence_ok: boolean;
};

export type ViewerResponse = {
  draw_id: string;
  corner: string;
  signal: string;
  true_theta: {
    A: number;
    k: number;
    sigma: number;
  };
  posterior_summary: SummaryRow[];
  npe_summaries: Array<{
    model_id: string;
    label: string;
    full_label: string;
    summary: SummaryRow[];
  }>;
  grid_summary: SummaryRow[] | null;
  mcmc_summary: SummaryRow[] | null;
  grid_metrics: GridMetrics | null;
  npe_grid_metrics: Record<string, GridMetrics>;
  mcmc_grid_metrics: GridMetrics | null;
  grid_metadata: GridMetadata | null;
  npe_grid_metadata: NpeGridMetadata | null;
  mcmc_metadata: McmcMetadata | null;
  mode_metadata: {
    mode: string;
    [key: string]: unknown;
  };
  local_distance: number | null;
  local_radius: number | null;
  inside_local_region: boolean | null;
  posterior_samples: number;
  npe_render_mode: "sample" | "grid";
  include_npe: boolean;
  elapsed_seconds: number;
  timing: {
    npe_sampling_seconds: number;
    npe_samples_per_second: number;
    grid_seconds: number | null;
    mcmc_seconds: number | null;
    mcmc_elapsed_seconds: number | null;
    grid_points_per_second: number | null;
    plot_seconds: number;
    total_seconds: number;
  };
  model: string;
  model_id: string;
  model_metadata: ModelOption | null;
  selected_npe_model_ids: string[];
  selected_npe_models: ModelOption[];
  summary: {
    checkpoint_context: string;
    training_mode?: string;
    train_simulations?: number;
    local_quantile?: number | null;
  };
  z_sample_shape: number[];
};

export type Overlay =
  | "local_flow"
  | "broad_mdn"
  | "broad_mdn_512k"
  | "broad_spline_4m"
  | "grid"
  | "mcmc";

export type ControlState = {
  mode: string;
  samples: number;
  npeMode: "sample" | "grid";
  npeGridSize: string;
  overlays: Overlay[];
  gridSize: string;
  activeView: "corner" | "signal";
};
