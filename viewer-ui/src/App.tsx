import {
  Activity,
  BarChart3,
  Grid2X2,
  Loader2,
  Plus,
  RefreshCw,
  Timer,
  Waves
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import {
  Badge,
  Button,
  Card,
  CardHeader,
  Field,
  MultiSelect,
  NumberField,
  SelectField
} from "./components/ui";
import { compactPath, formatValue } from "./lib/utils";
import type { ControlState, ModelOption, Overlay, SummaryRow, ViewerResponse } from "./types";

const CONTROL_STATE_KEY = "npePosteriorViewer.controls.v4";
const PREVIOUS_CONTROL_STATE_KEY = "npePosteriorViewer.controls.v3";
const LEGACY_V2_CONTROL_STATE_KEY = "npePosteriorViewer.controls.v2";
const LEGACY_CONTROL_STATE_KEY = "npePosteriorViewer.controls.v1";

const NPE_MODEL_IDS = ["local_flow", "broad_mdn", "broad_mdn_512k", "broad_spline_4m"] as const;
type NpeOverlay = (typeof NPE_MODEL_IDS)[number];
const GRID_SIZE_OPTIONS = ["45", "60", "90", "120", "150", "180"] as const;
const NPE_GRID_SIZE_OPTIONS = ["30", "45", "60", "75", "90", "120", "150", "180"] as const;
const MAX_POSTERIOR_DRAWS = 500_000;

const defaultControls: ControlState = {
  mode: "local",
  samples: 7000,
  npeMode: "sample",
  npeGridSize: "60",
  overlays: ["local_flow"],
  gridSize: "60",
  activeView: "corner"
};

function isNpeLayer(value: string): value is NpeOverlay {
  return NPE_MODEL_IDS.includes(value as NpeOverlay);
}

function isOverlay(value: unknown): value is Overlay {
  return typeof value === "string" && (isNpeLayer(value) || value === "grid" || value === "mcmc");
}

function uniqueOverlays(overlays: Overlay[]): Overlay[] {
  return overlays.filter((item, index) => overlays.indexOf(item) === index);
}

function sortedOverlays(overlays: Overlay[]): Overlay[] {
  return [...overlays].sort();
}

function sameOverlaySet(left: Overlay[], right: Overlay[]): boolean {
  const leftSorted = sortedOverlays(left);
  const rightSorted = sortedOverlays(right);
  return (
    leftSorted.length === rightSorted.length &&
    leftSorted.every((item, index) => item === rightSorted[index])
  );
}

function normalizeSelectValue<T extends string>(
  value: unknown,
  options: readonly T[],
  fallback: T
): T {
  return typeof value === "string" && options.includes(value as T) ? (value as T) : fallback;
}

function modelIdToLayer(value: unknown): Overlay {
  return typeof value === "string" && isNpeLayer(value) ? value : "local_flow";
}

function migrateNpeOverlay(overlays: unknown[], modelId: unknown): Overlay[] {
  const selectedModelLayer = modelIdToLayer(modelId);
  const migrated = overlays
    .map((value) => (value === "npe" ? selectedModelLayer : value))
    .filter(isOverlay);
  return uniqueOverlays(migrated);
}

function withNpeDefault(overlays: Overlay[], modelId: unknown): Overlay[] {
  return overlays.some(isNpeLayer) ? overlays : uniqueOverlays([modelIdToLayer(modelId), ...overlays]);
}

function flagsToOverlays(
  referenceValue: unknown,
  includeMcmcValue: unknown,
  includeNpeValue: unknown = true,
  modelId: unknown = "local_flow"
): Overlay[] {
  const overlays: Overlay[] =
    includeNpeValue === false || includeNpeValue === "0" ? [] : [modelIdToLayer(modelId)];
  if (referenceValue === "grid") overlays.push("grid");
  if (includeMcmcValue === true || includeMcmcValue === "1") overlays.push("mcmc");
  return uniqueOverlays(overlays);
}

function parseStoredControls(): Partial<ControlState> {
  try {
    const current = JSON.parse(localStorage.getItem(CONTROL_STATE_KEY) || "{}");
    if (Array.isArray(current.overlays)) return current;
  } catch {
    // fall through to previous state shapes
  }
  try {
    const previous = JSON.parse(localStorage.getItem(PREVIOUS_CONTROL_STATE_KEY) || "{}");
    if (Array.isArray(previous.overlays)) {
      return {
        ...previous,
        overlays: withNpeDefault(migrateNpeOverlay(previous.overlays, previous.modelId), previous.modelId)
      };
    }
  } catch {
    // fall through to older state shapes
  }
  try {
    const previous = JSON.parse(localStorage.getItem(LEGACY_V2_CONTROL_STATE_KEY) || "{}");
    if (Array.isArray(previous.overlays)) {
      return {
        ...previous,
        overlays: withNpeDefault(migrateNpeOverlay(previous.overlays, previous.modelId), previous.modelId)
      };
    }
  } catch {
    // fall through to legacy state
  }
  try {
    const legacy = JSON.parse(localStorage.getItem(LEGACY_CONTROL_STATE_KEY) || "{}");
    if (Object.keys(legacy).length === 0) return {};
    return {
      mode: legacy.mode,
      samples: Number.parseInt(legacy.samples, 10),
      overlays: legacy.comparison
        ? comparisonToOverlays(legacy.comparison)
        : flagsToOverlays(legacy.reference, legacy.includeMcmc, true, legacy.modelId),
      gridSize: legacy.gridSize
    };
  } catch {
    return {};
  }
}

function comparisonToOverlays(value: string): Overlay[] {
  if (value === "grid_mcmc") return ["local_flow", "grid", "mcmc"];
  if (value === "mcmc") return ["local_flow", "mcmc"];
  if (value === "grid") return ["local_flow", "grid"];
  return ["local_flow"];
}

function normalizeControls(input: Partial<ControlState>): ControlState {
  return {
    ...defaultControls,
    ...input,
    samples: Number.isFinite(input.samples)
      ? Math.min(Math.max(Number(input.samples), 1000), MAX_POSTERIOR_DRAWS)
      : defaultControls.samples,
    overlays: Array.isArray(input.overlays)
      ? uniqueOverlays(input.overlays.filter(isOverlay))
      : defaultControls.overlays,
    npeMode: input.npeMode === "grid" ? "grid" : "sample",
    npeGridSize: normalizeSelectValue(
      input.npeGridSize,
      NPE_GRID_SIZE_OPTIONS,
      defaultControls.npeGridSize
    ),
    gridSize: normalizeSelectValue(input.gridSize, GRID_SIZE_OPTIONS, defaultControls.gridSize),
    activeView: input.activeView === "signal" ? "signal" : "corner"
  };
}

function overlaysFromResponse(data: ViewerResponse): Overlay[] {
  const overlays: Overlay[] = data.selected_npe_model_ids.filter(isNpeLayer);
  if (data.grid_summary) overlays.push("grid");
  if (data.mcmc_summary) overlays.push("mcmc");
  return uniqueOverlays(overlays);
}

function inferenceControlsKey(controls: ControlState): string {
  return JSON.stringify({
    gridSize: controls.gridSize,
    npeMode: controls.npeMode,
    npeGridSize: controls.npeGridSize,
    overlays: sortedOverlays(controls.overlays),
    samples: controls.samples
  });
}

function responseMatchesControls(data: ViewerResponse, controls: ControlState): boolean {
  const selectedGridSize = Number.parseInt(controls.gridSize, 10);
  const selectedNpeGridSize = Number.parseInt(controls.npeGridSize, 10);
  const hasSelectedNpe = controls.overlays.some(isNpeLayer);
  const gridMatches =
    !controls.overlays.includes("grid") || data.grid_metadata?.grid_size === selectedGridSize;
  const npeGridMatches =
    controls.npeMode !== "grid" ||
    !hasSelectedNpe ||
    data.npe_grid_metadata?.grid_size === selectedNpeGridSize;
  return (
    data.posterior_samples === controls.samples &&
    data.npe_render_mode === controls.npeMode &&
    gridMatches &&
    npeGridMatches &&
    sameOverlaySet(overlaysFromResponse(data), controls.overlays)
  );
}

function metricRows(data: ViewerResponse | null) {
  if (!data) return [];
  const selectedModels = data.selected_npe_models || [];
  const localModelSelected = selectedModels.some((model) => model.id === "local_flow");
  const outside = localModelSelected && data.inside_local_region === false;
  const regionStatus = !localModelSelected
    ? "local NPE not selected"
    : data.inside_local_region === null
      ? "n/a"
      : data.inside_local_region
        ? "inside local region"
        : "outside local region";
  const selectedLabels = selectedModels.map((model) => model.plot_label || model.label);

  return [
    ["NPE layers", selectedLabels.length ? selectedLabels.join(", ") : "none"],
    selectedModels.length
      ? ["NPE mode", data.npe_render_mode === "grid" ? "grid evaluated" : "posterior samples"]
      : null,
    selectedModels.length
      ? ["training scopes", selectedModels.map((model) => model.training_scope || model.kind).join(", ")]
      : null,
    ["true A", formatValue(data.true_theta.A)],
    ["true k", formatValue(data.true_theta.k)],
    ["true sigma", formatValue(data.true_theta.sigma)],
    ["region status", regionStatus, outside ? "warn" : ""],
    localModelSelected ? ["local distance", formatValue(data.local_distance), outside ? "warn" : ""] : null,
    localModelSelected ? ["local radius", formatValue(data.local_radius)] : null,
    data.grid_metadata
      ? [
          "grid",
          `${data.grid_metadata.grid_size}^3, edge ${formatValue(data.grid_metadata.max_edge_mass)}`
        ]
      : null,
    ["draws", data.posterior_samples.toLocaleString()],
    ...selectedModels.map(
      (model) =>
        [`${model.plot_label || model.label} checkpoint`, compactPath(model.checkpoint)] as [string, string]
    )
  ].filter(Boolean) as [string, string, string?][];
}

function wassersteinDistanceItems(data: ViewerResponse | null) {
  if (!data?.grid_summary) return [];
  const npeItems = (data.selected_npe_models || [])
    .map((model) => {
      const metric = data.npe_grid_metrics?.[model.id]?.mean_normalized_wasserstein?.value;
      if (metric === undefined) return null;
      return {
        label: model.id === "local_flow" ? "Local NPE" : model.id === "broad_mdn" ? "Broad NPE" : model.plot_label || model.label,
        value: formatValue(metric)
      };
    })
    .filter(Boolean) as Array<{ label: string; value: string }>;
  const mcmcMetric = data.mcmc_grid_metrics?.mean_normalized_wasserstein?.value;
  return [
    ...npeItems,
    mcmcMetric === undefined ? null : { label: "MCMC", value: formatValue(mcmcMetric) }
  ].filter(Boolean) as Array<{ label: string; value: string }>;
}

function timingRows(data: ViewerResponse | null) {
  if (!data) return [];
  const timing = data.timing || {};
  return [
    data.selected_npe_model_ids.length
      ? [
          data.npe_render_mode === "grid" ? "NPE grid eval" : "NPE sampling",
          data.npe_render_mode === "grid"
            ? `${timing.npe_sampling_seconds.toFixed(3)} s, ${data.npe_grid_metadata?.grid_size ?? "?"}^3`
            : `${timing.npe_sampling_seconds.toFixed(3)} s, ${Math.round(
                timing.npe_samples_per_second
              ).toLocaleString()} samples/s`
        ]
      : null,
    data.mcmc_metadata
      ? [
          "MCMC sampling",
          `${(timing.mcmc_seconds || 0).toFixed(3)} s, ${data.mcmc_metadata.chains} x ${
            data.mcmc_metadata.steps
          }`
        ]
      : null,
    data.mcmc_metadata
      ? [
          "MCMC diagnostics",
          data.mcmc_metadata.convergence_ok ? "passed" : "not passed",
          data.mcmc_metadata.convergence_ok ? "" : "warn"
        ]
      : null,
    timing.grid_seconds === null
      ? null
      : [
          "grid reference",
          `${timing.grid_seconds.toFixed(3)} s, ${Math.round(
            timing.grid_points_per_second || 0
          ).toLocaleString()} points/s`
        ],
    ["plotting", `${timing.plot_seconds.toFixed(3)} s`],
    ["render time", `${data.elapsed_seconds.toFixed(2)} s`]
  ].filter(Boolean) as [string, string, string?][];
}

function WassersteinStrip({ data }: { data: ViewerResponse | null }) {
  const items = wassersteinDistanceItems(data);
  if (!items.length) return null;
  return (
    <div className="distance-panel" aria-label="Wasserstein distances to grid">
      <div className="distance-panel-title">Wasserstein to grid</div>
      <div className="distance-grid">
        {items.map((item) => (
          <div className="distance-tile" key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function SummaryTable({
  npeSummaries,
  gridRows,
  mcmcRows
}: {
  npeSummaries: ViewerResponse["npe_summaries"];
  gridRows: SummaryRow[] | null;
  mcmcRows: SummaryRow[] | null;
}) {
  const combined = [
    ...npeSummaries.flatMap((item) =>
      item.summary.map((row) => ({ ...row, source: item.label }))
    ),
    ...(gridRows || []).map((row) => ({ ...row, source: "Grid" })),
    ...(mcmcRows || []).map((row) => ({ ...row, source: "MCMC" }))
  ];
  if (!combined.length) return <div className="empty">No selected posterior layers.</div>;
  return (
    <table className="summary-table">
      <thead>
        <tr>
          <th>source</th>
          <th>param</th>
          <th>q05</th>
          <th>q16</th>
          <th>median</th>
          <th>q84</th>
          <th>q95</th>
        </tr>
      </thead>
      <tbody>
        {combined.map((row) => (
          <tr key={`${row.source}-${row.parameter}`}>
            <td>{row.source}</td>
            <td>{row.parameter}</td>
            <td>{row.q05}</td>
            <td>{row.q16}</td>
            <td>{row.median}</td>
            <td>{row.q84}</td>
            <td>{row.q95}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function KeyValueList({ rows }: { rows: [string, string, string?][] }) {
  return (
    <div className="kv-list">
      {rows.map(([label, value, tone]) => (
        <div className={`kv-row ${tone === "warn" ? "kv-warn" : ""}`} key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function DiagnosticPreview({
  active,
  image,
  label,
  onSelect
}: {
  active: boolean;
  image: string;
  label: string;
  onSelect: () => void;
}) {
  return (
    <button
      aria-pressed={active}
      className={`preview-tile ${active ? "preview-active" : ""}`}
      onClick={onSelect}
      type="button"
    >
      <span>{label}</span>
      <img alt={`${label} preview`} src={image} />
    </button>
  );
}

export default function App() {
  const [models, setModels] = useState<ModelOption[]>([]);
  const [controls, setControls] = useState<ControlState>(() =>
    normalizeControls(parseStoredControls())
  );
  const [data, setData] = useState<ViewerResponse | null>(null);
  const [loadingMode, setLoadingMode] = useState<"draw" | "update" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hasRendered, setHasRendered] = useState(false);
  const lastFailedRequestKey = useRef<string | null>(null);
  const loading = loadingMode !== null;

  const selectedNpeModelIds = controls.overlays.filter(isNpeLayer);
  const modelLabelById = useMemo(
    () => new Map(models.map((model) => [model.id, model.plot_label || model.label])),
    [models]
  );
  const selectedNpeLayerLabels = selectedNpeModelIds.map((modelId) =>
    modelLabelById.get(modelId) || (modelId === "local_flow" ? "Local NPE" : "Broad NPE")
  );
  const hasGrid = controls.overlays.includes("grid");
  const drawingNewSignal = loadingMode === "draw";
  const layerOptions = useMemo(
    (): Array<{ value: Overlay; label: string; icon: ReactNode; refreshable?: boolean }> => [
      ...models
        .filter((model) => isNpeLayer(model.id))
        .map((model) => ({
          value: model.id as NpeOverlay,
          label: model.plot_label || model.label,
          icon: model.id === "local_flow" ? <Activity size={14} /> : <BarChart3 size={14} />,
          refreshable: controls.npeMode === "sample"
        })),
      { value: "grid" as const, label: "Grid reference", icon: <Grid2X2 size={14} />, refreshable: false },
      { value: "mcmc" as const, label: "MCMC", icon: <Waves size={14} />, refreshable: true }
    ],
    [models, controls.npeMode]
  );

  useEffect(() => {
    async function loadModels() {
      const response = await fetch("/api/models");
      if (!response.ok) throw new Error(await response.text());
      const payload = (await response.json()) as ModelOption[];
      setModels(payload);
      setControls((current) => {
        const availableNpeIds = new Set(
          payload.map((model) => model.id).filter(isNpeLayer)
        );
        const overlays = current.overlays.filter(
          (layer) => layer === "grid" || layer === "mcmc" || availableNpeIds.has(layer)
        );
        if (overlays.length === current.overlays.length) return current;
        return { ...current, overlays };
      });
    }
    loadModels().catch((reason: unknown) => setError(String(reason)));
  }, []);

  useEffect(() => {
    localStorage.setItem(CONTROL_STATE_KEY, JSON.stringify(controls));
  }, [controls]);

  async function runInference({
    freshSignal,
    controlsOverride,
    refreshLayers = []
  }: {
    freshSignal: boolean;
    controlsOverride?: ControlState;
    refreshLayers?: Overlay[];
  }) {
    const requestControls = controlsOverride || controls;
    const requestKey = inferenceControlsKey(requestControls);
    setLoadingMode(freshSignal ? "draw" : "update");
    setError(null);
    const selectedModelIds = requestControls.overlays.filter(isNpeLayer);
    const query = new URLSearchParams({
      model_ids: selectedModelIds.join(","),
      mode: requestControls.mode,
      samples: String(requestControls.samples),
      npe_mode: requestControls.npeMode,
      reference: requestControls.overlays.includes("grid") ? "grid" : "none",
      mcmc: requestControls.overlays.includes("mcmc") ? "1" : "0",
      grid_size: requestControls.gridSize,
      npe_grid_size: requestControls.npeGridSize
    });
    if (refreshLayers.length) {
      query.set("refresh_layers", uniqueOverlays(refreshLayers).join(","));
    }
    if (!freshSignal && data?.draw_id) {
      query.set("draw_id", data.draw_id);
    } else if (!freshSignal) {
      query.set("reuse_current", "1");
    }
    try {
      const response = await fetch(`/api/new?${query.toString()}`);
      if (!response.ok) throw new Error(await response.text());
      setData((await response.json()) as ViewerResponse);
      setHasRendered(true);
      lastFailedRequestKey.current = null;
    } catch (reason: unknown) {
      lastFailedRequestKey.current = requestKey;
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoadingMode(null);
    }
  }

  function redrawLayer(layer: Overlay) {
    if (loading) return;
    const nextControls = controls.overlays.includes(layer)
      ? controls
      : { ...controls, overlays: uniqueOverlays([...controls.overlays, layer]) };
    if (nextControls !== controls) {
      setControls(nextControls);
    }
    void runInference({
      freshSignal: !hasRendered || !data?.draw_id,
      controlsOverride: nextControls,
      refreshLayers: [layer]
    });
  }

  useEffect(() => {
    if (models.length && !hasRendered && !loading && !data) {
      void runInference({ freshSignal: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models.length]);

  useEffect(() => {
    if (!hasRendered || !data || loading || responseMatchesControls(data, controls)) return;
    const requestKey = inferenceControlsKey(controls);
    if (lastFailedRequestKey.current === requestKey) return;
    const timeoutId = window.setTimeout(() => {
      void runInference({ freshSignal: false });
    }, 250);
    return () => window.clearTimeout(timeoutId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    controls.overlays,
    controls.samples,
    controls.gridSize,
    controls.npeMode,
    controls.npeGridSize,
    data,
    hasRendered,
    loading
  ]);

  const activeImage = controls.activeView === "corner" ? data?.corner : data?.signal;
  const activeAlt = controls.activeView === "corner" ? "posterior corner plot" : "signal predictive plot";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <Activity size={18} />
          </div>
          <div>
            <h1>NPE Posterior Viewer</h1>
            <p>Exponential decay posterior diagnostics</p>
          </div>
        </div>
        <div className={`control-grid ${controls.npeMode === "grid" ? "control-grid-npe-grid" : ""}`}>
          <Field label="Draw source">
            <SelectField
              ariaLabel="Signal source"
              value={controls.mode}
              onChange={(mode) => setControls((current) => ({ ...current, mode }))}
            >
              <option value="local">Local-region signal</option>
              <option value="x0">Original x0</option>
              <option value="prior">Prior predictive</option>
            </SelectField>
          </Field>
          <Field label="Draws">
            <NumberField
              ariaLabel="Posterior draws"
              max={MAX_POSTERIOR_DRAWS}
              min={1000}
              step={1000}
              value={controls.samples}
              onChange={(samples) => setControls((current) => ({ ...current, samples }))}
            />
          </Field>
          <Field label="Posterior layers" className="compare-field">
            <MultiSelect
              options={layerOptions}
              placeholder="No layers"
              onRefresh={redrawLayer}
              refreshDisabled={loading}
              value={controls.overlays}
              onChange={(overlays) => setControls((current) => ({ ...current, overlays }))}
            />
          </Field>
          <Field label="NPE">
            <SelectField
              ariaLabel="NPE rendering mode"
              value={controls.npeMode}
              onChange={(npeMode) =>
                setControls((current) => ({
                  ...current,
                  npeMode: npeMode === "grid" ? "grid" : "sample"
                }))
              }
            >
              <option value="sample">Samples</option>
              <option value="grid">Grid eval</option>
            </SelectField>
          </Field>
          {controls.npeMode === "grid" ? (
            <Field label="NPE grid">
              <SelectField
                ariaLabel="NPE grid size"
                value={controls.npeGridSize}
                onChange={(npeGridSize) => setControls((current) => ({ ...current, npeGridSize }))}
              >
                {NPE_GRID_SIZE_OPTIONS.map((value) => (
                  <option key={value} value={value}>
                    {value}^3
                  </option>
                ))}
              </SelectField>
            </Field>
          ) : null}
          <Field label="Ref grid">
            <SelectField
              ariaLabel="Reference grid size"
              value={controls.gridSize}
              onChange={(gridSize) => setControls((current) => ({ ...current, gridSize }))}
            >
              {GRID_SIZE_OPTIONS.map((value) => (
                <option key={value} value={value}>
                  {value}^3
                </option>
              ))}
            </SelectField>
          </Field>
          <Button
            className="run-button"
            disabled={loading}
            onClick={() => runInference({ freshSignal: true })}
          >
            {drawingNewSignal ? <Loader2 className="spin" size={16} /> : hasRendered ? <Plus size={16} /> : <RefreshCw size={16} />}
            {hasRendered ? "New draw" : "Draw & infer"}
          </Button>
        </div>
      </header>

      <main className="workspace">
        <aside className="inspector">
          <Card>
            <CardHeader
              title="Current draw"
              meta={
                data ? (
                  <Badge tone={data.inside_local_region === false ? "warn" : "muted"}>
                    {data.mode_metadata.mode}
                  </Badge>
                ) : null
              }
            />
            <div className="card-body">
              {data?.selected_npe_models.length ? (
                <div className="model-strip">
                  {data.selected_npe_models.map((model) => (
                    <Badge key={model.id} tone={model.has_local_region ? "default" : "ok"}>
                      {model.plot_label || model.label}
                    </Badge>
                  ))}
                </div>
              ) : null}
              <WassersteinStrip data={data} />
              {data ? <KeyValueList rows={metricRows(data)} /> : <div className="empty">No draw yet.</div>}
            </div>
          </Card>

          <Card>
            <CardHeader title="Runtime" meta={<Timer size={15} />} />
            <div className="card-body">
              {data ? <KeyValueList rows={timingRows(data)} /> : <div className="empty">Waiting for first run.</div>}
            </div>
          </Card>

          <Card className="summary-card">
            <CardHeader title="Posterior quantiles" meta={<BarChart3 size={15} />} />
            <div className="table-wrap">
              {data ? (
                <SummaryTable
                  npeSummaries={data.npe_summaries}
                  gridRows={hasGrid ? data.grid_summary : null}
                  mcmcRows={controls.overlays.includes("mcmc") ? data.mcmc_summary : null}
                />
              ) : (
                <div className="empty">No posterior samples yet.</div>
              )}
            </div>
          </Card>
        </aside>

        <Card className="visual-card">
          <CardHeader
            title="Posterior diagnostics"
            meta={
              <div className="visual-meta">
                <Badge tone="muted">{controls.activeView === "corner" ? "corner" : "signal"}</Badge>
                {selectedNpeLayerLabels.map((label) => (
                  <Badge key={label} tone="muted">
                    {label}
                  </Badge>
                ))}
                {selectedNpeLayerLabels.length ? (
                  <Badge tone="muted">
                    {controls.npeMode === "grid" ? `NPE grid ${controls.npeGridSize}^3` : "NPE samples"}
                  </Badge>
                ) : null}
                {hasGrid ? <Badge tone="muted">grid {controls.gridSize}^3</Badge> : null}
                {controls.overlays.includes("mcmc") ? <Badge tone="muted">MCMC</Badge> : null}
                {loading ? <Badge tone="default">rendering</Badge> : null}
              </div>
            }
          />
          <div className="visual-content">
            {loading ? (
              <div className="visual-placeholder visual-loading">
                <Loader2 className="spin" size={22} />
                <span>
                  {loadingMode === "update" ? "Updating current draw" : "Sampling posterior diagnostics"}
                </span>
              </div>
            ) : error ? (
              <div className="visual-placeholder visual-error">
                <Waves size={22} />
                <span>{error}</span>
              </div>
            ) : data && activeImage ? (
              <div className="visual-comparison">
                <div className="visual-main">
                  <img alt={activeAlt} className="result-image" src={activeImage} />
                </div>
                <div className="preview-rail" aria-label="Diagnostic views">
                  <DiagnosticPreview
                    active={controls.activeView === "corner"}
                    image={data.corner}
                    label="Corner"
                    onSelect={() => setControls((current) => ({ ...current, activeView: "corner" }))}
                  />
                  <DiagnosticPreview
                    active={controls.activeView === "signal"}
                    image={data.signal}
                    label="Signal"
                    onSelect={() => setControls((current) => ({ ...current, activeView: "signal" }))}
                  />
                </div>
              </div>
            ) : (
              <div className="visual-placeholder">
                <Grid2X2 size={22} />
                <span>No render yet</span>
              </div>
            )}
          </div>
        </Card>
      </main>
    </div>
  );
}
