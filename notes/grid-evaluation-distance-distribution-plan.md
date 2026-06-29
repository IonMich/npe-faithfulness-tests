# Decay Grid-Evaluation Distance Distribution Plot Plan

Date: 2026-06-26

## Purpose

Create a separate plot showing the distribution of distances for observations in
an exponential-decay grid-evaluation panel.

This plan is only about the decay model:

```text
theta = (A, k, sigma)
x_j = A exp(-k t_j) + epsilon_j
epsilon_j ~ Normal(0, sigma^2)
```

It is not about the SBI tutorial Gaussian example.

For a global prior-predictive panel there is no privileged `x_0`. Therefore the
distance distribution should not be "distance away from `x_0`" unless the panel
is explicitly a local-`x_0` panel. The default global plot should describe where
the evaluated observations sit inside the declared evaluation distribution.

## Clarified Interpretation

The intended object is:

```text
{distance(x_i) for observations x_i that received grid references}
```

not:

```text
distance(x_i, x_0)
```

for the global case.

The plot should answer:

```text
Which parts of the evaluation distribution did the grid-reference panel cover?
```

It should be paired with, but separate from, plots of NPE-to-grid discrepancy.

## Current State

Relevant script:

```text
scripts/evaluate_decay_amortization_panel.py
```

Current panel distributions:

- `x0`
- `prior_predictive`
- `local_x`
- `parameter_region`

Current distance metadata:

- `local_x` records summary-space distance from the local region around `x_0`.
- `parameter_region` records prior-covariance Mahalanobis distance in
  `z = log(theta)` space from a declared parameter center.
- `prior_predictive` currently has no natural distance recorded because it is a
  global draw from the prior predictive.

Existing distance-sweep script:

```text
scripts/plot_npe_wasserstein_vs_distance.py
```

That script is anchored to the local q=0.005 region around `x_0`, so it is the
wrong object for a global prior-predictive grid-evaluation distance
distribution.

## Distance Definitions

### Primary global distance: prior z-score radius

For observations generated from prior draws, the simulator already knows the
true generating parameter:

```text
z_i = log(theta_i)
```

Use the prior-standardized radius:

```text
d_z(theta_i) =
sqrt(sum_j ((z_ij - prior_mean_j) / prior_std_j)^2)
```

This is not a distance from `x_0`. It is a distance from the prior center in
parameter space, measured in prior standard deviations. It tells us whether the
grid-evaluated observations came mostly from central prior mass or from tails.

### Secondary global distance: prior-predictive summary radius

Define summaries:

```text
s(x) = make_context_summaries(x)
```

Fit a pilot prior-predictive summary distribution:

```text
mu_s = mean(s(x))
Sigma_s = covariance(s(x)) or diagonal/ridge covariance
```

Then define:

```text
d_s(x_i) =
sqrt((s(x_i) - mu_s)^T Sigma_s^{-1} (s(x_i) - mu_s))
```

This is an observable-space distance. It is useful because a deployed NPE sees
`x`, not the true `theta`.

### Optional raw observation distance

For completeness, we can also compute a whitened observation-space radius:

```text
d_x(x_i) =
sqrt((x_i - mu_x)^T Sigma_x^{-1} (x_i - mu_x))
```

Because `x` is 40-dimensional in this model, this should be secondary. A
diagonal or ridge covariance is safer than an unregularized empirical inverse.

### Local-panel-only distance

For a `local_x` panel, keep the existing local metric:

```text
d_local(x_i) / radius
```

but label it explicitly as distance from the `x_0` local training region. Do not
reuse this label for prior-predictive global panels.

## Proposed Outputs

For a global prior-predictive panel:

```text
runs/01_exponential_decay/09_distance_distributions/
<run_name>/
  figures/grid_eval_distance_distribution.png
  figures/grid_eval_distance_distribution_by_metric.png
  results/grid_eval_distance_distribution.csv
  results/grid_eval_distance_distribution_summary.json
```

The main figure should include:

- histogram or ECDF of `d_z`;
- histogram or ECDF of `d_s`;
- optional vertical lines at median, q90, q95;
- optional overlay of a larger prior-predictive pilot distribution to show
  whether the evaluated grid panel is representative.

For local panels:

- include `d_local / radius`;
- show the training boundary at `1.0`;
- title and labels must explicitly say `local x_0 region`.

For parameter-region panels:

- include `d_z / parameter_radius`;
- show the declared parameter-region boundary at `1.0`.

## Implementation Plan

### Step 1: Add distance metadata to panel summaries

Extend `scripts/evaluate_decay_amortization_panel.py` so each observation row
records a `distance_metrics` dictionary.

For `prior_predictive`:

```json
{
  "prior_z_radius": 1.73,
  "summary_prior_predictive_radius": 2.41
}
```

For `local_x`:

```json
{
  "local_x0_distance": 0.42,
  "local_x0_distance_over_radius": 0.83,
  "prior_z_radius": 1.64,
  "summary_prior_predictive_radius": 1.98
}
```

For `parameter_region`:

```json
{
  "parameter_region_distance": 1.21,
  "parameter_region_distance_over_radius": 0.61,
  "prior_z_radius": 2.75,
  "summary_prior_predictive_radius": 3.10
}
```

### Step 2: Make prior-predictive summary whitening reusable

Add a helper that fits or loads pilot summary statistics:

```text
fit_prior_predictive_summary_metric(...)
```

Inputs:

- `pilot_simulations`
- `seed`
- `context_kind`
- `k_grid`
- `n_observations`

Outputs:

- summary mean;
- summary scale or covariance factor;
- pilot distance quantiles;
- pilot distribution samples for overlay.

Use diagonal/ridge-whitened covariance unless a full covariance is clearly
stable. Store enough metadata so future plots use the same distance definition.

### Step 3: Add a plotting script

Create:

```text
scripts/plot_decay_grid_eval_distance_distribution.py
```

Inputs:

```text
--panel-summary runs/.../decay_amortization_panel_summary.json
--output-dir runs/...
--pilot-simulations 100000
--metrics prior_z_radius,summary_prior_predictive_radius
--seed 20260626
```

The script should:

1. Load the panel summary.
2. Read or compute the requested distance metrics.
3. If pilot overlay is requested, draw a large prior-predictive pilot sample.
4. Save a CSV with one row per evaluated observation.
5. Save JSON with quantiles and plot metadata.
6. Save one compact figure.

### Step 4: Handle older panel summaries

Existing panel summaries may lack `distance_metrics`. The plotting script should
backfill what it can:

- If `z_true` is available, compute `prior_z_radius`.
- If `x` is available, compute `summary_prior_predictive_radius`.
- If only aggregate metrics are available, fail with a clear message asking for
  a panel rerun with distance metadata enabled.

### Step 5: Optional linkage with discrepancy

Keep the distance-distribution plot separate from Wasserstein-vs-distance. Then
optionally add a second figure:

```text
npe_discrepancy_vs_global_distance.png
```

This is useful, but it answers a different question:

```text
Does NPE error increase in the prior or prior-predictive tails?
```

The primary requested plot is the distribution of distances covered by the
grid-evaluation panel.

## Validation Plan

Smoke run on an existing prior-predictive panel:

```text
uv run scripts/plot_decay_grid_eval_distance_distribution.py \
  --panel-summary runs/01_exponential_decay/07_amortization_panels/00_smoke_stage1_prior/results/decay_amortization_panel_summary.json \
  --output-dir runs/01_exponential_decay/09_distance_distributions/00_smoke_stage1_prior
```

Checks:

- The figure title says prior-predictive/global when the panel is global.
- There is no `x_0` wording for `prior_predictive`.
- `prior_z_radius` values are finite and match the known generating `z_true`.
- Pilot overlay quantiles are saved.
- Local panels still label local distance as local-to-`x_0`.

## Deliverables

- Distance metrics in amortization panel summaries.
- New distance-distribution plotting script.
- One smoke artifact for an existing or rerun prior-predictive panel.
- A note documenting which distance metric is primary for global amortization
  coverage.

## Non-Goals

- Do not define global coverage by distance from `x_0`.
- Do not treat the distance distribution as an accuracy metric by itself.
- Do not replace observation-specific posterior discrepancy or tolerance ratios.
- Do not claim the grid panel covers the full infinite `x` support; summarize
  high-probability coverage under the declared evaluation distribution.
