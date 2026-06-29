# Single-Decay Local NPE Data-Scaling Plan

## Purpose

Measure how local single-decay NPE faithfulness changes as the number of
accepted local simulator pairs increases, while holding the statistical target,
local region, architecture, optimizer, reference posterior, and observed signal
fixed.

This is the repo's cleanest first scaling-law experiment because the
single-decay case has a calibrated grid/MCMC/HMC target and the existing local
flow already has near-target and passing runs.

## Quantity To Scale

The primary x-axis is accepted local training pairs:

```text
S = number of accepted local (theta, x) pairs used for NPE training
```

For local rejection sampling, simulator cost is not just `S`. The sweep records:

- `train_simulations`: accepted training pairs seen by the learner.
- `val_simulations`: fixed accepted validation pairs per seed.
- `pool_candidate_count`: total simulator proposals needed for the largest
  nested pool.
- `standalone_candidate_count_estimate`: estimated proposal count if the same
  scale had been collected by itself.

For unweighted hard-local runs, effective sample size equals accepted sample
count. If kernel weighting is later enabled, the x-axis should become weighted
ESS in addition to raw accepted count.

## Metrics

Primary metric:

```text
mean normalized Wasserstein to the grid posterior at x0
```

This is the repository's main posterior-faithfulness metric.

Secondary metrics:

- `target_ratio = Wasserstein / target_wasserstein`
- best validation NLL in standardized target coordinates
- best validation NLL adjusted back to target-z coordinates
- training seconds
- total seconds including local-data collection
- fitted power-law parameters when enough scale points exist

The validation NLL is useful for optimization scaling, but it is not a
faithfulness metric. Wasserstein remains the decision metric.

## Confounder Controls

Hold fixed across all scale points:

- observed signal `x0` via `--observed-seed`
- exact grid reference and target threshold
- local-region definition
- local context summary
- architecture: spline-flow transforms, hidden widths, bins
- optimizer, batch size, learning rate, patience, and maximum epochs
- posterior sample count used for evaluation
- validation set within each replicate seed

For each replicate seed:

1. Fit or load one local region.
2. Collect one accepted local pool of size `max(train_sizes) + val_size`.
3. Shuffle once.
4. Use nested training prefixes for every scale.
5. Use the same validation suffix for every scale.
6. Reinitialize the same architecture with the same seed for every scale in
   that replicate.

This isolates data scale better than comparing historical runs, which changed
local quantile, architecture, context, seed, and proposal strategy together.

## Recommended Full Sweep

```bash
uv run scripts/decay_local_scaling_sweep.py \
  --preset full \
  --output-root runs/01_exponential_decay/12_local_scaling/01_local_data_scaling
```

Full preset design:

```text
train_simulations = 10000, 20000, 40000, 80000, 150000, 300000
seeds             = 20260701, 20260702, 20260703, 20260704, 20260705
val_simulations   = 35000
local_quantile    = 0.005
context_kind      = indirect
transforms        = 8
hidden_features   = 192,192
bins              = 16
epochs            = 220
patience          = 55
batch_size        = 4096
posterior_samples = 100000
reference_grid    = 90
```

The expected runtime is substantial because the largest scale must collect a
large local pool and train 30 flows. Use the pilot preset first.

## Pilot And Smoke Commands

Pilot:

```bash
uv run scripts/decay_local_scaling_sweep.py \
  --preset pilot \
  --output-root runs/01_exponential_decay/12_local_scaling/00_pilot
```

Smoke:

```bash
uv run scripts/decay_local_scaling_sweep.py \
  --preset smoke \
  --output-root runs/01_exponential_decay/12_local_scaling/00_smoke
```

The smoke preset is only for software verification. Do not interpret its
scaling fit.

## Outputs

The sweep writes:

```text
<output-root>/results/local_data_scaling_rows.csv
<output-root>/results/local_data_scaling_summary.csv
<output-root>/results/local_data_scaling_summary.json
<output-root>/figures/local_data_scaling.png
<output-root>/runs/n<train>_seed<seed>/results/local_scaling_run_summary.json
```

The aggregate plot should show:

- Wasserstein versus accepted train simulations.
- target ratio versus accepted train simulations.
- validation NLL versus accepted train simulations.
- training seconds versus accepted train simulations.

## Fit Form

Fit the median Wasserstein across seeds with:

```text
W(S) = W_inf + A * S^(-alpha)
```

Only fit when there are at least three positive scale points. Treat this as a
diagnostic curve, not a law, until the full sweep has repeated seeds and clear
monotonic trend.

## Interpretation Rules

- If validation NLL improves but Wasserstein plateaus, the NPE objective is no
  longer aligned enough with posterior faithfulness at `x0`.
- If both validation NLL and Wasserstein plateau above target, the likely floor
  is architecture, summary, or local-objective bias.
- If Wasserstein continues to shrink with no visible floor, run the broad
  Chinchilla-style sweep next to separate data from parameter scale.
- Report uncertainty across seeds; do not report a single seed as a scaling law.
