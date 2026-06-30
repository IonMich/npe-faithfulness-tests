# Single-Decay Broad NPE Chinchilla-Style Scaling Plan

## Purpose

Measure how broad amortized single-decay NPE improves when simulation budget and
model size are scaled together or separately. This is the closest analogue to
LLM Chinchilla-style scaling in this repository:

```text
LLM tokens      -> simulator pairs (theta, x)
LLM parameters  -> NPE density-estimator parameters
LLM validation loss -> NPE validation NLL
downstream quality  -> posterior Wasserstein across held-out observations
```

The central question is whether the broad amortized map

```math
x \mapsto q_\phi(\theta \mid x)
```

is data-limited, model-limited, or dominated by a posterior-faithfulness floor.

## Quantity To Scale

Use two explicit axes:

```text
D = number of broad prior-predictive simulator pairs
P = trainable NPE parameter count
```

Track compute as:

```text
C_proxy = D * P
```

This is not exact FLOPs, but it is stable enough for within-repo comparisons.
Also record wall-clock training seconds.

## Metrics

Training/optimization metrics:

- best validation NLL
- final train/validation NLL gap
- epochs completed
- training seconds

Posterior-faithfulness metrics:

- mean normalized Wasserstein to grid posterior at the original `x0`
- median/mean/max Wasserstein across a fixed held-out observation panel
- target pass count across the panel
- target ratio distribution

The multi-observation panel is mandatory. A broad amortized estimator should not
be evaluated only at the original `x0`.

## Confounder Controls

Hold fixed:

- prior and simulator
- observed `x0`
- held-out observation panel
- grid-reference settings
- posterior sample count per observation
- optimizer family and learning-rate schedule
- batch-size rule
- random seeds per grid cell
- train/validation split protocol

Use the same held-out panel for every `(D, P)` cell. This avoids confusing
model/data scale with easier or harder observations.

## Architecture Grid

Start with MDN because the existing broad scaling result found it to be the best
cost/fidelity tradeoff among the tested broad families.

Suggested MDN grid:

```text
small:  hidden_dim=96,  hidden_layers=2, mdn_components=3
base:   hidden_dim=128, hidden_layers=3, mdn_components=5
large:  hidden_dim=192, hidden_layers=4, mdn_components=8
xlarge: hidden_dim=256, hidden_layers=5, mdn_components=12
```

After MDN is understood, optionally repeat with affine-flow models:

```text
flow_context_dim = 64, 96, 128
flow_layers      = 4, 6, 8, 10
```

Do not mix MDN and flow results in one fit. They are different model families
with different floors.

## Data Grid

Recommended full grid:

```text
D = 10000, 30000, 100000, 300000, 1000000
```

Use at least three seeds for each cell:

```text
seeds = 20260801, 20260802, 20260803
```

The first pilot can use:

```text
D = 10000, 30000, 100000
models = small, base, large
seeds = 20260801
```

## Compute-Optimal Analysis

For each cell, aggregate across seeds:

```text
median_validation_nll(D, P)
median_panel_wasserstein(D, P)
median_training_seconds(D, P)
```

Fit two surfaces separately:

```text
NLL(D, P) = NLL_inf + A * D^(-alpha) + B * P^(-beta)
W(D, P)   = W_inf   + C * D^(-gamma) + E * P^(-delta)
```

Then analyze constant-compute slices by minimizing fitted error subject to:

```text
D * P = constant
```

This gives the repository analogue of a compute-optimal Chinchilla curve. The
result should be reported as empirical for this simulator and implementation,
not as a universal rule.

## Implementation Sketch

Add a future script:

```text
scripts/decay_broad_chinchilla_sweep.py
```

Responsibilities:

1. Generate or load a fixed held-out observation panel.
2. For each `(D, architecture, seed)` cell, call the broad Stage 1 training
   logic with one family at a time.
3. Count trainable parameters from the constructed model.
4. Evaluate the trained model at original `x0`.
5. Evaluate the same model on the fixed held-out panel with grid references.
6. Write one per-cell JSON and one aggregate CSV/JSON.
7. Plot heatmaps of NLL and Wasserstein over `(D, P)`.
8. Plot compute slices of `D * P` versus panel Wasserstein.

Reuse `scripts/npe_stage1_decay.py` model families first, but avoid a large
subprocess-only implementation if possible. A direct Python implementation can
reuse fixed panels and avoid recomputing reference grids unnecessarily.

## Interpretation Rules

- If larger `D` helps at fixed `P`, the broad estimator is data-limited.
- If larger `P` helps at fixed `D`, it is capacity-limited.
- If NLL improves but Wasserstein does not, the density objective is improving
  away from posterior features that matter to the reference metric.
- If neither helps, revisit context representation, posterior family, or
  calibration diagnostics before spending more compute.

## Implemented Data-Scaling Pilot

Added a fixed-architecture broad-prior scaling sweep:

```bash
uv run scripts/decay_broad_scaling_sweep.py \
  --preset pilot \
  --output-root runs/01_exponential_decay/15_broad_scaling/01_mdn_pilot \
  --train-simulations 1000,2000,4000,8000,16000,32000,64000 \
  --skip-existing
```

Controls:

- MDN only, fixed architecture: `hidden_dim=128`, `hidden_layers=3`,
  `mdn_components=5`.
- Two replicate seeds: `20260901`, `20260902`.
- Nested broad prior-predictive training prefixes per seed.
- One fixed validation set with `12,000` prior-sampled `(theta, x)` pairs.
- One fixed standardization sample with `60,000` prior-sampled signals.
- `20,000` posterior samples at the original `x0`.
- W rescored against the cached `300^3` `x0` grid reference.

Outputs:

- `runs/01_exponential_decay/15_broad_scaling/01_mdn_pilot/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/01_mdn_pilot/results/broad_scaling_summary.csv`
- `runs/01_exponential_decay/15_broad_scaling/01_mdn_pilot/results/broad_scaling_rows.csv`
- `runs/01_exponential_decay/15_broad_scaling/01_mdn_pilot/figures/broad_scaling_law.png`
- `runs/01_exponential_decay/15_broad_scaling/01_mdn_pilot/figures/broad_scaling_log_excess.png`

Storage was small: `13 MiB` for 14 saved posterior sample files and summaries.

Median results:

| Broad train signals | x0 grid-300 W | Target ratio | Validation NLL in z units |
| ---: | ---: | ---: | ---: |
| 1,000 | 3.9153 | 229.09 | -0.3393 |
| 2,000 | 2.1211 | 124.11 | -0.8579 |
| 4,000 | 0.9688 | 56.69 | -1.8163 |
| 8,000 | 0.5937 | 34.74 | -2.2557 |
| 16,000 | 0.2635 | 15.42 | -2.4997 |
| 32,000 | 0.3424 | 20.04 | -2.6074 |
| 64,000 | 0.2378 | 13.91 | -2.8840 |

Fitted-floor power-law diagnostics:

| Metric | alpha | Raw R2 | Log-excess R2 |
| --- | ---: | ---: | ---: |
| x0 grid-300 W | 1.03 | 0.996 | 0.916 |
| Validation NLL in z units | 0.47 | 0.982 | 0.983 |
| Best validation NLL in z units | 0.47 | 0.982 | 0.983 |

Interpretation:

- The broad MDN definitely has a scaling signal in the small-data regime.
- Validation NLL is the cleaner scaling-law metric: the log-excess fit is
  close to linear and remains monotone through `64k`.
- `x0` Wasserstein improves dramatically from `1k` to `16k`, but it is
  seed-sensitive and non-monotone at `32k`/`64k`.
- Even after the feasible `64k` scale-up, median W is still `13.9x` the
  calibrated grid-300 target. This is much better than the smallest runs, but
  still far from target.

Next step: do not just spend blindly on more broad data. Use NLL as the
primary scaling-law curve, and improve the broad model's vertical offset or
exponent by testing larger MDNs, richer summaries/context, or flow/MDN
hybrids. W should remain an `x0` audit and should eventually be extended to a
held-out observation panel before claiming broad posterior faithfulness.

## Large Validation NLL Cache

For Kaplan-style scaling plots, the NLL y-axis should be averaged over many
held-out simulator pairs, analogous to cross-entropy over many held-out tokens.
This is separate from `W(x0)`, which is a single-observation posterior audit.

Implemented a reusable broad prior-predictive validation cache:

```bash
uv run scripts/cache_decay_broad_validation.py \
  --simulations 1000000 \
  --output runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz
```

Measured storage:

| Quantity | Value |
| --- | ---: |
| Validation pairs | 1,000,000 |
| `x_val` shape | `1000000 x 40` |
| `z_val` shape | `1000000 x 3` |
| Raw float32 arrays | 164.0 MiB |
| Compressed cache | 151.8 MiB |
| Cache generation + save | 5.43 s |

Measured NLL evaluation on the existing scaled broad MDN checkpoint:

```bash
uv run scripts/benchmark_decay_validation_nll_cache.py \
  --validation-cache runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz
```

Result:

| Quantity | Value |
| --- | ---: |
| Model parameters | 134,480 |
| Evaluated pairs | 1,000,000 |
| CPU evaluation time | 0.98 s |
| Throughput | 1.02M pairs/s |
| NLL in z units | -3.1310 |

So storage and final evaluation cost are both acceptable for `1M` validation
pairs.

Important caveat: using `1M` pairs for early stopping at every epoch is
wasteful. The sweep script now treats `--validation-cache` as the final
reported NLL evaluator only; early stopping still uses `--val-simulations`.

Recommended setup for broad NLL scaling:

```bash
uv run scripts/decay_broad_scaling_sweep.py \
  --preset pilot \
  --output-root runs/01_exponential_decay/15_broad_scaling/04_mdn_pilot_val100k_1m \
  --train-simulations 1000,2000,4000,8000,16000,32000,64000 \
  --val-simulations 100000 \
  --validation-cache runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz \
  --skip-existing
```

This uses:

- `100k` fixed prior-predictive pairs for checkpoint selection / early stopping.
- `1M` fixed prior-predictive pairs for the final plotted NLL.

The `100k` early-validation loss stayed smooth:

| Broad train signals | Best validation NLL in z units |
| ---: | ---: |
| 1,000 | 0.1878 |
| 2,000 | -0.3638 |
| 4,000 | -1.7412 |
| 8,000 | -2.2990 |
| 16,000 | -2.5401 |
| 32,000 | -2.6892 |
| 64,000 | -2.9188 |

Fitted-floor diagnostic:

```text
best validation NLL alpha ~= 0.511
raw R2 ~= 0.969
log-excess R2 ~= 0.975
```

The `1M` final mean NLL exposed rare tail failures in one seed/data prefix
around `32k`. This is not evaluation noise; density NLL is unbounded, and a
bad posterior tail can dominate the mean. Future scaling plots should report
the final NLL mean plus tail diagnostics such as median, q95, q99, q999, and
max. The sweep now records these summaries for future runs.

## Panel Marginal W References

The old `x0` W target:

```text
W_target(x0) ~= 0.01709
```

is valid only for the original `x0` audit. It came from MCMC/HMC/grid agreement
for that one observed signal. It should not be reused as a universal target for
other signals.

For broad NPE scaling, the W metric should be a panel average:

```text
panel_W = mean_i W_marginal(q_phi(theta | x_i), p(theta | x_i))
```

where `x_i` are fixed held-out prior-predictive signals. This is still the same
kind of marginal-W metric used for `x0`: mean over `A`, `k`, and `sigma` 1D
Wasserstein distances normalized by the reference posterior marginal sd. The
difference is that we average over many signals instead of one signal.

To avoid storing full grids, I added a marginal reference cache:

```bash
uv run scripts/cache_decay_panel_marginals.py \
  --output runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/decay_panel16_grid180_marginals.npz \
  --panel-size 16 \
  --grid-size 180 \
  --target-sample-count 20000 \
  --target-repeats 5 \
  --force
```

Measured cache size:

| Quantity | Value |
| --- | ---: |
| Held-out signals | 16 |
| Grid size per signal | `180^3` |
| Full grids stored | 0 |
| Cached marginal file | 0.063 MiB |
| Build time | 31.6 s |

Each signal stores only:

```text
theta_axes[signal, parameter, grid_index]
marginal_weights[signal, parameter, grid_index]
target_wasserstein[signal]
```

The per-signal target is not the `x0` MCMC/HMC target. It is a numerical
evaluation floor: the q84 W obtained by drawing exact-grid posterior samples
and comparing them back to the cached marginal reference. For the current
`panel16/grid180/20k-sample` cache, these target floors are roughly
`0.008-0.013`.

The broad sweep now accepts:

```bash
--panel-marginal-cache runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/decay_panel16_grid180_marginals.npz
```

When supplied, the top row of `broad_scaling_law.png` changes from `x0` W to:

```text
panel mean marginal W
panel mean target ratio
```

The `x0` W and its `0.01709` target remain in the run JSON as an audit, but no
longer define the main broad scaling-law W plot.

Before final scale-up, use this combined setup:

```bash
uv run scripts/decay_broad_scaling_sweep.py \
  --preset pilot \
  --output-root runs/01_exponential_decay/15_broad_scaling/<next-run> \
  --train-simulations ... \
  --val-simulations 100000 \
  --validation-cache runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz \
  --panel-marginal-cache runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/decay_panel16_grid180_marginals.npz \
  --panel-posterior-samples 20000 \
  --skip-existing
```

This fixes the earlier plot issues:

- the main W plot is no longer `x0`-only;
- the W target ratio is calibrated per signal, then averaged;
- the `0.01709` target is retained only for the `x0` audit;
- the NLL scaling check should be read from the fitted-floor log-excess plot,
  not raw NLL versus `log D`;
- final `1M` NLL mean should be reported with tail summaries because rare
  density failures can dominate the mean.

## Moderate Corrected Scale-Up

Ran one moderate scale-up with the corrected metric setup before choosing the
large sweep:

```bash
uv run scripts/decay_broad_scaling_sweep.py \
  --preset pilot \
  --output-root runs/01_exponential_decay/15_broad_scaling/06_mdn_moderate_panel128k \
  --train-simulations 8000,16000,32000,64000,128000 \
  --seeds 20260901,20260902 \
  --val-simulations 100000 \
  --validation-cache runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz \
  --panel-marginal-cache runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/decay_panel16_grid180_marginals.npz \
  --panel-posterior-samples 20000 \
  --skip-existing
```

Outputs:

- `runs/01_exponential_decay/15_broad_scaling/06_mdn_moderate_panel128k/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/06_mdn_moderate_panel128k/results/broad_scaling_summary.csv`
- `runs/01_exponential_decay/15_broad_scaling/06_mdn_moderate_panel128k/figures/broad_scaling_law.png`
- `runs/01_exponential_decay/15_broad_scaling/06_mdn_moderate_panel128k/figures/broad_scaling_log_excess.png`

Median results:

| Broad train signals | Panel mean W | Panel mean target ratio | x0 W audit | x0 target ratio | Final 1M NLL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,000 | 0.9544 | 89.71 | 0.8472 | 49.57 | -2.2190 |
| 16,000 | 0.7241 | 67.57 | 0.2672 | 15.64 | -2.4984 |
| 32,000 | 0.6470 | 60.31 | 0.2959 | 17.31 | -2.4837 |
| 64,000 | 0.4819 | 44.94 | 0.2310 | 13.52 | -2.9600 |
| 128,000 | 0.3901 | 36.19 | 0.1419 | 8.30 | -3.1874 |

The corrected panel W curve is monotone and much cleaner than the old `x0`
curve:

```text
panel mean W alpha ~= 0.307
raw R2 ~= 0.986
log-excess R2 ~= 0.987
```

The `x0` audit also improves by `128k`, but it remains an audit and is not the
primary broad metric. At `128k`, the two seeds agree well:

| Seed | Panel mean W | Panel mean target ratio | x0 W | Final 1M NLL |
| ---: | ---: | ---: | ---: | ---: |
| 20260901 | 0.3969 | 36.53 | 0.1499 | -3.1914 |
| 20260902 | 0.3834 | 35.86 | 0.1338 | -3.1835 |

NLL tail diagnostics at `128k` were no longer pathological. The worst per-pair
NLL values were finite but not mean-dominating:

| Seed | q99 | q999 | max |
| ---: | ---: | ---: | ---: |
| 20260901 | 3.26 | 7.55 | 74.42 |
| 20260902 | 2.99 | 7.13 | 308.92 |

Decision for the next large sweep:

- Do not spend on an extreme single-point scale-up yet.
- The corrected panel-W curve is still `~36x` target at `128k`, so a data-only
  leap to target is unrealistic for this fixed MDN.
- The next useful large run should separate data scaling from robustness:

```text
D = 64k, 128k, 256k, 512k
seeds = 3-5
same MDN base architecture
same 100k early-validation and 1M final-NLL cache
same panel16/grid180 marginal W cache
```

This will answer whether the panel-W exponent stays near `0.3` and whether the
floor remains far above target. If the 512k point is still many multiples above
target, the following run should improve model family/capacity rather than only
adding more broad signals.

## Corrected Scale-Up To 512k

Ran the corrected panel-W / 1M-NLL sweep through `512k` broad prior-predictive
training signals:

```bash
uv run scripts/decay_broad_scaling_sweep.py \
  --preset pilot \
  --output-root runs/01_exponential_decay/15_broad_scaling/07_mdn_panel512k \
  --train-simulations 64000,128000,256000,512000 \
  --seeds 20260901,20260902,20260903 \
  --val-simulations 100000 \
  --validation-cache runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz \
  --panel-marginal-cache runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/decay_panel16_grid180_marginals.npz \
  --panel-posterior-samples 20000 \
  --skip-existing
```

Outputs:

- `runs/01_exponential_decay/15_broad_scaling/07_mdn_panel512k/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/07_mdn_panel512k/results/broad_scaling_summary.csv`
- `runs/01_exponential_decay/15_broad_scaling/07_mdn_panel512k/figures/broad_scaling_law.png`
- `runs/01_exponential_decay/15_broad_scaling/07_mdn_panel512k/figures/broad_scaling_log_excess.png`

Run storage was still small: `11 MiB` for summaries and saved posterior
samples. The large reusable caches remain the separate `1M` validation cache
and the `x0` grid cache.

Median results:

| Broad train signals | Panel mean W | Panel mean target ratio | x0 W audit | x0 target ratio | Final 1M NLL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64,000 | 0.5354 | 49.78 | 0.2552 | 14.93 | -2.9334 |
| 128,000 | 0.3759 | 35.41 | 0.3044 | 17.81 | -3.1760 |
| 256,000 | 0.2965 | 27.73 | 0.1829 | 10.70 | -3.3437 |
| 512,000 | 0.2340 | 21.96 | 0.1462 | 8.56 | -3.4108 |

The corrected panel metric improved monotonically. The old `x0` audit is still
seed-sensitive and should not be used as the broad scaling-law y-axis.

Fitted-floor diagnostics for this `64k-512k` window:

| Metric | Fitted floor | alpha | Raw R2 | Log-excess R2 |
| --- | ---: | ---: | ---: | ---: |
| Panel mean W | 0.1615 | 0.769 | 0.999 | 0.997 |
| Final 1M NLL | -3.5341 | 0.786 | 0.998 | 0.995 |
| Best 100k-val NLL | -3.5480 | 0.729 | 0.997 | 0.993 |

Important interpretation:

- The `512k` panel ratio `21.96` is close to the earlier no-floor extrapolated
  expectation of about `23`, so the moderate-run scaling trend did survive one
  larger scale-up.
- A direct no-floor fit to panel target ratio over `64k-512k` gives
  `ratio ~= 3593 * D^-0.389`.
- Bootstrap planning intervals from resampling the three seeds per `D` give
  approximate `90%` intervals:

| Target panel ratio | Point estimate D | 90% bootstrap interval |
| ---: | ---: | ---: |
| 20 | 0.61M | 0.44M-0.76M |
| 15 | 1.29M | 0.82M-1.73M |
| 10 | 3.64M | 1.94M-6.21M |
| 5 | 21.6M | 8.19M-55.1M |
| 2 | 227M | 51.6M-988M |
| 1 | 1.35B | 213M-8.77B |

These intervals are useful for planning only. They assume the no-floor ratio
trend continues, which is doubtful near target. The fitted-floor W model implies
a floor around `0.1615`, or about `15x` the current panel target floor. That is
not proof of a true asymptote from only four data points, but it is a strong
warning that the fixed base MDN may hit a model/objective floor before the
panel target.

Decision:

- A data-only run to `1M-2M` is justified if the next question is whether the
  ratio can cross `20` and approach `15`.
- A data-only run to target is not justified. The extrapolated cost is too high
  and the fitted-floor warning is too strong.
- Before any much larger run, add a model/capacity axis: larger MDN and/or flow
  family at fixed `256k-512k`, using the same panel marginal cache and `1M` NLL
  cache. If the floor/vertical offset improves, then scale that better family.

## MDN Parameter-Axis Pilot

Started the Kaplan-style parameter axis by holding the broad simulator budget
small enough to make larger models feasible. The base `512k` data-scaling run
already provided the `44,722` parameter anchor. I added three larger MDNs at
`64k` broad training signals:

| Label | Hidden | Layers | Components | Parameters |
| --- | ---: | ---: | ---: | ---: |
| base | 128 | 3 | 5 | 44,722 |
| large | 192 | 4 | 8 | 134,480 |
| xlarge | 256 | 5 | 12 | 304,504 |
| xxlarge | 384 | 5 | 16 | 668,704 |

Example larger-model command:

```bash
uv run scripts/decay_broad_scaling_sweep.py \
  --preset pilot \
  --output-root runs/01_exponential_decay/15_broad_scaling/08_mdn_param_axis/large \
  --train-simulations 64000 \
  --seeds 20260901,20260902,20260903 \
  --val-simulations 100000 \
  --validation-cache runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz \
  --panel-marginal-cache runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/decay_panel16_grid180_marginals.npz \
  --panel-posterior-samples 20000 \
  --hidden-dim 192 \
  --hidden-layers 4 \
  --mdn-components 8 \
  --skip-existing
```

Added an aggregate plotting script:

```bash
uv run scripts/plot_broad_param_scaling.py \
  --output-root runs/01_exponential_decay/15_broad_scaling/08_mdn_param_axis \
  runs/01_exponential_decay/15_broad_scaling/07_mdn_panel512k/results/broad_scaling_summary.json \
  runs/01_exponential_decay/15_broad_scaling/08_mdn_param_axis/large/results/broad_scaling_summary.json \
  runs/01_exponential_decay/15_broad_scaling/08_mdn_param_axis/xlarge/results/broad_scaling_summary.json \
  runs/01_exponential_decay/15_broad_scaling/08_mdn_param_axis/xxlarge/results/broad_scaling_summary.json
```

Outputs:

- `runs/01_exponential_decay/15_broad_scaling/08_mdn_param_axis/results/broad_param_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/08_mdn_param_axis/results/broad_param_scaling_summary.csv`
- `runs/01_exponential_decay/15_broad_scaling/08_mdn_param_axis/figures/broad_param_scaling.png`

At `D=64k`, capacity improves validation NLL monotonically, but panel W improves
only weakly and non-monotonically:

| D | Parameters | Panel mean W | Panel target ratio | Final 1M NLL | Median train seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64,000 | 44,722 | 0.5354 | 49.78 | -2.9334 | 54.1 |
| 64,000 | 134,480 | 0.4397 | 41.23 | -3.0431 | 58.6 |
| 64,000 | 304,504 | 0.4659 | 43.42 | -3.0781 | 112.9 |
| 64,000 | 668,704 | 0.4359 | 40.59 | -3.1156 | 153.8 |

Then tested whether the `P` benefit persists at `D=128k` before paying for the
full larger grid:

| D | Parameters | Panel mean W | Panel target ratio | Final 1M NLL | Median train seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128,000 | 44,722 | 0.3759 | 35.41 | -3.1760 | 68.0 |
| 128,000 | 134,480 | 0.3827 | 35.81 | -3.1730 | 346.3 |

Decision:

- There is a real parameter-axis signal in NLL at `64k`.
- The panel-W signal from more MDN parameters is weak at `64k` and absent at
  `128k` for the first larger model.
- The `128k` larger model is about `5x` slower than the base model and did not
  improve the main posterior-faithfulness metric. This makes a full
  `128k x {304k, 669k}` extension low-value for the current base MDN family.
- Next useful step is not simply bigger MDNs. Compare other posterior families
  at modest budgets, especially affine flows, while keeping the same
  `panel16/grid180` W cache and `1M` NLL cache.

## Saved UI Checkpoint

Changed `scripts/decay_broad_scaling_sweep.py` so model checkpoints are saved by
default. Use `--no-save-models` for metric-only sweeps.

Saved the best corrected broad MDN seed for UI inspection:

```bash
uv run scripts/decay_broad_scaling_sweep.py \
  --preset pilot \
  --output-root runs/01_exponential_decay/15_broad_scaling/09_ui_best_broad_mdn_512k_seed20260902 \
  --train-simulations 512000 \
  --seeds 20260902 \
  --val-simulations 100000 \
  --validation-cache runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz \
  --panel-marginal-cache runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/decay_panel16_grid180_marginals.npz \
  --panel-posterior-samples 20000 \
  --skip-existing
```

Saved checkpoint:

```text
runs/01_exponential_decay/15_broad_scaling/09_ui_best_broad_mdn_512k_seed20260902/runs/n512000_seed20260902/results/mdn_model.pt
```

Metrics reproduced the previous best seed:

| Metric | Value |
| --- | ---: |
| Panel mean W | 0.2184 |
| Panel target ratio | 20.31 |
| Final 1M NLL | -3.4329 |
| x0 audit W | 0.1462 |
| x0 audit target ratio | 8.56 |

The posterior viewer now exposes this checkpoint as:

```text
broad_mdn_512k
```

alongside the existing `local_flow` entry. The older `broad_mdn` can still be
passed explicitly to the viewer, but it is no longer shown by default.

## Affine-Flow Family Pilot

Before scaling other families heavily, ran a small `D x P` pilot for the broad
affine-flow family. This family was the next most plausible candidate because
it can represent non-Gaussian continuous posteriors with invertible
transformations, whereas larger MDNs improved NLL but did not materially improve
panel W.

Flow P grid:

| Label | Hidden | Context | Flow layers | Parameters |
| --- | ---: | ---: | ---: | ---: |
| small | 64 | 32 | 4 | 19,640 |
| base | 96 | 64 | 6 | 127,300 |
| large | 128 | 96 | 8 | 291,344 |

Each flow used:

```text
D = 16k, 64k
seeds = 20260901, 20260902
same 100k early-validation set
same 1M final-NLL validation cache
same panel16/grid180 marginal W cache
```

Example command:

```bash
uv run scripts/decay_broad_scaling_sweep.py \
  --preset pilot \
  --output-root runs/01_exponential_decay/15_broad_scaling/10_affine_flow_param_axis/flow_large \
  --family affine_flow \
  --train-simulations 16000,64000 \
  --seeds 20260901,20260902 \
  --val-simulations 100000 \
  --validation-cache runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz \
  --panel-marginal-cache runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/decay_panel16_grid180_marginals.npz \
  --panel-posterior-samples 20000 \
  --hidden-dim 128 \
  --hidden-layers 3 \
  --flow-layers 8 \
  --flow-context-dim 96 \
  --skip-existing
```

Aggregate outputs:

- `runs/01_exponential_decay/15_broad_scaling/10_affine_flow_param_axis/results/broad_param_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/10_affine_flow_param_axis/results/broad_param_scaling_summary.csv`
- `runs/01_exponential_decay/15_broad_scaling/10_affine_flow_param_axis/figures/broad_param_scaling.png`

Affine-flow median results:

| D | Parameters | Panel mean W | Panel target ratio | Final 1M NLL | Median train seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16,000 | 19,640 | 0.7569 | 71.25 | -2.5297 | 14.9 |
| 16,000 | 127,300 | 0.9432 | 87.98 | -2.4587 | 35.1 |
| 16,000 | 291,344 | 0.9211 | 86.46 | -2.5236 | 59.5 |
| 64,000 | 19,640 | 0.5398 | 50.42 | -2.8926 | 49.3 |
| 64,000 | 127,300 | 0.5436 | 50.96 | -2.8410 | 94.6 |
| 64,000 | 291,344 | 0.5037 | 46.88 | -3.0237 | 193.6 |

Same-seed base-MDN reference from the corrected moderate run:

| D | Parameters | Panel mean W | Panel target ratio | Final 1M NLL | Median train seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16,000 | 44,722 | 0.7241 | 67.57 | -2.4984 | 16.7 |
| 64,000 | 44,722 | 0.4819 | 44.94 | -2.9600 | 42.4 |

Interpretation:

- Affine flows do not beat the base MDN on panel W at `16k` or `64k`.
- The largest flow improves NLL at `64k` relative to the base MDN
  (`-3.0237` vs `-2.9600`), but panel W remains worse (`46.88x` vs `44.94x`)
  and runtime is about `4.6x` higher.
- Increasing flow parameters is not reliably helpful at `16k`; the larger
  flows are worse than the small flow on both panel W and NLL.
- At these budgets, affine flow is not the next family to scale deeply if the
  main target is panel posterior faithfulness. It may still be useful for NLL,
  but that is not the limiting metric.

## Mac Mini Affine-Flow Remote Probe

After adding the structured remote train endpoint, reran the base affine-flow
probe on the Mac mini and duplicated it locally for a speed/control check.

Common settings:

```text
family = affine_flow
hidden_dim = 96
hidden_layers = 3
flow_layers = 6
flow_context_dim = 64
D = 16k,64k
seeds = 20260901,20260902
jobs = 2
torch_threads = 2
device = cpu
same 1M final-NLL validation cache
same panel16/grid180 marginal W cache
```

Outputs:

- Mini affine-flow: `runs/01_exponential_decay/15_broad_scaling/22_mini_affine_flow_base_probe/results/broad_scaling_summary.json`
- Local affine-flow duplicate: `runs/01_exponential_decay/15_broad_scaling/23_local_affine_flow_base_probe/results/broad_scaling_summary.json`
- Mini MDN baseline: `runs/01_exponential_decay/15_broad_scaling/24_mini_mdn_base_probe/results/broad_scaling_summary.json`

Mini base affine-flow medians:

| D | Panel mean W | Panel target ratio | Final 1M NLL | Train seconds |
| ---: | ---: | ---: | ---: | ---: |
| 16,000 | 0.7398 | 69.12 | -2.5069 | 21.2 |
| 64,000 | 0.5621 | 52.64 | -2.8667 | 52.8 |

Local duplicate base affine-flow medians:

| D | Panel mean W | Panel target ratio | Final 1M NLL | Train seconds |
| ---: | ---: | ---: | ---: | ---: |
| 16,000 | 0.9432 | 87.98 | -2.4587 | 34.2 |
| 64,000 | 0.5436 | 50.96 | -2.8410 | 90.5 |

Mini base MDN medians:

| D | Panel mean W | Panel target ratio | Final 1M NLL | Train seconds |
| ---: | ---: | ---: | ---: | ---: |
| 16,000 | 0.8352 | 77.98 | -2.3763 | 7.5 |
| 64,000 | 0.4983 | 46.76 | -2.8452 | 23.5 |

Interpretation:

- The Mac mini is faster than local for the affine-flow run by about `1.6x`
  at `16k` and `1.7x` at `64k`.
- On the same Mac mini, the base affine flow is not faster than the base MDN:
  it is about `2.8x` slower at `16k` and `2.2x` slower at `64k`.
- The affine flow slightly improves NLL on the mini, especially at `16k`, but
  it does not improve the panel W metric at `64k`; the mini MDN has better
  panel W there.
- This reinforces the earlier local conclusion: affine flow is not the next
  obvious family to scale deeply for panel-W faithfulness. If the goal is NLL,
  a larger or more expressive flow may still be worth a targeted probe, but it
  is not a speed win.

## Broad Spline-Flow Plan

Next family to test: a broad prior-predictive conditional spline flow using the
existing `zuko.flows.NSF` machinery, now exposed as the Stage-1 `spline_flow`
family. Unlike the local flow, this uses the same full signal `x` input,
standardization, broad prior-predictive training pairs, 1M validation-NLL
cache, panel marginal W cache, checkpoint format, and Stage-1 UI loader path as
the MDN and affine-flow broad sweeps.

Initial spline-flow P grid:

| Label | Hidden | Hidden layers | NSF transforms | Spline bins | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 64 | 2 | 4 | 8 | 45,844 |
| base | 96 | 2 | 6 | 8 | 121,374 |
| large | 144 | 2 | 8 | 8 | 297,768 |

Run first:

```text
D = 16k,64k
seeds = 20260901,20260902
jobs = 2
torch_threads = 2
device = cpu
same 1M NLL cache
same panel16/grid180 marginal W cache
```

Compare against:

- mini base MDN at `16k/64k`;
- affine-flow base probe at `16k/64k`;
- the existing local affine-flow P-axis result, as secondary context only
  because same-seed training values are machine-local rather than bitwise
  comparable across M2/M4.

Decision rule:

- If spline flow improves panel target ratio at `64k` relative to mini MDN
  without an excessive runtime penalty, scale spline flow deeper.
- If it only improves NLL while W remains worse, keep it as an NLL candidate
  but do not make it the next W-focused scale-up family.

## Broad Spline-Flow Pilot Results

Implemented the Stage-1 `spline_flow` family and ran the requested small
Mac-mini sweep:

```text
family = spline_flow
D = 16k,64k
seeds = 20260901,20260902
jobs = 2
torch_threads = 2
device = cpu
same 100k early-validation set
same 1M final-NLL validation cache
same panel16/grid180 marginal W cache
```

Local mirrored outputs contain only summary JSON/CSV/figures, not full remote
checkpoints:

- `runs/01_exponential_decay/15_broad_scaling/25_mini_spline_flow_param_axis/spline_small/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/25_mini_spline_flow_param_axis/spline_base/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/25_mini_spline_flow_param_axis/spline_large/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/25_mini_spline_flow_param_axis/spline_param_axis/results/broad_param_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/25_mini_spline_flow_param_axis/spline_param_axis/figures/broad_param_scaling.png`

Same-machine comparison against the mini MDN and mini affine-flow baselines:

| Model | Parameters | D | Panel mean W | Panel target ratio | Final 1M NLL | Median train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MDN base | 44,722 | 16,000 | 0.8352 | 77.98 | -2.3763 | 7.5 |
| MDN base | 44,722 | 64,000 | 0.4983 | 46.76 | -2.8452 | 23.5 |
| Affine-flow base | 127,300 | 16,000 | 0.7398 | 69.12 | -2.5069 | 21.2 |
| Affine-flow base | 127,300 | 64,000 | 0.5621 | 52.64 | -2.8667 | 52.8 |
| Spline-flow small | 45,844 | 16,000 | 0.4604 | 43.21 | -2.7878 | 19.8 |
| Spline-flow small | 45,844 | 64,000 | 0.3219 | 30.25 | -3.2208 | 42.2 |
| Spline-flow base | 121,374 | 16,000 | 0.5940 | 55.39 | -2.7469 | 19.2 |
| Spline-flow base | 121,374 | 64,000 | 0.3899 | 36.53 | -3.0798 | 51.7 |
| Spline-flow large | 297,768 | 16,000 | 0.6978 | 65.23 | -2.6286 | 30.9 |
| Spline-flow large | 297,768 | 64,000 | 0.3635 | 34.30 | -3.0703 | 65.0 |

Interpretation:

- The small spline flow is clearly better than the same-machine MDN and
  affine-flow baselines on both panel W and NLL at `16k` and `64k`.
- The improvement is not just an NLL artifact. At `64k`, the small spline
  improves the panel target ratio from the MDN's `46.76x` to `30.25x`.
- Scaling spline-flow parameters upward at fixed `D = 16k,64k` does not help.
  The base and large spline flows are slower and worse than the small spline
  on the main panel-W metric.
- The likely next useful spline experiment is data scaling for the small
  spline-flow architecture, not a larger-`P` spline sweep. A direct next run is
  `D = 64k,128k,256k,512k`, `seeds = 20260901,20260902,20260903`, same caches.

## Broad Spline-Flow D-Axis Scale-Up

Ran the recommended medium data-axis scale-up for the small spline-flow
architecture:

```text
family = spline_flow
hidden_dim = 64
hidden_layers = 2
flow_layers = 4
spline_bins = 8
parameters = 45,844
D = 64k,128k,256k,512k
seeds = 20260901,20260902,20260903
jobs = 2
torch_threads = 2
device = cpu
same 100k early-validation set
same 1M final-NLL validation cache
same panel16/grid180 marginal W cache
```

Outputs:

- `runs/01_exponential_decay/15_broad_scaling/28_mini_spline_flow_small_panel512k/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/28_mini_spline_flow_small_panel512k/results/broad_scaling_summary.csv`
- `runs/01_exponential_decay/15_broad_scaling/28_mini_spline_flow_small_panel512k/figures/broad_scaling_law.png`
- `runs/01_exponential_decay/15_broad_scaling/28_mini_spline_flow_small_panel512k/figures/broad_scaling_log_excess.png`

Local mirrored storage was small: `848 KiB` for summaries and figures only.
Full checkpoints remain on the Mac mini.

Median results:

| D | Panel mean W | Panel target ratio | Final 1M NLL | Median train seconds |
| ---: | ---: | ---: | ---: | ---: |
| 64,000 | 0.3769 | 36.00 | -3.1712 | 37.1 |
| 128,000 | 0.2611 | 24.72 | -3.3048 | 81.9 |
| 256,000 | 0.2340 | 21.81 | -3.3968 | 167.7 |
| 512,000 | 0.1880 | 17.48 | -3.4847 | 439.4 |

Per-seed 512k results:

| Seed | Panel mean W | Panel target ratio | Final 1M NLL |
| ---: | ---: | ---: | ---: |
| 20260901 | 0.1826 | 17.21 | -3.4706 |
| 20260902 | 0.2218 | 20.95 | -3.4847 |
| 20260903 | 0.1880 | 17.48 | -3.4925 |

Fitted-floor diagnostics:

| Metric | Fitted floor | Alpha | Raw R2 | Log-excess R2 |
| --- | ---: | ---: | ---: | ---: |
| Panel mean W | 0.1733 | 1.064 | 0.982 | 0.931 |
| Final 1M NLL | -3.7956 | 0.331 | 0.999 | 0.999 |
| Best 100k-val NLL | -3.7811 | 0.340 | 0.999 | 0.999 |

Interpretation:

- The small spline flow has a clear data-axis scaling signal over
  `64k-512k`. NLL is extremely smooth, and panel W improves from `36.0x` to
  `17.5x` target.
- At `256k`, the small spline roughly matches the earlier base-MDN `512k`
  panel ratio. At `512k`, it beats that MDN point.
- The fitted panel-W floor is close to the largest observed point because four
  D values are not enough to separate a real asymptote from local curvature.
  Read the W fit as a warning, not as proof that the spline cannot improve
  past `~17x`.
- The earlier P-axis result does not contradict this. For spline flows, adding
  transforms/width at `16k/64k` changed optimization and model family behavior,
  and did not help under the fixed optimizer/epoch budget. The small spline is
  data-limited in this window; larger splines may need different optimization
  or larger D before they become useful.

Decision:

- The next W-focused run should extend the same small spline to `1M` and
  possibly `2M`, or add a serial reproducibility check at `64k/128k` first if
  exact same-seed repeatability matters.
- Do not spend on larger spline-flow parameter sweeps until the small spline
  data curve bends clearly or optimization settings are revisited.

## Broad Spline-Flow Extension To 2M

Extended the small spline-flow D-axis on the Mac mini:

```text
family = spline_flow
hidden_dim = 64
hidden_layers = 2
flow_layers = 4
spline_bins = 8
parameters = 45,844
D = 1,024,000 and 2,048,000
seeds = 20260901,20260902
jobs = 2
torch_threads = 2
device = cpu
same 100k early-validation set
same 1M final-NLL validation cache
same panel16/grid180 marginal W cache
```

A simultaneous local/MacBook run for seeds `20260903,20260904` was stopped
before completion because it was much slower than the Mac mini. The local
partial run is excluded from all metrics below. Future intensive broad scaling
runs should use the Mac mini unless there is a specific reason to collect
machine-local controls.

Outputs:

- `runs/01_exponential_decay/15_broad_scaling/29_spline_flow_small_d2m_both_machines/mini/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/29_spline_flow_small_d2m_both_machines/mini/results/broad_scaling_summary.csv`
- `runs/01_exponential_decay/15_broad_scaling/29_spline_flow_small_d2m_both_machines/mini/figures/broad_scaling_law.png`

Mini-only median results, combining with the previous Mac-mini `64k-512k`
small-spline sweep for context:

| D | Seeds | Panel mean W | Panel target ratio | Final 1M NLL | Median train seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64,000 | 3 | 0.3769 | 36.00 | -3.1712 | 37.1 |
| 128,000 | 3 | 0.2611 | 24.72 | -3.3048 | 81.9 |
| 256,000 | 3 | 0.2340 | 21.81 | -3.3968 | 167.7 |
| 512,000 | 3 | 0.1880 | 17.48 | -3.4847 | 439.4 |
| 1,024,000 | 2 | 0.1823 | 17.11 | -3.4929 | 490.3 |
| 2,048,000 | 2 | 0.1779 | 16.78 | -3.5103 | 1089.2 |

Raw 1M/2M rows:

| Seed | D | Panel mean W | Panel target ratio | Final 1M NLL | Train seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20260901 | 1,024,000 | 0.1722 | 16.02 | -3.4939 | 499.1 |
| 20260902 | 1,024,000 | 0.1925 | 18.21 | -3.4918 | 481.5 |
| 20260901 | 2,048,000 | 0.1719 | 16.12 | -3.5080 | 1234.7 |
| 20260902 | 2,048,000 | 0.1839 | 17.44 | -3.5126 | 943.6 |

The no-floor target-ratio extrapolation worsened after adding `1M/2M`:

```text
ratio ~= 323.5 * D^-0.212
log-space R2 ~= 0.855
```

A fitted-floor curve explains the extended data much better:

| Metric | Fitted floor | Alpha | Raw R2 | Log-excess R2 |
| --- | ---: | ---: | ---: | ---: |
| Panel target ratio | 16.22 | 1.095 | 0.990 | 0.965 |
| Panel mean W | 0.1722 | 1.052 | 0.988 | 0.977 |
| Final 1M NLL | -3.5540 | 0.686 | 0.991 | 0.965 |

Interpretation:

- The spline-flow D-axis improved strongly up to `512k`, but the W metric
  mostly flattened from `512k` to `2M`: `17.48x -> 17.11x -> 16.78x`.
- NLL still improves with data, so the density objective has not completely
  saturated. The posterior-W metric appears to be hitting either a model,
  objective, panel-reference, or sampling/optimization floor.
- A blind jump above `2M` is not the best next step. If we go higher, it should
  be a targeted Mac-mini-only diagnostic such as one `4M` seed to test the
  floor, not a broad multi-seed scale-up.
- Before spending heavily above `2M`, better candidates are: improve/revisit
  optimization for the larger spline, increase posterior samples for W at
  `1M/2M` to check W noise, or compare an additional posterior family.

## Fixed-P 4M Diagnostic

Ran one Mac-mini-only diagnostic seed at `D=4,096,000` for the two comparable
fixed-parameter broad NPEs:

```text
seed = 20260901
validation cache = broad_prior_val_1m_float32.npz
panel W cache = decay_panel16_grid180_marginals.npz
posterior samples per panel signal = 20,000
jobs = 1 per family, launched concurrently on the Mac mini
device = cpu
torch_threads = 2
```

Outputs:

- `runs/01_exponential_decay/15_broad_scaling/34_mini_fixed_p_4m_diagnostic/mdn/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/34_mini_fixed_p_4m_diagnostic/spline/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/31_mdn_vs_spline_fixed_p_d_scaling/figures/mdn_vs_spline_fixed_p_2x2_4m.png`
- `runs/01_exponential_decay/15_broad_scaling/31_mdn_vs_spline_fixed_p_d_scaling/results/mdn_vs_spline_fixed_p_summary_4m.json`

| Family | Parameters | D | Seeds | Panel mean W | Panel target ratio | Final 1M NLL | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MDN base | 44,722 | 4,096,000 | 1 | 0.1817 | 16.89 | -3.5141 | 1371.1 |
| Spline flow small | 45,844 | 4,096,000 | 1 | 0.1283 | 12.02 | -3.5431 | 2482.4 |

Context against the preceding fixed-P results:

- MDN does not improve in W from its `2.048M` median to this single `4.096M`
  seed: target ratio `16.60x -> 16.89x`. Its NLL still improves
  `-3.4927 -> -3.5141`.
- The small spline-flow result is materially better than the previous
  `2.048M` spline median: target ratio `16.78x -> 12.02x`, W
  `0.1779 -> 0.1283`, and NLL `-3.5103 -> -3.5431`.
- This means the apparent small-spline W floor around `16-17x` after `1M/2M`
  was not a reliable final floor. It was likely a combination of single-family
  seed noise, local curvature in the scaling curve, or optimization/sampling
  variance at those data sizes.
- The result is still only one seed at `4.096M`; do not refit the exponent as
  if this were a settled multi-seed point. The most defensible next diagnostic
  is another `4.096M` spline seed, or a paired second seed for both MDN and
  spline if we want a cleaner family comparison.

## Fixed-P 8M Diagnostic

Ran the next power-of-two fixed-P diagnostic on the Mac mini:

```text
seed = 20260901
D = 8,192,000
validation cache = broad_prior_val_1m_float32.npz
panel W cache = decay_panel16_grid180_marginals.npz
posterior samples per panel signal = 20,000
jobs = 1 per family, launched concurrently on the Mac mini
device = cpu
torch_threads = 2
```

Outputs:

- `runs/01_exponential_decay/15_broad_scaling/35_mini_fixed_p_8m_diagnostic/mdn/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/35_mini_fixed_p_8m_diagnostic/spline/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/31_mdn_vs_spline_fixed_p_d_scaling/figures/mdn_vs_spline_fixed_p_2x2_8m.png`
- `runs/01_exponential_decay/15_broad_scaling/31_mdn_vs_spline_fixed_p_d_scaling/results/mdn_vs_spline_fixed_p_summary_8m.json`

| Family | Parameters | D | Seeds | Epochs | Panel mean W | Panel target ratio | Final 1M NLL | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MDN base | 44,722 | 8,192,000 | 1 | 72 | 0.1892 | 17.58 | -3.5121 | 2363.2 |
| Spline flow small | 45,844 | 8,192,000 | 1 | 32 | 0.1610 | 15.03 | -3.5269 | 2513.4 |

Context against the preceding `4.096M` single-seed diagnostic:

| Family | D | Panel mean W | Panel target ratio | Final 1M NLL |
| --- | ---: | ---: | ---: | ---: |
| MDN base | 4,096,000 | 0.1817 | 16.89 | -3.5141 |
| MDN base | 8,192,000 | 0.1892 | 17.58 | -3.5121 |
| Spline flow small | 4,096,000 | 0.1283 | 12.02 | -3.5431 |
| Spline flow small | 8,192,000 | 0.1610 | 15.03 | -3.5269 |

Interpretation:

- The single `8.192M` seed does not improve either family over the single
  `4.096M` seed on panel-W or final 1M NLL.
- MDN is roughly flat-to-worse from `4.096M` to `8.192M`; this is consistent
  with the broader evidence that the fixed-P MDN is no longer data-limited in
  the useful W metric at this architecture.
- The spline `8.192M` result is worse than its unusually good `4.096M` seed.
  Since the spline `8.192M` run early-stopped after only 32 epochs, this should
  not be interpreted as a clean law violation by itself. It is evidence that
  the high-D fixed-P spline curve is noisy or optimization-sensitive enough
  that single-seed extrapolations are not trustworthy.
- The next defensible experiment is not a blind jump above `8M`. It should be
  either repeat seeds at `4.096M/8.192M`, or change the optimization schedule
  for high-D spline runs and rerun the same seed to separate data scaling from
  early-stopping/optimization effects.

## Fixed-P 16M Diagnostic And Final Readout

Ran the final fixed-P power-of-two diagnostic on the Mac mini:

```text
seed = 20260901
D = 16,384,000
validation cache = broad_prior_val_1m_float32.npz
panel W cache = decay_panel16_grid180_marginals.npz
posterior samples per panel signal = 20,000
jobs = 1 per family, launched concurrently on the Mac mini
device = cpu
torch_threads = 2
```

Outputs:

- `runs/01_exponential_decay/15_broad_scaling/36_mini_fixed_p_16m_diagnostic/mdn/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/36_mini_fixed_p_16m_diagnostic/spline/results/broad_scaling_summary.json`
- `runs/01_exponential_decay/15_broad_scaling/31_mdn_vs_spline_fixed_p_d_scaling/figures/mdn_vs_spline_fixed_p_2x2_16m.png`
- `runs/01_exponential_decay/15_broad_scaling/31_mdn_vs_spline_fixed_p_d_scaling/results/mdn_vs_spline_fixed_p_summary_16m.json`

| Family | Parameters | D | Seeds | Epochs | Panel mean W | Panel target ratio | Final 1M NLL | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MDN base | 44,722 | 16,384,000 | 1 | 63 | 0.1711 | 15.96 | -3.5272 | 4147.8 |
| Spline flow small | 45,844 | 16,384,000 | 1 | 50 | 0.1812 | 17.03 | -3.5566 | 7358.2 |

High-D single-seed context:

| Family | D | Panel mean W | Panel target ratio | Final 1M NLL |
| --- | ---: | ---: | ---: | ---: |
| MDN base | 4,096,000 | 0.1817 | 16.89 | -3.5141 |
| MDN base | 8,192,000 | 0.1892 | 17.58 | -3.5121 |
| MDN base | 16,384,000 | 0.1711 | 15.96 | -3.5272 |
| Spline flow small | 4,096,000 | 0.1283 | 12.02 | -3.5431 |
| Spline flow small | 8,192,000 | 0.1610 | 15.03 | -3.5269 |
| Spline flow small | 16,384,000 | 0.1812 | 17.03 | -3.5566 |

Final fixed-P interpretation:

- The low/mid-D region still supports the original scaling-law motivation:
  panel-W and NLL improve strongly as data increases through the first few
  hundred thousand simulations.
- At fixed small parameter count, the high-D W curve is not a clean power law.
  MDN hovers around `16-18x` target ratio from `2M` through `16M`, and spline
  has one very good `4M` W point that is not sustained at `8M/16M`.
- The NLL objective continues to improve more smoothly than panel-W at high D.
  The spline `16M` NLL is the best density score observed here, but its panel-W
  is worse than the MDN `16M` point. This confirms that final NLL and posterior
  marginal W are not interchangeable diagnostics: NLL measures average
  conditional density fit over prior-predictive validation signals, while panel
  W measures posterior shape/calibration on a small set of signals and can
  expose different errors.
- These fixed-P runs do not justify another blind D-only jump. The better next
  scaling-law experiment is a Chinchilla-style parameter/data sweep around the
  useful high-D range, with repeat seeds or an optimization control. For the
  present architectures, simply adding data past a few million simulations is
  mostly probing optimization/seed/objective mismatch rather than a stable W
  scaling exponent.
