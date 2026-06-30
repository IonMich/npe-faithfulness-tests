# Sign Model Target Calibration

Date: 2026-06-25

## Question

The historical target `0.034` came from the exponential-decay model. This checks whether it is appropriate for the next simplest model, the sign-symmetry stress case:

```math
x = (\theta_1^2, \theta_2) + \epsilon,
\qquad
\epsilon \sim \mathcal N(0, \mathrm{diag}(0.22^2, 0.16^2)),
\qquad
\theta \sim \mathcal N(0, \mathrm{diag}(1.8^2, 1.8^2)).
```

Observed signal:

```text
x0 = (0.3740799958, -0.3548783615)
```

## Method

Script:

```bash
uv run scripts/calibrate_sign_target.py
```

The script builds a dense `1001 x 1001` grid posterior over \((\theta_1,\theta_2)\), compares the existing sign MCMC/HMC/NPE samples to that grid, and reports both:

- raw coordinates: \((\theta_1,\theta_2)\)
- diagnostic coordinates: \((|\theta_1|,\theta_2)\)

Diagnostic coordinates factor out the sign degeneracy. Raw coordinates and mode mass are still reported because the exact posterior is bimodal and should have approximately equal positive/negative \(\theta_1\) mass.

## Results

The grid does not clip the posterior: edge mass is effectively zero.

Grid sampling noise for 60k samples:

| Metric | Median mean normalized W |
| --- | ---: |
| raw | `0.00520` |
| diagnostic | `0.00605` |

Distances to the grid posterior:

| Method | Raw mean W | Diagnostic mean W | Positive mode mass error |
| --- | ---: | ---: | ---: |
| MCMC | `0.07480` | `0.02331` | `0.06180` |
| HMC | `0.01135` | `0.01267` | `0.00156` |
| NPE | `0.03044` | `0.03261` | `0.01175` |

Pairwise diagnostic distances:

| Pair | Mean normalized W |
| --- | ---: |
| MCMC-HMC | `0.02089` |
| MCMC-NPE | `0.02217` |
| HMC-NPE | `0.02877` |

## Calibrated Targets

Using the same rule as the decay model, set the model-specific target to:

```text
target = max(full MCMC-to-grid, full HMC-to-grid)
```

For this sign case:

| Target | Value |
| --- | ---: |
| diagnostic mean normalized Wasserstein | `0.02331` |
| raw mean normalized Wasserstein | `0.07480` |
| positive mode mass error | `0.06180` |

## Interpretation

The historical `0.034` target was too loose for the diagnostic sign posterior. Under the model-calibrated diagnostic target:

```text
NPE diagnostic W = 0.03261
calibrated diagnostic target = 0.02331
```

So the previous sign run should no longer be called target-faithful to the grid posterior. It was only faithful under the inherited decay-model threshold and under pairwise agreement with MCMC/HMC.

The NPE does pass raw-coordinate and mode-mass checks, but the diagnostic shape is still biased relative to the grid posterior, mostly in \(|\theta_1|\).

## Successful NPE Follow-Up

Run:

```text
runs/02_stress_sign/01_npe_flow/21_npe_flow_stress_tests_sign_absfold_q008_linear
```

This run changed the sign-case NPE parameterization to learn
\((|\theta_1|,\theta_2)\), restored sign symmetry by random sign sampling, used
a tighter local window (`local_quantile=0.008`, `kernel_quantile=0.008`), and
enabled local linear adjustment.

Grid-calibrated results:

| Metric | Value |
| --- | ---: |
| calibrated diagnostic target | `0.023314` |
| NPE-to-grid diagnostic mean W | `0.023256` |
| NPE-to-grid raw mean W | `0.017167` |
| NPE positive mode mass error | `0.000525` |

This meets the calibrated MCMC/HMC faithfulness target by the exact-grid
criterion. The built-in pairwise agreement flag for the run remains false
because pairwise finite-sample MCMC-NPE and HMC-NPE distances are slightly above
the same number; the calibrated truth criterion is NPE-to-grid.

## Artifacts

- Summary: `runs/02_stress_sign/02_reference_calibration/01_sign_grid_reference/results/sign_target_calibration_summary.json`
- Plot: `runs/02_stress_sign/02_reference_calibration/01_sign_grid_reference/figures/sign_target_calibration.png`
- Successful NPE summary: `runs/02_stress_sign/02_reference_calibration/06_sign_absfold_q008_linear/results/sign_target_calibration_summary.json`
- Successful NPE plot: `runs/02_stress_sign/02_reference_calibration/06_sign_absfold_q008_linear/figures/sign_target_calibration.png`
