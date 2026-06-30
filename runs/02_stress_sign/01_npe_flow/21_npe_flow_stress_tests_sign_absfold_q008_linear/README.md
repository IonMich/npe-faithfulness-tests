# 01_npe_flow / 21_npe_flow_stress_tests_sign_absfold_q008_linear

Status: `grid-faithful`

This is the first sign-symmetry NPE flow run that matches the calibrated
MCMC/HMC faithfulness target against the exact grid posterior.

## Model

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

## NPE Setup

- Flow parameterization: `case_transform`, training on \((|\theta_1|, \theta_2)\).
- Symmetry restoration: random sign draw for \(\theta_1\) after sampling the flow.
- Local region: `local_quantile=0.008`, `kernel_quantile=0.008`.
- Local linear adjustment: enabled.
- Training samples: `60000`; validation samples: `12000`; NPE posterior samples: `80000`.

## Calibrated Grid Result

| Metric | Value |
| --- | ---: |
| Calibrated diagnostic target | `0.023314` |
| NPE-to-grid diagnostic mean normalized W | `0.023256` |
| NPE-to-grid raw mean normalized W | `0.017167` |
| NPE positive mode mass error | `0.000525` |

Per-dimension diagnostic NPE-to-grid distances:

| Dimension | Normalized W |
| --- | ---: |
| \(|\theta_1|\) | `0.020312` |
| \(\theta_2\) | `0.026199` |

The run passes the calibrated NPE-to-grid target by a narrow margin. The
finite-sample pairwise MCMC-NPE and HMC-NPE agreement checks remain slightly
above the same number, so use the exact-grid calibration artifact below as the
truth-faithfulness result for this run.

## Runtime

| Component | Seconds |
| --- | ---: |
| MCMC | `0.33` |
| HMC | `8.24` |
| NPE total with simulation | `253.05` |
| NPE training | `252.19` |

## Artifacts

- Samples and run summary: `results/`
- MCMC/HMC/NPE corner overlay: `figures/sign_mcmc_hmc_npe_corner.png`
- Posterior predictive plot: `figures/sign_predictive.png`
- Trace plot: `figures/sign_trace.png`
- Grid calibration summary: `../../02_reference_calibration/06_sign_absfold_q008_linear/results/sign_target_calibration_summary.json`
- Grid calibration plot: `../../02_reference_calibration/06_sign_absfold_q008_linear/figures/sign_target_calibration.png`
