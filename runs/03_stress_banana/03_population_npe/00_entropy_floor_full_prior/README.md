# 03_population_npe / 00_entropy_floor_full_prior

Status: `reference`

This reference run estimates the Banana full-prior entropy floor in raw
coordinates `theta=(theta1, theta2)`.

The estimator integrates `theta2` analytically, then computes the evidence with
posterior-centered one-dimensional Gauss-Hermite quadrature over `theta1`.

| Quantity | Value |
| --- | ---: |
| Validation examples | `1000000` |
| Quadrature order | `64` |
| Entropy floor | `-0.5282618872901833 +/- 0.0009994588666007852` |

Artifact: `results/banana_population_floor_summary.json`
