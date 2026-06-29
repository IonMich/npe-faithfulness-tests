# NPE Stage 1 results

Date: 2026-06-22

## Run

Command:

```bash
uv run scripts/npe_stage1_decay.py \
  --device cpu \
  --epochs 100 \
  --train-simulations 20000 \
  --val-simulations 5000 \
  --posterior-samples 60000 \
  --reference-grid-size 90
```

The script trained four conditional posterior families on simulator-generated
pairs `(theta, x)`:

- diagonal Gaussian
- full-covariance Gaussian
- mixture density network with 5 full-covariance Gaussian components
- conditional affine coupling flow with 6 coupling layers

All models use the same MLP observation encoder and learn a density over
standardized `z = log(theta)`.

## Outputs

- Summary: `runs/01_exponential_decay/02_npe_stage1_local_summary/11_npe_stage1/results/npe_stage1_summary.json`
- Samples: `runs/01_exponential_decay/02_npe_stage1_local_summary/11_npe_stage1/results/npe_stage1_samples.npz`
- Training curves: `runs/01_exponential_decay/02_npe_stage1_local_summary/11_npe_stage1/figures/npe_stage1_training_curves.png`
- Posterior overlay: `runs/01_exponential_decay/02_npe_stage1_local_summary/11_npe_stage1/figures/npe_stage1_corner_overlay.png`
- Posterior predictive overlay: `runs/01_exponential_decay/02_npe_stage1_local_summary/11_npe_stage1/figures/npe_stage1_predictive_overlay.png`

## Results

Mean normalized Wasserstein distance to the 90 x 90 x 90 grid posterior reference:

| Family | Best validation NLL | Train time | Mean normalized Wasserstein |
| --- | ---: | ---: | ---: |
| Diagonal Gaussian | `-1.6947` | `9.17 s` | `0.34831` |
| Full Gaussian | `-1.9488` | `8.94 s` | `0.58981` |
| MDN | `-1.8887` | `10.56 s` | `0.52253` |
| Affine flow | `-1.8389` | `29.01 s` | `0.32909` |

Grid reference posterior medians:

- `A`: `5.2920`
- `k`: `0.5699`
- `sigma`: `0.3410`

NPE posterior medians:

| Family | `A` | `k` | `sigma` |
| --- | ---: | ---: | ---: |
| Diagonal Gaussian | `5.2435` | `0.5552` | `0.3515` |
| Full Gaussian | `5.2497` | `0.5621` | `0.3931` |
| MDN | `5.2574` | `0.5619` | `0.3782` |
| Affine flow | `5.3493` | `0.5715` | `0.3372` |

## Interpretation

The models learned the broad posterior location and predictive behavior, but they
are not yet as faithful as the direct MCMC/HMC samplers for this single observed
dataset. This is expected for a first amortized NPE pass trained from a broad
prior with only 20,000 simulations.

The affine flow is best by the grid-reference metric in this run, but only
slightly better than the diagonal Gaussian. The full Gaussian and MDN learned
broader/noisier `sigma` posteriors for this `x_o`.

This gives a concrete next target: increase simulation budget, improve training
stability, and then test sequential/focused NPE where simulations are generated
near the observed posterior rather than across the full broad prior.
