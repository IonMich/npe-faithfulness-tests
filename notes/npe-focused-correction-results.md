# Focused and corrected NPE results

Date: 2026-06-22

## Motivation

The first broad-prior amortized NPE run learned the rough posterior shape but was
not faithful enough for the observed decay dataset. Most prior simulations are far
from the posterior region for this particular observation `x_o`.

This run tests two amendments:

1. **Focused proposal simulations**: simulate parameters from a broad Gaussian
   proposal in `z = log(theta)` centered on a pilot MCMC/HMC posterior.
2. **Importance correction**:
   - proposal correction with `p_prior(z) / r(z)`
   - exact target correction with
     `p(x_o | z) p_prior(z) / q_phi(z | x_o)`

The second correction is only available here because the decay likelihood is
tractable. It is not available for a truly likelihood-free simulator.

## Command

```bash
uv run scripts/npe_focused_decay.py \
  --device cpu \
  --epochs 100 \
  --train-simulations 20000 \
  --val-simulations 5000 \
  --posterior-samples 120000 \
  --resampled-samples 60000 \
  --reference-grid-size 90 \
  --proposal-inflation 2.5
```

## Outputs

- Summary: `runs/01_exponential_decay/02_npe_stage1_local_summary/01_npe_focused/results/npe_focused_summary.json`
- Samples: `runs/01_exponential_decay/02_npe_stage1_local_summary/01_npe_focused/results/npe_focused_samples.npz`
- Proposal-corrected corner: `runs/01_exponential_decay/02_npe_stage1_local_summary/01_npe_focused/figures/npe_focused_proposal_corrected_corner_overlay.png`
- Exact-target-corrected corner: `runs/01_exponential_decay/02_npe_stage1_local_summary/01_npe_focused/figures/npe_focused_exact_corrected_corner_overlay.png`
- Exact-target-corrected predictive: `runs/01_exponential_decay/02_npe_stage1_local_summary/01_npe_focused/figures/npe_focused_exact_corrected_predictive_overlay.png`

## Proposal correction only

Mean normalized Wasserstein distance to the grid posterior:

| Family | Proposal-corrected distance | Importance ESS fraction |
| --- | ---: | ---: |
| Diagonal Gaussian | `0.17754` | `0.785` |
| Full Gaussian | `0.23307` | `0.931` |
| MDN | `0.24735` | `0.905` |
| Affine flow | `0.16449` | `0.952` |

Focused simulations plus prior/proposal correction improve over broad-prior NPE,
but are still not as faithful as MCMC/HMC for this observed dataset.

## Exact target correction

Mean normalized Wasserstein distance to the grid posterior:

| Family | Exact-target-corrected distance | Importance ESS fraction |
| --- | ---: | ---: |
| Diagonal Gaussian | `0.04125` | `0.294` |
| Full Gaussian | `0.03092` | `0.822` |
| MDN | `0.03061` | `0.766` |
| Affine flow | `0.03083` | `0.874` |

Grid reference medians:

- `A`: `5.2920`
- `k`: `0.5699`
- `sigma`: `0.3410`

Exact-target-corrected medians:

| Family | `A` | `k` | `sigma` |
| --- | ---: | ---: | ---: |
| Diagonal Gaussian | `5.2938` | `0.5714` | `0.3407` |
| Full Gaussian | `5.2901` | `0.5710` | `0.3414` |
| MDN | `5.2914` | `0.5709` | `0.3413` |
| Affine flow | `5.2900` | `0.5709` | `0.3413` |

## Interpretation

Yes, the neural methods can be amended to produce faithful posteriors here:

- As pure proposal-corrected focused NPE, they improve substantially but are not
  yet fully faithful.
- As neural proposal distributions followed by exact posterior importance
  correction, full Gaussian, MDN, and affine flow reach grid-reference fidelity
  comparable to MCMC/HMC.

The exact correction uses the tractable likelihood, so it is not a likelihood-free
SBI method. For a simulator-only setting, the analogous path is sequential NPE
with proposal correction, more simulation rounds near `x_o`, and calibration
checks such as SBC.
