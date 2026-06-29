# Multi-observation NPE faithfulness check

Date: 2026-06-22

## Question

Can the broad amortized Stage 1 NPE model be faithful across multiple observations
`x_i`, rather than only at the original observed dataset `x_o`?

## Method

The script `scripts/evaluate_npe_multi_x.py`:

1. Generates several new observations from the prior predictive.
2. For each observation, samples each trained Stage 1 posterior estimator.
3. Builds an independent numerical grid posterior reference for that observation.
4. Computes mean normalized Wasserstein distance from each NPE posterior to the
   grid reference.

Command:

```bash
uv run scripts/evaluate_npe_multi_x.py \
  --num-observations 8 \
  --posterior-samples 40000 \
  --grid-size 70 \
  --output-dir runs/01_exponential_decay/02_npe_stage1_local_summary/08_npe_multi_x/results \
  --figure-dir runs/01_exponential_decay/02_npe_stage1_local_summary/08_npe_multi_x/figures
```

## Results

Aggregate mean normalized Wasserstein distance across 8 observations:

| Family | Median | Mean | Max |
| --- | ---: | ---: | ---: |
| Diagonal Gaussian | `0.4510` | `0.5042` | `1.1255` |
| Full Gaussian | `0.3222` | `0.4076` | `0.7378` |
| MDN | `0.4060` | `0.4085` | `0.7118` |
| Affine flow | `0.4317` | `0.4294` | `0.6178` |

For comparison, the direct MCMC/HMC samplers were around `0.03` against the grid
reference for the original observation.

## Interpretation

The Stage 1 broad-prior NPE model is not broadly faithful at this simulation
budget. The learned map

```text
x -> p(theta | x)
```

is qualitatively useful but locally inaccurate for many observations.

This supports the diagnosis that the dominant problem is global amortization under
a broad prior with insufficient simulation coverage, not just the choice of
posterior family. To make globally amortized NPE faithful, we need a much larger
training budget, stronger flows/encoders, and calibration checks across many
observations.

If we instead focus on one observation, SNPE can improve local fidelity, but that
trades away global amortization.
