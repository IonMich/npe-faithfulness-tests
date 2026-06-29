# Scaled Broad Stage 1 NPE Results

This run tests whether the broad amortized map

```math
x \mapsto q_\phi(\theta \mid x)
```

becomes substantially more faithful when trained with more simulations and larger networks, without focusing on a single observed data set.

## Scaled command

```bash
uv run scripts/npe_stage1_decay.py \
  --device cpu \
  --epochs 180 \
  --train-simulations 100000 \
  --val-simulations 20000 \
  --batch-size 1024 \
  --hidden-dim 192 \
  --hidden-layers 4 \
  --mdn-components 8 \
  --flow-layers 8 \
  --flow-context-dim 96 \
  --posterior-samples 80000 \
  --reference-grid-size 90 \
  --output-dir runs/01_exponential_decay/02_npe_stage1_local_summary/12_npe_stage1_scaled/results \
  --figure-dir runs/01_exponential_decay/02_npe_stage1_local_summary/12_npe_stage1_scaled/figures
```

The follow-up broad faithfulness check used:

```bash
uv run scripts/evaluate_npe_multi_x.py \
  --stage1-dir runs/01_exponential_decay/02_npe_stage1_local_summary/12_npe_stage1_scaled/results \
  --num-observations 8 \
  --posterior-samples 40000 \
  --grid-size 70 \
  --target-wasserstein 0.034 \
  --output-dir runs/01_exponential_decay/02_npe_stage1_local_summary/09_npe_multi_x_scaled/results \
  --figure-dir runs/01_exponential_decay/02_npe_stage1_local_summary/09_npe_multi_x_scaled/figures
```

## MC-level faithfulness target

We use `0.034` as the strict target mean normalized Wasserstein distance. This is anchored to the previously converged MC samplers against the grid reference:

- MCMC was about `0.0335`.
- HMC was about `0.0316`.
- The target is therefore `0.034`, approximately the worse of the two trusted MC distances.

A broad amortized NPE method passes this strict criterion at an observation only if

```math
W_\mathrm{norm}(q_\phi(\theta \mid x_i), p(\theta \mid x_i)) \le 0.034.
```

## Single observed data set

Mean normalized Wasserstein distance to the grid posterior at the original observation:

| Family | Original Stage 1 | Scaled Stage 1 |
| --- | ---: | ---: |
| Diagonal Gaussian | 0.3483 | 0.3600 |
| Full Gaussian | 0.5898 | 0.2481 |
| MDN | 0.5225 | 0.1564 |
| Affine flow | 0.3291 | 0.2548 |

The scaled MDN improved most at the original observation. Full Gaussian and affine flow also improved. Diagonal Gaussian did not, which is expected because its independent-normal posterior shape is structurally too restrictive.

## Multi-observation broad faithfulness

Aggregate mean normalized Wasserstein distance over 8 independently simulated observations:

| Family | Original median | Scaled median | Original mean | Scaled mean | Original max | Scaled max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Diagonal Gaussian | 0.4472 | 0.2558 | 0.5014 | 0.3168 | 1.1221 | 0.5908 |
| Full Gaussian | 0.3212 | 0.2705 | 0.4088 | 0.2650 | 0.7377 | 0.5037 |
| MDN | 0.4045 | 0.1913 | 0.4098 | 0.2260 | 0.7149 | 0.3872 |
| Affine flow | 0.4280 | 0.3881 | 0.4259 | 0.3679 | 0.6132 | 0.6701 |

Scaling improves broad amortized faithfulness for most families. MDN is the best of the scaled models in this run, with the lowest median, mean, and max distance across the 8 observations.

Against the strict `0.034` target, no broad model passed any of the 8 observations:

| Family | Passes / 8 | Scaled median / target | Scaled max / target |
| --- | ---: | ---: | ---: |
| Diagonal Gaussian | 0 | 7.5x | 17.4x |
| Full Gaussian | 0 | 8.0x | 14.8x |
| MDN | 0 | 5.6x | 11.4x |
| Affine flow | 0 | 11.4x | 19.7x |

## Training cost

| Family | Original seconds | Scaled seconds | Scaled epochs |
| --- | ---: | ---: | ---: |
| Diagonal Gaussian | 9.2 | 62.2 | 119 |
| Full Gaussian | 8.9 | 67.2 | 118 |
| MDN | 10.6 | 123.0 | 123 |
| Affine flow | 29.0 | 605.0 | 140 |

The affine flow was much more expensive but did not dominate the scaled faithfulness metrics. For this toy problem and this implementation, the scaled MDN is the better cost/fidelity tradeoff.

## Interpretation

Scaling helps, so the earlier failure was not purely conceptual. The broad amortized learner had insufficient capacity/simulation budget relative to the difficulty of learning a posterior-valued function across the full prior predictive distribution.

However, scaling alone did not make the broad NPE posterior as faithful as the direct grid/MCMC/HMC reference. The strict target is `0.034`. The best scaled broad NPE result here was the MDN: `0.156` at the original observation and `0.226` mean over multiple observations.

The current ranking is:

1. Exact likelihood-based grid/HMC/MCMC reference: faithful for this tractable example.
2. Exact-target corrected focused NPE: faithful, but uses the tractable likelihood and is not the pure likelihood-free case.
3. Scaled broad MDN NPE: substantially improved but still approximate.
4. Small broad NPE: not faithful enough.

Figures:

- `runs/01_exponential_decay/02_npe_stage1_local_summary/12_npe_stage1_scaled/figures/npe_stage1_corner_overlay.png`
- `runs/01_exponential_decay/02_npe_stage1_local_summary/12_npe_stage1_scaled/figures/npe_stage1_predictive_overlay.png`
- `runs/01_exponential_decay/02_npe_stage1_local_summary/12_npe_stage1_scaled/figures/npe_stage1_training_curves.png`
- `runs/01_exponential_decay/02_npe_stage1_local_summary/09_npe_multi_x_scaled/figures/npe_multi_x_wasserstein.png`
