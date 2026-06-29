# Local x-region NPE results

This experiment tests the focused setup:

1. Simulate from the original prior and simulator.
2. Keep only simulations whose generated `x` is close to the observed `x0`.
3. Train `q_phi(theta | x)` only on this local region.
4. Evaluate `q_phi(theta | x0)` against the grid posterior.
5. Use `0.034` as the strict MC-level normalized Wasserstein target.

Because the accept/reject event depends only on `x`, the conditional target inside the accepted region is still the true prior posterior:

```math
p(\theta \mid x, x \in R(x_0)) = p(\theta \mid x)
```

for `x` inside the region. So this local experiment does not need proposal correction. It is different from theta-proposal-focused SNPE, where sampling from `r(theta)` changes the target unless corrected.

## Region definition

The local region is defined in a low-dimensional summary space:

- 8 binned means of the simulated curve.
- A rough noise summary from first differences.
- A rough global scale summary.
- Early-minus-late mean difference.

The summary dimensions are standardized under a pilot prior-predictive sample. A simulation is accepted if its standardized summary distance to `x0` is below a radius chosen by prior-predictive quantile.

For example, `summary_quantile = 0.005` means the closest 0.5% of prior-predictive summaries around `x0`.

## Script

The implementation is:

```bash
scripts/npe_local_region_decay.py
```

The first real local MDN run was:

```bash
uv run scripts/npe_local_region_decay.py \
  --families mdn \
  --train-simulations 20000 \
  --val-simulations 5000 \
  --pilot-simulations 120000 \
  --summary-quantile 0.005 \
  --max-candidates 10000000 \
  --epochs 180 \
  --hidden-dim 192 \
  --hidden-layers 4 \
  --mdn-components 8 \
  --posterior-samples 60000 \
  --reference-grid-size 90 \
  --target-wasserstein 0.034 \
  --output-dir runs/01_exponential_decay/02_npe_stage1_local_summary/05_npe_local_region_q0005_mdn_20k/results \
  --figure-dir runs/01_exponential_decay/02_npe_stage1_local_summary/05_npe_local_region_q0005_mdn_20k/figures
```

## Results

| Run | Family | Region quantile | Accepted train | Candidate sims | Train seconds | Mean normalized W | Target ratio | Pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `q002_mdn_20k` | MDN | 0.020 | 20k | 1.3M | 12.1 | 0.2316 | 6.81x | no |
| `q0005_mdn_20k` | MDN | 0.005 | 20k | 4.7M | 10.3 | 0.0867 | 2.55x | no |
| `q0001_mdn_20k` | MDN | 0.001 | 20k | 24.7M | 9.6 | 0.2065 | 6.07x | no |
| `q0005_mdn_60k_h256_c16` | MDN | 0.005 | 60k | 13.8M | 39.4 | 0.1523 | 4.48x | no |
| `q0005_gaussians_20k` | Diagonal Gaussian | 0.005 | 20k | 4.7M | 7.9 | 0.0888 | 2.61x | no |
| `q0005_gaussians_20k` | Full Gaussian | 0.005 | 20k | 4.7M | 7.5 | 0.1605 | 4.72x | no |
| `q0005_flow_20k` | Affine flow | 0.005 | 20k | 4.7M | 49.8 | 0.1512 | 4.45x | no |

Best result so far:

```text
MDN, 0.5% local region, 20k accepted training simulations:
mean normalized Wasserstein = 0.0867
target ratio = 2.55x
```

## Interpretation

Local focusing helped substantially. The best local result, `0.0867`, is better than the scaled broad MDN at the original observation, `0.1564`.

But the strict MC-level target, `0.034`, was not reached. The experiment also showed that scaling is not monotonic:

- Making the region too wide, 2%, under-focused the training distribution.
- Making the region too narrow, 0.1%, worsened the posterior, probably because the accepted set became less useful for learning the full conditional map around `x0`.
- Increasing MDN size and accepted training count at the 0.5% region also worsened this run, suggesting optimization/model-selection instability rather than a simple data shortage.

The best radius among these tests is currently the 0.5% summary region. The next likely improvement is not just more simulations; it is a better focused objective, for example a smooth kernel-weighted local loss around `x0` or a sequential SNPE procedure with calibration checks.

Figures:

- `runs/01_exponential_decay/02_npe_stage1_local_summary/05_npe_local_region_q0005_mdn_20k/figures/npe_local_region_curves.png`
- `runs/01_exponential_decay/02_npe_stage1_local_summary/05_npe_local_region_q0005_mdn_20k/figures/npe_local_region_corner_overlay.png`
- `runs/01_exponential_decay/02_npe_stage1_local_summary/05_npe_local_region_q0005_mdn_20k/figures/npe_local_region_predictive_overlay.png`
- `runs/01_exponential_decay/02_npe_stage1_local_summary/05_npe_local_region_q0005_mdn_20k/figures/npe_local_region_training_curves.png`
