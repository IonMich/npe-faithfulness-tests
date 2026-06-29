# Sequential SNPE results

This experiment tests proposal-focused sequential neural posterior estimation without using the analytic likelihood for inference.

The strict target remains:

```text
mean normalized Wasserstein <= 0.034
```

This target is anchored to the converged MCMC/HMC distances to the grid posterior.

## Custom proposal-corrected SNPE

Implementation:

```bash
scripts/snpe_sequential_decay.py
```

Round procedure:

1. Round 1 samples `z = log(theta)` from the prior.
2. Train `q_phi(z | x)` on simulations from the current proposal.
3. At `x0`, sample from the learned proposal-posterior.
4. Correct samples by importance weights `p(z) / r_t(z)`.
5. Fit an inflated Gaussian to the corrected samples and use it as the next proposal.

The correction uses only prior/proposal densities, not `p(x0 | theta)`.

### MDN, 4 rounds, inflation 2.5

Command:

```bash
uv run scripts/snpe_sequential_decay.py \
  --families mdn \
  --rounds 4 \
  --train-simulations 25000 \
  --val-simulations 5000 \
  --proposal-inflation 2.5 \
  --output-dir runs/01_exponential_decay/04_snpe_sbi/07_snpe_sequential_mdn_r4_n25k/results \
  --figure-dir runs/01_exponential_decay/04_snpe_sbi/07_snpe_sequential_mdn_r4_n25k/figures
```

| Round | Mean normalized W | Target ratio | Correction ESS fraction | Pass |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.3685 | 10.84x | 1.000 | no |
| 2 | 0.1547 | 4.55x | 0.947 | no |
| 3 | 0.1679 | 4.94x | 0.944 | no |
| 4 | 0.0878 | 2.58x | 0.966 | no |

Best in this run: `0.0878`.

### MDN, 6 rounds, inflation 1.5

| Round | Mean normalized W | Target ratio | Correction ESS fraction | Pass |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.3685 | 10.84x | 1.000 | no |
| 2 | 0.1547 | 4.55x | 0.473 | no |
| 3 | 0.1680 | 4.94x | 0.490 | no |
| 4 | 0.1542 | 4.54x | 0.769 | no |
| 5 | 0.1150 | 3.38x | 0.641 | no |
| 6 | 0.2755 | 8.10x | 0.567 | no |

Tighter proposals did not help. The run became less stable after round 5.

### Gaussian families, 4 rounds, inflation 2.5

| Family | Best round | Best mean normalized W | Target ratio | Pass |
| --- | ---: | ---: | ---: | --- |
| Diagonal Gaussian | 2 | 0.0848 | 2.49x | no |
| Full Gaussian | 3 | 0.1322 | 3.89x | no |

The diagonal Gaussian was surprisingly competitive here, but still missed the target.

## sbi SNPE-C

Implementation:

```bash
scripts/snpe_sbi_decay.py
```

Dependency added with:

```bash
uv add sbi
```

### sbi MDN

The sbi MDN run reached:

| Round | Mean normalized W | Target ratio | Pass |
| ---: | ---: | ---: | --- |
| 1 | 0.9023 | 26.54x | no |
| 2 | 1.3219 | 38.88x | no |

It then failed in round 3 with a non-positive-definite matrix inside the MDN proposal correction.

### sbi MAF

Command:

```bash
uv run scripts/snpe_sbi_decay.py \
  --rounds 4 \
  --simulations-per-round 25000 \
  --density-estimator maf \
  --output-dir runs/01_exponential_decay/04_snpe_sbi/03_snpe_sbi_maf_r4_n25k/results \
  --figure-dir runs/01_exponential_decay/04_snpe_sbi/03_snpe_sbi_maf_r4_n25k/figures
```

| Round | Mean normalized W | Target ratio | Pass |
| ---: | ---: | ---: | --- |
| 1 | 0.4244 | 12.48x | no |
| 2 | 0.4275 | 12.57x | no |
| 3 | 0.4241 | 12.47x | no |
| 4 | 0.4299 | 12.64x | no |

This standard-library configuration was not competitive with the custom proposal-corrected baseline.

## Interpretation

Sequential SNPE is legitimate, but these implementations still did not reach the strict MC-level target.

Best result across these sequential runs:

```text
custom diagonal Gaussian, round 2: 0.0848
custom MDN, round 4:             0.0878
target:                          0.0340
```

So sequential proposal focusing improved over broad amortization, but it did not produce MC-faithful posteriors in this setup.

The correction ESS fractions in the custom runs were mostly healthy, so the main failure is not obvious importance-weight collapse. The likely issue is residual density-estimation bias: the learned conditional posterior is visually close but still systematically shifted enough to fail a strict MC-level Wasserstein target.

Figures:

- `runs/01_exponential_decay/04_snpe_sbi/07_snpe_sequential_mdn_r4_n25k/figures/snpe_sequential_round_distances.png`
- `runs/01_exponential_decay/04_snpe_sbi/07_snpe_sequential_mdn_r4_n25k/figures/snpe_sequential_final_corner_overlay.png`
- `runs/01_exponential_decay/04_snpe_sbi/06_snpe_sequential_gaussians_r4_n25k/figures/snpe_sequential_round_distances.png`
- `runs/01_exponential_decay/04_snpe_sbi/03_snpe_sbi_maf_r4_n25k/figures/snpe_sbi_round_distances.png`
