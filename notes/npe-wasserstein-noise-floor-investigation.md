# NPE Wasserstein Noise-Floor Investigation

## Question

Could the apparent single-decay local NPE Wasserstein floor around `0.035` be
caused by finite NPE posterior samples or by finite grid/reference resolution?

## Method

I used the saved 150k local linear flow checkpoint:

```text
runs/01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706
```

This run is useful because it saved both:

- `npe_flow_decay_model.pt`
- `npe_flow_decay_samples.npz`

The controlled scaling sweep did not save checkpoints or posterior samples, so
its individual 150k/300k points cannot be re-evaluated without retraining.

The probe script is:

```bash
uv run scripts/npe_metric_noise_floor_probe.py --help
```

It measures:

- MCMC/HMC-to-grid Wasserstein as grid size changes;
- fixed NPE sample-to-grid Wasserstein as grid size changes;
- repeated NPE posterior sample-to-grid Wasserstein as posterior sample count
  changes;
- same-model NPE sample-to-sample Wasserstein as a posterior-sampling noise
  estimate.

## Main Probe

Command:

```bash
uv run scripts/npe_metric_noise_floor_probe.py \
  --output-root runs/01_exponential_decay/12_local_scaling/03_metric_noise_probe \
  --grid-sizes 60,90,120,150 \
  --metric-grid-size 90 \
  --sample-sizes 10000,25000,50000,100000,180000,300000,500000 \
  --repeats 3 \
  --device cpu
```

Outputs:

- `runs/01_exponential_decay/12_local_scaling/03_metric_noise_probe/results/metric_noise_probe_summary.json`
- `runs/01_exponential_decay/12_local_scaling/03_metric_noise_probe/results/grid_sensitivity.csv`
- `runs/01_exponential_decay/12_local_scaling/03_metric_noise_probe/results/fixed_sample_grid_sensitivity.csv`
- `runs/01_exponential_decay/12_local_scaling/03_metric_noise_probe/results/posterior_sample_sensitivity_summary.csv`
- `runs/01_exponential_decay/12_local_scaling/03_metric_noise_probe/results/posterior_sample_self_noise_summary.csv`
- `runs/01_exponential_decay/12_local_scaling/03_metric_noise_probe/figures/metric_noise_probe.png`

Runtime was about `39s`.

## Posterior Sample Count

At fixed grid size `90`, increasing NPE posterior samples barely changes the
estimated NPE-to-grid Wasserstein once `100k` samples are used:

| NPE posterior samples | Median W to grid 90 | Repeat sd |
| ---: | ---: | ---: |
| 10,000 | 0.03486 | 0.00044 |
| 25,000 | 0.03403 | 0.00033 |
| 50,000 | 0.03333 | 0.00027 |
| 100,000 | 0.03322 | 0.00036 |
| 180,000 | 0.03306 | 0.00035 |
| 300,000 | 0.03313 | 0.00018 |
| 500,000 | 0.03303 | 0.00016 |

The same-model sample-to-sample W continues to drop with sample count, but it is
already much smaller than the observed `0.033` floor at `100k+` samples:

| NPE posterior samples | Median same-model sample-to-sample W |
| ---: | ---: |
| 10,000 | 0.01448 |
| 25,000 | 0.00864 |
| 50,000 | 0.00749 |
| 100,000 | 0.00605 |
| 180,000 | 0.00484 |
| 300,000 | 0.00354 |
| 500,000 | 0.00296 |

Conclusion: `100k` posterior samples are enough for the current grid-90 metric.
Increasing NPE posterior samples cannot explain or remove the `~0.033` absolute
Wasserstein floor.

## Grid Resolution

Grid resolution has a large effect. With the same fixed 180k NPE sample, the
absolute W drops strongly as the grid is refined:

| Grid size per dim | Grid points | Fixed NPE sample W | MCMC-to-grid W | HMC-to-grid W |
| ---: | ---: | ---: | ---: | ---: |
| 60 | 216,000 | 0.04754 | 0.04760 | 0.04638 |
| 90 | 729,000 | 0.03310 | 0.03348 | 0.03162 |
| 120 | 1,728,000 | 0.02632 | 0.02684 | 0.02462 |
| 150 | 3,375,000 | 0.02257 | 0.02315 | 0.02058 |

This already shows that the apparent `~0.033` level is mostly a grid-90
reference artifact: MCMC, HMC, and NPE all move down together when the grid is
refined.

## Extended Grid Checks

Command:

```bash
uv run scripts/npe_metric_noise_floor_probe.py \
  --output-root runs/01_exponential_decay/12_local_scaling/03_metric_noise_probe_grid_extended \
  --grid-sizes 90,120,150,180,210 \
  --metric-grid-size 150 \
  --sample-sizes 100000 \
  --repeats 1 \
  --device cpu
```

Runtime was about `42s`.

| Grid size per dim | Grid points | Fixed NPE sample W | MCMC-to-grid W | HMC-to-grid W |
| ---: | ---: | ---: | ---: | ---: |
| 90 | 729,000 | 0.03310 | 0.03348 | 0.03162 |
| 120 | 1,728,000 | 0.02632 | 0.02684 | 0.02462 |
| 150 | 3,375,000 | 0.02257 | 0.02315 | 0.02058 |
| 180 | 5,832,000 | 0.02026 | 0.02089 | 0.01803 |
| 210 | 9,261,000 | 0.01879 | 0.01942 | 0.01633 |

Single-grid high-resolution checks:

| Grid size per dim | Grid points | Build seconds | Fixed NPE sample W | MCMC-to-grid W | HMC-to-grid W |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 240 | 13,824,000 | 5.00 | 0.01783 | 0.01838 | 0.01520 |
| 300 | 27,000,000 | 9.95 | 0.01677 | 0.01709 | 0.01372 |

The `300^3` check completed in about `52s` total. It is memory-heavier, but
still feasible for a one-off 3D decay reference on this machine.

## NLL Versus Wasserstein

Validation NLL is much cheaper operationally because it is computed during
training and does not require:

- posterior sampling;
- grid construction;
- weighted grid/sample Wasserstein comparisons.

However, NLL and posterior Wasserstein are not interchangeable. In the full
controlled scaling sweep, validation NLL improved monotonically from 40k to
300k simulations, while grid-90 Wasserstein plateaued. After this probe, that
plateau should not be interpreted as proof that posterior quality stopped
improving; the absolute grid-90 W floor is too coarse.

For this 3D decay case, high-resolution Wasserstein is not too expensive for
audit runs. For broader sweeps or higher-dimensional problems, grid-based
Wasserstein becomes the wrong primary scaling metric because grid cost grows
cubically here and exponentially in dimension generally.

## Saved-Sample Controlled Audit

To make the corrected W check concrete for the controlled local scaling setup,
I reran two seeds of the full controlled sweep with `--save-samples` and
`--cache-pools`, then rescored those saved posterior samples against the cached
`300^3` `x0` reference.

Outputs:

- `runs/01_exponential_decay/12_local_scaling/04_full_large_controlled_saved_samples/results/local_data_scaling_grid300_summary.json`
- `runs/01_exponential_decay/12_local_scaling/04_full_large_controlled_saved_samples/figures/local_data_scaling_grid300.png`

Storage was modest: `72 MiB` for the two-seed saved-sample rerun, including
`37 MiB` for two accepted-pool caches and `34.5 MiB` for eight posterior sample
files. The cached grid-300 reference itself is `290 MiB`.

The corrected two-seed median W values were:

| Accepted train simulations | Corrected W median | Best validation NLL target-z median |
| ---: | ---: | ---: |
| 40,000 | 0.03330 | -4.4541 |
| 80,000 | 0.04089 | -4.4582 |
| 150,000 | 0.02071 | -4.4609 |
| 300,000 | 0.02407 | -4.4647 |

The high-grid W audit confirms that coarse grid-90 W overstated the apparent
absolute floor, but it does not produce a clean monotone W scaling curve at the
largest sizes. Seed effects remain visible. Validation NLL is smoother,
available during training, and much cheaper to collect.

## Recommendation

Use this metric protocol going forward:

1. Use validation NLL as the cheap primary scaling curve during broad data/model
   sweeps.
2. Use Wasserstein only as an audit metric at selected scale points.
3. For single-decay 3D audit runs, use at least grid `240`, and preferably grid
   `300`, with `100k-180k` NPE posterior samples.
4. Do not interpret absolute grid-90 Wasserstein values around `0.03-0.035` as
   a model error floor.
5. Keep using target ratios when using a coarse grid, because MCMC/HMC-to-grid
   and NPE-to-grid shift together, but do not use the coarse absolute W as the
   scaling-law dependent variable near the target.

Current answer:

```text
The apparent Wasserstein floor is mostly grid/reference discretization, not
finite NPE posterior sampling. NLL is the right cheap primary metric for large
scaling sweeps, but high-grid Wasserstein remains feasible and valuable as a
selected audit metric for the 3D decay problem.
```
