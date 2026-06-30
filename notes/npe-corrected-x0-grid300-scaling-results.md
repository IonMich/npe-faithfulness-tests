# Corrected x0 Grid-300 NPE Scaling Results

## Reference Cache

Cached the original single-decay observation `x0` reference posterior at
`300^3` grid resolution:

```bash
uv run scripts/cache_decay_x0_grid_reference.py --grid-size 300
```

Outputs:

- `runs/01_exponential_decay/13_reference_cache/01_x0_grid300/results/decay_x0_grid300_reference.npz`
- `runs/01_exponential_decay/13_reference_cache/01_x0_grid300/results/decay_x0_grid300_reference_metadata.json`

Reference size and timing:

| Quantity | Value |
| --- | ---: |
| Grid points | 27,000,000 |
| Raw core arrays | 824.0 MiB |
| Compressed `.npz` | 279.0 MiB |
| Build time | 11.5 s |
| Compressed save time | 14.6 s |
| Total time | 40.6 s |

Grid-300 calibration:

| Reference comparison | Mean normalized W |
| --- | ---: |
| MCMC to grid-300 | 0.01709 |
| HMC to grid-300 | 0.01372 |

The grid-300 target used in plots is `0.01709`, the worse of MCMC/HMC.

## Corrected Plots

Regenerated x0 scaling plots against the cached grid-300 reference:

```bash
uv run scripts/plot_corrected_npe_x0_scaling.py
```

Outputs:

- `runs/01_exponential_decay/14_corrected_scaling_x0/01_grid300/results/corrected_npe_x0_scaling_summary.json`
- `runs/01_exponential_decay/14_corrected_scaling_x0/01_grid300/results/corrected_stage1_broad_x0_rows.csv`
- `runs/01_exponential_decay/14_corrected_scaling_x0/01_grid300/results/corrected_flow_x0_rows.csv`
- `runs/01_exponential_decay/14_corrected_scaling_x0/01_grid300/results/corrected_flow_x0_summary.csv`
- `runs/01_exponential_decay/14_corrected_scaling_x0/01_grid300/figures/corrected_stage1_broad_x0_scaling.png`
- `runs/01_exponential_decay/14_corrected_scaling_x0/01_grid300/figures/corrected_flow_x0_scaling.png`

## Broad Stage-1 x0 Results

Corrected single-`x0` Wasserstein values:

| Family | Train simulations | Old grid W | Corrected grid-300 W | Best val NLL |
| --- | ---: | ---: | ---: | ---: |
| Diagonal Gaussian | 20,000 | 0.3483 | 0.3477 | -1.6947 |
| Diagonal Gaussian | 100,000 | 0.3600 | 0.3599 | -2.1338 |
| Full Gaussian | 20,000 | 0.5898 | 0.5893 | -1.9488 |
| Full Gaussian | 100,000 | 0.2481 | 0.2467 | -2.3886 |
| MDN | 20,000 | 0.5225 | 0.5216 | -1.8887 |
| MDN | 100,000 | 0.1564 | 0.1551 | -2.4527 |
| Affine flow | 20,000 | 0.3291 | 0.3254 | -1.8389 |
| Affine flow | 100,000 | 0.2548 | 0.2502 | -2.4981 |

The broad Stage-1 conclusion is unchanged by grid correction: the scaled MDN is
best at `x0`, but still far above the grid-300 target.

## Flow x0 Results

Corrected historical flow-search rows:

| Group | Train simulations | Runs | Corrected W median | Corrected W min | Corrected W max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local q=0.005 linear flow | 40,000 | 1 | 0.02753 | 0.02753 | 0.02753 |
| Local q=0.005 linear flow | 100,000 | 4 | 0.02406 | 0.01927 | 0.03569 |
| Local q=0.005 linear flow | 150,000 | 1 | 0.01677 | 0.01677 | 0.01677 |
| Local q=0.005 linear flow | 250,000 | 1 | 0.01557 | 0.01557 | 0.01557 |
| Weighted-proposal broad flow | 5,000 | 2 | 7.64571 | 3.26665 | 12.02477 |
| Weighted-proposal broad flow | 100,000 | 1 | 0.05393 | 0.05393 | 0.05393 |
| Weighted-proposal broad flow | 150,000 | 1 | 0.08065 | 0.08065 | 0.08065 |

The corrected local flow curve now crosses the grid-300 target near the
150k/250k historical runs. This is consistent with the earlier finding that
grid-90 was masking the true x0 accuracy once models were close to the target.

## Controlled Saved-Sample Audit

Reran the controlled local sweep with posterior sample saving and accepted-pool
caching for the first two seeds:

```bash
uv run scripts/decay_local_scaling_sweep.py \
  --preset full \
  --output-root runs/01_exponential_decay/12_local_scaling/04_full_large_controlled_saved_samples \
  --local-region-summary runs/01_exponential_decay/12_local_scaling/02_full_large_controlled/results/local_region.json \
  --train-simulations 40000,80000,150000,300000 \
  --save-samples \
  --cache-pools \
  --skip-existing
```

I stopped after seeds `20260701` and `20260702`. The script had started
collecting the next seed's pool, but no completed sample artifact was lost.

The accepted-pool cache stores the accepted local `z`, contexts, distances, and
nested train/validation indices. That is better than saving only RNG seeds:
seeds would make the run reproducible, but would still require replaying all
rejected candidates each time.

Storage after the two-seed audit:

| Artifact | Size |
| --- | ---: |
| Cached grid-300 `x0` reference | 290 MiB |
| Controlled saved-sample rerun root | 72 MiB |
| Two accepted-pool caches | 37 MiB |
| Eight posterior sample files | 34.5 MiB |

Rescored the saved samples against the cached grid-300 reference:

```bash
uv run scripts/rescore_local_scaling_with_cached_reference.py \
  --scaling-root runs/01_exponential_decay/12_local_scaling/04_full_large_controlled_saved_samples
```

Outputs:

- `runs/01_exponential_decay/12_local_scaling/04_full_large_controlled_saved_samples/results/local_data_scaling_grid300_summary.json`
- `runs/01_exponential_decay/12_local_scaling/04_full_large_controlled_saved_samples/results/local_data_scaling_grid300_summary.csv`
- `runs/01_exponential_decay/12_local_scaling/04_full_large_controlled_saved_samples/results/local_data_scaling_grid300_rows.csv`
- `runs/01_exponential_decay/12_local_scaling/04_full_large_controlled_saved_samples/figures/local_data_scaling_grid300.png`

Two-seed corrected controlled medians:

| Accepted train simulations | Corrected W median | Corrected target ratio median | Best val NLL target-z median |
| ---: | ---: | ---: | ---: |
| 40,000 | 0.03330 | 1.948 | -4.4541 |
| 80,000 | 0.04089 | 2.393 | -4.4582 |
| 150,000 | 0.02071 | 1.212 | -4.4609 |
| 300,000 | 0.02407 | 1.408 | -4.4647 |

The corrected W audit still shows seed sensitivity and a non-monotone
150k-to-300k tail. In contrast, validation NLL keeps improving monotonically.
This supports using NLL as the primary scaling-law metric and using high-grid
Wasserstein as a selected posterior-quality audit, not as the main dependent
variable for broad sweeps.

## Limitations

- These corrected plots are still single-observation `x0` plots.
- The original controlled 40k/80k/150k/300k local scaling sweep did not save
  model checkpoints or posterior samples, so only rerun points with saved
  posterior samples can be corrected to grid-300.
- The controlled saved-sample audit currently has two seeds, not the full five
  seeds from the original controlled sweep.
- The historical flow-search points are not perfectly controlled scaling-law
  evidence because architectures, seeds, and some proposal settings differ.
