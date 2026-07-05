# 03_population_npe / 02_flow2_residual_full_prior_512k_ensemble4_e30

Status: `near_floor`

This run applies the single-decay/sign Flow2 residual NSF population recipe to
the Label Switching model. It trains an equal-weight 4-member density ensemble
on full-prior simulations and evaluates NLL in sorted target coordinates
`z_sorted=(mu_low, mu_high, log_sigma)`.

## Model

The likelihood is the exchangeable two-component Gaussian mixture:

```text
x_i ~ 0.5 Normal(mu_1, sigma^2) + 0.5 Normal(mu_2, sigma^2)
i = 1, ..., 80
```

The raw prior is Gaussian in `(mu_1, mu_2, log_sigma)`. The NLL target sorts
the two means and keeps `log_sigma`; physical `sigma` is only a display
coordinate.

The entropy floor is computed in these same sorted coordinates. Per-signal raw
evidence is estimated by symmetric Gaussian-mixture importance sampling with
4096 samples per signal. The sorted posterior density includes the `log 2`
fold factor for the two raw label permutations.

## Training Recipe

This is the same Flow2 residual NSF recipe used for the sign, Linear6, and
Banana population runs, with 30 epochs because the 15-epoch Label Switching
run still had a slightly larger residual gap.

| Setting | Value |
| --- | ---: |
| Ensemble members | `4` |
| Training simulations per member | `512000` |
| Epochs | `30` |
| Batch size | `512` |
| Flow | `2` NSF transforms, `8` spline bins |
| Conditioner | width `80`, `2` hidden layers, ReLU residual blocks |
| Inter-transform permutations | random |
| Learning rate | `0.00325` |
| Schedule | `cosine_step`, `500` warmup steps |
| Weight decay | `0.0002` |
| Device | `mps` on the Mac mini |
| Wall time | `1272.55s` |

Command:

```sh
uv run scripts/train_sign_population_npe.py \
  --model label_switch \
  --output-root runs/04_stress_label_switch/03_population_npe/02_flow2_residual_full_prior_512k_ensemble4_e30 \
  --seeds 20261111,20261112,20261113,20261114 \
  --train-simulations 512000 \
  --val-simulations 16384 \
  --validation-examples 50000 \
  --epochs 30 \
  --batch-size 512 \
  --eval-batch-size 65536 \
  --device mps \
  --label-importance-samples 4096 \
  --label-importance-batch-size 32 \
  --label-prior-mixture 0.03 \
  --label-proposal-inflation 2.0
```

## Full-Prior NLL Result

| Quantity | Value |
| --- | ---: |
| Equal-weight ensemble NLL | `-3.0924999546557705 +/- 0.008216554964424975` |
| Label Switching population entropy floor | `-3.1011164846256136 +/- 0.008210014576857754` |
| Gap to floor | `0.008616529969842749` |
| Combined standard error | `0.011615339634966923` |
| Combined-SE z-score | `0.7418233336805297` |
| Paired gap standard error | `0.0006014399753545138` |

Individual member NLLs:

| Member | Seed | NLL |
| ---: | ---: | ---: |
| 1 | `20261111` | `-3.086963474009037` |
| 2 | `20261112` | `-3.0881328991317747` |
| 3 | `20261113` | `-3.0869163677096365` |
| 4 | `20261114` | `-3.0888446278238297` |

Conclusion: this reaches the same practical near-floor level as the
single-decay, sign, and Linear6 population NPEs. The combined uncertainty does
not resolve the gap, but the paired cache does resolve a small residual
`0.00862 +/- 0.00060`, so this is a `near_floor` result rather than an exact
floor hit.

![Label Switching population NPE training loss](../../../00_shared_assets/readme_label_switch_posteriors/label_switch_population_training_loss.png)

The source run skipped per-epoch validation. The curves are member training
NLLs in target `z_sorted` units against total training wall time; the marker is
the final 50k-example full-prior validation NLL.

## Posterior Shape Check

A fresh full-prior signal was drawn with seed `20260707`, draw index `10`,
giving
`z_sorted=(-1.2226, 1.4333, -0.7126)`. The comparison is in the NLL target
coordinates and includes an exact finite grid, MCMC, and the population NPE.
Mean normalized marginal Wasserstein distance to the exact grid is `0.02729`
for the NPE and `0.02979` for MCMC; the NPE-to-MCMC diagnostic is `0.02365`.
The exact grid uses `96^3` points and has edge mass `1.38e-05`.

![Label Switching exact grid, MCMC, and NPE posterior overlay](../../../00_shared_assets/readme_label_switch_posteriors/label_switch_population_prior_signal_corner.png)

## Artifacts

- Ensemble summary: `results/label_switch_population_ensemble_summary.json`
- Member checkpoints: `member_*_seed*/results/label_switch_population_spline_flow_model.pt`
- Training script: `../../../../scripts/train_sign_population_npe.py`
- Training-loss plot: `../../../00_shared_assets/readme_label_switch_posteriors/label_switch_population_training_loss.png`
- Posterior diagnostic summary: `../../../00_shared_assets/readme_label_switch_posteriors/label_switch_population_prior_signal_summary.json`
