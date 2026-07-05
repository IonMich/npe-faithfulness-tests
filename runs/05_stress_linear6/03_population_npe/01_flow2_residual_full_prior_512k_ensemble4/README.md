# 03_population_npe / 01_flow2_residual_full_prior_512k_ensemble4

Status: `near_floor`

This run applies the single-decay/sign Flow2 residual NSF population recipe to
the Linear6 model with minimal new machinery. It trains an equal-weight
4-member density ensemble on full-prior simulations and evaluates exact
full-prior NLL in the target coordinates
`z=(w1, ..., w6, log_sigma)`.

## Model

The simulator is the Linear6 stress model from the root README:

```text
y = sum_j w_j phi_j(t) + epsilon
epsilon ~ Normal(0, sigma^2)
z = (w1, ..., w6, log_sigma)
```

The entropy floor is computed in the same `z` coordinates using the
linear-Gaussian conditional posterior. Conditional on `log_sigma`, the weights
have an exact Gaussian posterior; the evidence integrates over `log_sigma` with
one-dimensional Gauss-Hermite quadrature.

## Training Recipe

This is the same Flow2 residual NSF recipe used for the sign population run.

| Setting | Value |
| --- | ---: |
| Ensemble members | `4` |
| Training simulations per member | `512000` |
| Epochs | `15` |
| Batch size | `512` |
| Flow | `2` NSF transforms, `8` spline bins |
| Conditioner | width `80`, `2` hidden layers, ReLU residual blocks |
| Inter-transform permutations | random |
| Learning rate | `0.00325` |
| Schedule | `cosine_step`, `500` warmup steps |
| Weight decay | `0.0002` |
| Device | `mps` on the Mac mini |
| Wall time | `509.93s` |

Command:

```sh
uv run scripts/train_sign_population_npe.py \
  --model linear6 \
  --output-root runs/05_stress_linear6/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4 \
  --seeds 20260901,20260902,20260903,20260904 \
  --train-simulations 512000 \
  --val-simulations 16384 \
  --validation-examples 1000000 \
  --epochs 15 \
  --batch-size 512 \
  --eval-batch-size 65536 \
  --device mps \
  --linear6-quadrature-order 64
```

## Full-Prior NLL Result

| Quantity | Value |
| --- | ---: |
| Equal-weight ensemble NLL | `-10.7798363088208 +/- 0.00352546522008669` |
| Linear6 population entropy floor | `-10.786310881823061 +/- 0.0035296909220397434` |
| Gap to floor | `0.006474573002262658` |
| Paired gap standard error | `0.00012034586019226568` |

Individual member NLLs:

| Member | Seed | NLL |
| ---: | ---: | ---: |
| 1 | `20260901` | `-10.767926209837078` |
| 2 | `20260902` | `-10.771354126898288` |
| 3 | `20260903` | `-10.775631882891656` |
| 4 | `20260904` | `-10.771390669057906` |

Conclusion: this reaches the same practical near-floor level as the
single-decay and sign population NPEs, but it is not an exact floor hit. The
paired validation comparison resolves the remaining `0.00647` z-unit gap, so a
larger data scale or ensemble would be needed to test whether the gap can close.

![Linear6 population NPE training loss](../../../00_shared_assets/readme_linear6_posteriors/linear6_population_training_loss.png)

The source run skipped per-epoch validation. The curves are member training
NLLs in target `z` units against total training wall time; the marker is the
final 1M-example full-prior validation NLL.

## Posterior Shape Check

A fresh full-prior signal was drawn with seed `20260707`, draw index `1`,
giving
`z=(1.897, -0.2525, 0.00248, -0.4968, -0.1156, 0.4749, -1.228)`.
Against exact posterior samples, the population NPE has mean normalized
marginal Wasserstein distance `0.01407` in the NLL target coordinates.

![Linear6 exact reference and NPE posterior overlay](../../../00_shared_assets/readme_linear6_posteriors/linear6_population_prior_signal_corner.png)

## Artifacts

- Ensemble summary: `results/linear6_population_ensemble_summary.json`
- Member checkpoints: `member_*_seed*/results/linear6_population_spline_flow_model.pt`
- Training script: `../../../../scripts/train_sign_population_npe.py`
- Training-loss plot: `../../../00_shared_assets/readme_linear6_posteriors/linear6_population_training_loss.png`
- Posterior diagnostic summary: `../../../00_shared_assets/readme_linear6_posteriors/linear6_population_prior_signal_summary.json`
