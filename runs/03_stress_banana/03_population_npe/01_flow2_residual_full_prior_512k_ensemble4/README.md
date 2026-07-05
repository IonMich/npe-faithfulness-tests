# 03_population_npe / 01_flow2_residual_full_prior_512k_ensemble4

Status: `floor_pass`

This run applies the single-decay/sign Flow2 residual NSF population recipe to
the Banana model. It trains an equal-weight 4-member density ensemble on
full-prior simulations and evaluates full-prior NLL in raw target coordinates
`theta=(theta1, theta2)`.

## Model

The simulator is the Banana stress model from the root README:

```text
x1 = theta1 + epsilon1
x2 = theta2 + 0.65 * (theta1^2 - 0.70) + epsilon2
epsilon ~ Normal(0, diag(0.20^2, 0.18^2))
theta ~ Normal(0, diag(1.8^2, 1.8^2))
```

The entropy floor is computed in the same raw coordinates by integrating
`theta2` analytically and using one-dimensional Gauss-Hermite evidence
integration over `theta1`.

## Training Recipe

This is the same Flow2 residual NSF recipe used for the sign and Linear6
population runs. The context is `(x1, x2, x2 - b*(x1^2-c), x1^2-c)`; the target
is raw `theta`.

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
| Wall time | `503.56s` |

Command:

```sh
uv run scripts/train_sign_population_npe.py \
  --model banana \
  --output-root runs/03_stress_banana/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4 \
  --seeds 20261001,20261002,20261003,20261004 \
  --train-simulations 512000 \
  --val-simulations 16384 \
  --validation-examples 1000000 \
  --epochs 15 \
  --batch-size 512 \
  --eval-batch-size 65536 \
  --device mps \
  --banana-quadrature-order 64
```

## Full-Prior NLL Result

| Quantity | Value |
| --- | ---: |
| Equal-weight ensemble NLL | `-0.5275281520603539 +/- 0.0010009409814320856` |
| Banana population entropy floor | `-0.5282618872901833 +/- 0.0009994588666007852` |
| Gap to floor | `0.0007337352298294745` |
| Combined standard error | `0.0014144966858699785` |
| Combined-SE z-score | `0.5187253085560922` |
| Paired gap standard error | `0.00003502006784055921` |

Individual member NLLs:

| Member | Seed | NLL |
| ---: | ---: | ---: |
| 1 | `20261001` | `-0.5262527626214027` |
| 2 | `20261002` | `-0.5266075029367209` |
| 3 | `20261003` | `-0.5268711535412073` |
| 4 | `20261004` | `-0.5264787333939075` |

Conclusion: this passes the full-prior floor criterion under the common
combined-standard-error accounting. The paired validation cache still resolves
the tiny residual gap, so this should be described as a floor pass with a small
remaining bias rather than an exact zero-gap model.

![Banana population NPE training loss](../../../00_shared_assets/readme_banana_posteriors/banana_population_training_loss.png)

## Posterior Shape Check

A fresh full-prior signal was drawn with seed `20260707`, draw index `1`,
giving `theta=(1.419, -1.175)` and
`x=(1.367, -0.0473)`. Against exact posterior samples, the population NPE has
mean normalized marginal Wasserstein distance `0.01022`; the MCMC reference is
`0.01072`.

![Banana exact grid, MCMC, and NPE posterior overlay](../../../00_shared_assets/readme_banana_posteriors/banana_population_prior_signal_corner.png)

## Artifacts

- Ensemble summary: `results/banana_population_ensemble_summary.json`
- Member checkpoints: `member_*_seed*/results/banana_population_spline_flow_model.pt`
- Training script: `../../../../scripts/train_sign_population_npe.py`
- Training-loss plot: `../../../00_shared_assets/readme_banana_posteriors/banana_population_training_loss.png`
- Posterior diagnostic summary: `../../../00_shared_assets/readme_banana_posteriors/banana_population_prior_signal_summary.json`
