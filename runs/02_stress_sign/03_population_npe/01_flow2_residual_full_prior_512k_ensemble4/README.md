# 03_population_npe / 01_flow2_residual_full_prior_512k_ensemble4

Status: `near_floor`

This run applies the strongest single-decay population-training recipe to the
sign-symmetry model with minimal new machinery. It trains an equal-weight
4-member folded-target NPE ensemble on full prior-predictive pairs and evaluates
the result against the folded full-prior population NLL floor.

## Model

```math
x =
\begin{bmatrix}
\theta_1^2 \\
\theta_2
\end{bmatrix}
+ \epsilon,
\qquad
\epsilon \sim \mathcal N(0, \mathrm{diag}(0.22^2, 0.16^2)),
\qquad
\theta \sim \mathcal N(0, \mathrm{diag}(1.8^2, 1.8^2)).
```

Density target:

```math
z = (|\theta_1|,\theta_2).
```

## Training Recipe

The implementation reuses the `npe_stage1_decay` model builder and train loop,
with only the simulator and folded sign target swapped in.

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
| Device | `mps` |
| Wall time | `663.83s` |

Command:

```sh
uv run scripts/train_sign_population_npe.py \
  --output-root runs/02_stress_sign/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4 \
  --seeds 20260901,20260902,20260903,20260904 \
  --train-simulations 512000 \
  --val-simulations 16384 \
  --validation-examples 1000000 \
  --epochs 15 \
  --batch-size 512 \
  --eval-batch-size 65536 \
  --skip-training-validation
```

## Full-Prior NLL Result

Evaluation uses 1M fresh full-prior validation pairs with seed `20260705`.

| Quantity | Value |
| --- | ---: |
| Equal-weight ensemble NLL | `-1.422612950153621 +/- 0.0011687392391493943` |
| Folded population entropy floor | `-1.426941782495585 +/- 0.0011526154301947824` |
| Gap to floor | `0.004328832341963906` |
| Combined standard error | `0.0016414852235249054` |
| Gap z-score | `2.6371436549810814` |

Individual member NLLs:

| Member | Seed | NLL |
| --- | ---: | ---: |
| 1 | `20260901` | `-1.4213997079646588` |
| 2 | `20260902` | `-1.422586984015584` |
| 3 | `20260903` | `-1.4195335237932205` |
| 4 | `20260904` | `-1.422411425444007` |

Conclusion: this is a decent full-prior sign result and is close to the folded
NLL floor, but it has not cleanly reached the floor. The measured gap is still
about `2.64` combined standard errors above the entropy-floor estimate.

![Sign population NPE training loss](../../../00_shared_assets/readme_sign_posteriors/sign_population_training_loss.png)

The source run skipped per-epoch validation. The curves are member training NLLs
in folded target units; the marker is the final 1M-example full-prior validation
NLL.

## Posterior-Shape Check

A fresh full-prior signal was drawn with seed `20260707`, draw index `1`, giving
\(\theta=(1.419,-1.175)\) and \(x=(1.956,-0.932)\). Against a dense exact grid,
mean normalized Wasserstein in folded diagnostic coordinates is `0.02112` for
MCMC and `0.02163` for this population NPE ensemble.

![Sign population exact grid, MCMC, and NPE posterior overlay](../../../00_shared_assets/readme_sign_posteriors/sign_population_prior_signal_corner.png)

## Artifacts

- Ensemble summary: `results/sign_population_ensemble_summary.json`
- Member checkpoints and summaries: `member_*_seed*/results/`
- Training script: `../../../../scripts/train_sign_population_npe.py`
