# Global Faithfulness Upgrade Plan

Date: 2026-07-05

## Goal

Apply the workflow used for the sign-symmetry population NPE to every model
that does not yet have a clean global full-prior faithfulness result.

Global faithfulness here means:

```text
full-prior validation NLL in the chosen target coordinates
is statistically indistinguishable from an independently estimated
Bayes conditional entropy floor for the same coordinates.
```

This is deliberately not the older local `x_0` target, and it is not the
legacy MCMC/HMC/NPE pairwise agreement target. Local posterior overlays,
exact-grid comparisons, and MCMC checks remain useful diagnostics, but they do
not establish a global population result by themselves.

No finite run is expected to be exactly on the entropy floor. The practical
criterion is that the measured NLL gap should not be statistically resolved
above the floor, using the combined uncertainty of the NPE validation estimate
and the entropy-floor estimate.

## Current Status

| Model | Current best status | Global action needed |
| --- | --- | --- |
| Single-exponential decay | `near_floor` full-prior z-NLL result | Not a strict floor hit; keep as the reference recipe and plotting/evaluation template while tracking the remaining resolved gap. |
| Sign symmetry | `near_floor` full-prior folded NLL result | Not globally faithful yet; close the remaining NLL-floor gap or reduce uncertainty until the floor hit is statistically clean. |
| Banana | `floor_pass` full-prior raw-coordinate NLL result | Second completed transfer after Linear6; gap is within the entropy-floor uncertainty under the common criterion. |
| Label-switching mixture | `near_floor` full-prior sorted z-NLL result | Third completed transfer after Linear6/Banana; combined uncertainty does not resolve the gap, but the paired cache still resolves a small residual. |
| Linear6 | `near_floor` full-prior z-NLL result | First completed transfer after sign; remaining gap is real but at the same practical near-floor level as single decay and sign. |
| Ordered two-exponential decay | `blocked` after floor probe | The full-prior floor probe is now repeatable, but the transferred Flow2/nearby recipes are still far above it. Do not claim global faithfulness without a better target/context/family. |

## Reusable Workflow

The sign update should become the template, with model hooks rather than a new
one-off script for every model.

1. Define the target density coordinates.

   - Use the coordinates in which NLL will be reported.
   - Handle symmetries by changing the target, not by hoping the flow learns an
     arbitrary label/sign convention.
   - Record all Jacobians. If the model trains in log coordinates but reports
     physical coordinates, the NLL floor and NPE NLL must include the same
     coordinate transform.

2. Estimate the full-prior entropy floor.

   - Draw validation pairs from the full prior predictive distribution.
   - For each pair, evaluate or estimate `-log p(g(theta) | x)`.
   - Store the estimate, standard error, coordinate target, seeds, numerical
     method, and validation size under `runs/00_shared_assets/readme_entropy/`.
   - Update the common entropy-floor table only after the numerical method is
     independently checked.

3. Train the population NPE with the single-decay/sign recipe.

   Default proof recipe:

   ```text
   ensemble members              4
   train simulations per member  512k first, then 2.048M if needed
   epochs                        15
   batch size                    512
   flow                          2 NSF transforms, 8 spline bins
   conditioner                   width 80, 2 hidden layers, ReLU residual blocks
   inter-transform permutations  random
   learning rate                 0.00325
   schedule                      cosine_step, 500 warmup steps
   weight decay                  0.0002
   ```

   Reuse `npe_stage1_decay.py` training primitives. Prefer extracting a shared
   population-training harness from `train_sign_population_npe.py` before adding
   more model-specific scripts.

4. Evaluate exact full-prior NLL.

   - Use a fresh validation cache, ideally `1_000_000` examples for the final
     claim.
   - Report individual member NLLs and equal-weight ensemble NLL using
     `logsumexp`.
   - Compute the gap to the entropy floor and the combined standard-error
     z-score.
   - Do not tune on the final 1M cache. If fitting convex ensemble weights, use
     a separate fitting split and report a held-out/full-cache result.

5. Render diagnostics with reusable plotting paths.

   - Extend existing plotting scripts by adding modes. Do not create parallel
     plotting scripts when `plot_broad_efficiency_training_curves.py`,
     `render_decay_readme_posteriors.py`, or `npe_posterior_viewer.py` can be
     reused.
   - Every model should get a training-loss plot with wall time on the x-axis,
     matching the single-decay and sign plots.
   - For low-dimensional models, include a prior-predictive posterior overlay
     with exact grid/reference, NPE, and MCMC.
   - For higher-dimensional models, include marginal or pair-plot diagnostics
     plus posterior predictive checks. These are diagnostics, not the global
     pass criterion.

6. Update documentation and run status.

   Required artifacts per model:

   ```text
   runs/<model>/03_population_npe/<run>/README.md
   runs/<model>/03_population_npe/<run>/results/*summary.json
   runs/00_shared_assets/readme_<model>_posteriors/*
   root README section update
   runs/<model>/README.md status row
   runs/README.md best-status row
   ```

7. Commit and push the finished model slice before starting the next model.

   Each model-level checkpoint should include the training/evaluation code
   needed to reproduce the result, the run summaries, README assets, status
   indexes, and documentation changes. Do not carry completed model artifacts
   as uncommitted local state while moving to another model.

## Model-Specific Plans

### 0. Near-Floor Baselines

Single decay and sign are the two useful near-floor baselines, but neither
should be described as exactly on the floor.

Single decay:

```text
best deployed NLL  -3.63128 +/- 0.00252
entropy floor      -3.63865 +/- 0.00253
gap                0.00737, about 2.1 combined SE
```

The fitted data-scaling asymptote is closer to the entropy estimate, but that
is a fit diagnostic, not a deployed-model NLL. If strict global faithfulness is
required for single decay too, the next action is a larger data/ensemble proof
or a larger validation/floor estimate to determine whether the apparent gap is
real.

Sign:

```text
ensemble NLL  -1.42261 +/- 0.00117
folded floor  -1.42694 +/- 0.00115
gap           0.00433, about 2.64 combined SE
```

Sign is slightly less close by the same standard-error accounting, but it is
already useful as the first transfer of the single-decay recipe to another full
prior.

### 1. Linear6

Done as the first remaining-model transfer. It is a near-floor global
population result because most of the floor can be computed with model
structure instead of brute force.

Target coordinates:

```text
z = (w_1, ..., w_6, log_sigma)
```

Completed result:

```text
ensemble NLL  -10.77984 +/- 0.00353
entropy floor -10.78631 +/- 0.00353
gap            0.00647 in z units
paired gap SE  0.000120
```

Completed artifacts:

- `runs/05_stress_linear6/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4/`
- `runs/00_shared_assets/readme_linear6_posteriors/`
- root README Linear6 section and run-status indexes.

Floor method:

- Use the linear-Gaussian likelihood conditional on `sigma`.
- Integrate over `log sigma` with one-dimensional Gauss-Hermite quadrature.
- Conditional on each `sigma`, use the Gaussian posterior for the weights.
- Report NLL in `log_sigma` coordinates to avoid unnecessary Jacobian
  bookkeeping.

Follow-up if we want to close the resolved gap:

- Promote to `2.048M` simulations per member using the same recipe, or add
  another 4-member `512k` ensemble and test the 8-member equal-weight mixture.

Promotion gate:

```text
ensemble NLL - floor <= 2 combined SE
```

If NLL passes but posterior diagnostics show rare failures, add a panel
Wasserstein distribution over prior-predictive signals before marking the model
globally faithful.

### 2. Banana

Done as the second remaining-model transfer. It is two-dimensional, exact-grid
reference calculations are practical, and the full-prior NLL is within the
entropy-floor uncertainty under the common criterion.

Target coordinates:

```text
g(theta) = (theta_1, theta_2)
```

Completed result:

```text
ensemble NLL   -0.52753 +/- 0.00100
entropy floor  -0.52826 +/- 0.00100
gap             0.00073 in raw theta units
combined z      0.52
paired gap SE   0.000035
```

Completed artifacts:

- `runs/03_stress_banana/03_population_npe/00_entropy_floor_full_prior/`
- `runs/03_stress_banana/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4/`
- `runs/00_shared_assets/readme_banana_posteriors/`
- root README Banana section and run-status indexes.

Floor method:

- Integrate `theta2` analytically conditional on `theta1`.
- Use posterior-centered one-dimensional Gauss-Hermite evidence integration
  over `theta1`.
- Report NLL in raw `theta=(theta1, theta2)` coordinates.

Training recipe:

- Context: raw `x`, dewarped summary `x2 - b*(x1^2-c)`, and curvature
  `x1^2-c`.
- Target: raw coordinates, matching the floor.
- 4-member Flow2 recipe at `512k` per member.

Diagnostics:

- Exact grid vs NPE vs MCMC on one fresh prior-predictive signal.
- NPE-to-exact mean normalized marginal Wasserstein: `0.01025`.
- MCMC-to-exact mean normalized marginal Wasserstein: `0.01188`.

Follow-up if we want to close even the paired residual:

- The paired gap is small but statistically resolved because the same 1M cache
  is used for NPE and exact NLL. A larger ensemble or longer/data-scaled run
  can test whether the residual `0.00073` gap closes, but it is already below
  the common full-prior floor-pass threshold.

### 3. Label-Switching Mixture

Done as the third remaining-model transfer. It now has a full-prior
population NPE result in symmetry-aware sorted coordinates.

Target coordinates:

```text
g(z) = (mu_low, mu_high, log_sigma)
mu_low = min(mu_1, mu_2)
mu_high = max(mu_1, mu_2)
```

Result:

- Run:
  `runs/04_stress_label_switch/03_population_npe/02_flow2_residual_full_prior_512k_ensemble4_e30`
- Target:
  `z_sorted=(mu_low, mu_high, log_sigma)`.
- Recipe:
  4-member Flow2 residual NSF ensemble, `512k` full-prior simulations per
  member, `30` epochs on the Mac mini.
- Full-prior validation NLL:
  `-3.09250 +/- 0.00822`.
- Sorted-coordinate entropy floor:
  `-3.10112 +/- 0.00821`, estimated on the same 50k validation cache with a
  symmetric Gaussian-mixture importance estimator and the sorted-coordinate
  `log 2` fold factor.
- Gap:
  `0.00862`, or `0.74` combined standard errors. The paired cache still
  resolves the residual (`0.00862 +/- 0.00060`), so this is `near_floor`, not
  a strict floor hit.

Diagnostics:

- The README posterior overlay now includes exact finite grid, MCMC, and NPE
  layers in sorted target coordinates.
- Mean normalized marginal Wasserstein to the exact grid is `0.02729` for the
  NPE and `0.02979` for MCMC on the representative full-prior signal.

### 4. Ordered Two-Exponential Decay

This is now the remaining blocker. The floor-first part of the workflow exists,
but the single-decay/sign training recipe has not transferred successfully.

Current target coordinates:

```text
g(z) = (log(A1 + A2), log(A1/A2), log k1, log Delta k, log sigma)
```

The transform is invertible and has unit Jacobian, so NPE NLL and floor NLL are
reported in the same ridge coordinates without an additional correction.

Current floor probes:

```text
10k validation, seed 23  -3.28149 +/- 0.02423
10k validation, seed 31  -3.28349 +/- 0.02427
```

Both runs use the same posterior-centered Gaussian-mixture importance sampler.
The independent seeds agree at the reported precision, but this should still be
treated as a probe-scale floor until the final method is cross-checked or scaled.

Floor cross-check:

```text
4k validation, seed 20260731, Gaussian-mixture importance  -3.33831 +/- 0.03801
4k validation, seed 20260731, tempered SMC                -3.30120 +/- 0.03815
SMC - importance                                           0.03711, about 0.69 conservative combined SE
```

This does not make the floor final, but it is enough to show that the importance
floor is not obviously the source of the `~0.08` NPE miss. The SMC run used
`2048` particles, `96` cosine-spaced beta steps, `2` MH moves per step, and a
local proposal scale multiplier of `0.25`.

Artifacts:

- `runs/06_two_exponential/03_population_npe/00_floor_crosscheck_importance_4k_8192_seed20260731/results/two_exp_population_floor_summary.json`
- `runs/06_two_exponential/03_population_npe/00_floor_crosscheck_smc_4k_p2048_b96_seed20260731/results/two_exp_population_floor_summary.json`

Training probes tried:

```text
512k x 4 Flow2, 15 epochs       NLL -3.19327, gap 0.08823
512k x 4 Flow2, 30 epochs       NLL -3.19892, gap 0.08257
1.024M x 1 Flow2, 30 epochs     NLL -3.19045, gap 0.09104
512k x 1 Flow4, 30 epochs       NLL -3.17108, gap 0.11041
512k x 1 MAF4, 30 epochs        NLL -3.17836, gap 0.10314
512k x 1 augmented context      NLL -3.17555, gap 0.10595
128k x 1 CPU MDN8 smoke         NLL -3.01387, gap 0.26762
```

The best plain NPE result is the 4-member Flow2 30-epoch ensemble, and it is
still about `0.083` NLL units above the floor. A diverse ensemble over the
available probes improves only to about `-3.2036`, still roughly `0.078` above
the floor. This is qualitatively different from Linear6/Banana/Label Switching
and should be treated as a real miss, not an uncertainty issue.

Useful infrastructure completed:

- `train_sign_population_npe.py` can now sample, train, evaluate, and floor-probe
  `--model two_exp` on the full prior.
- The same harness also supports a second two-exponential floor estimator:
  prior-to-posterior tempered SMC with systematic resampling.
- Two-exponential sampling is chunked, avoiding the large profile-context memory
  spike seen in the 2M probe.
- The full Gaussian and MDN covariance heads in `npe_stage1_decay.py` now use
  dimension-generic Cholesky parameter counts, fixing the prior 5D MDN bug.
- `plot_broad_efficiency_training_curves.py` has a reusable two-exponential
  population-loss mode ready once there is a result worth documenting.

Next viable experiments:

- Stop scaling the same Flow2 recipe blindly; the 1M single-member probe did not
  improve fixed-cache NLL.
- Try a genuinely richer conditional posterior family, such as a conditional
  mixture of flows or mixture-of-experts, rather than another single normalizing
  flow.
- Revisit target/context design with an invertible transform that separates the
  ridge more cleanly than the current log-sum/log-ratio coordinates.
- Use exact-posterior or high-quality importance samples for a subset of signals
  to diagnose where the amortized posterior misses mass before launching another
  Mac mini training run.

Diagnostics still required before any eventual pass:

- MCMC/HMC agreement as a sampler sanity check.
- Posterior predictive overlays and pairwise marginals for representative
  full-prior signals.
- A final held-out validation cache and floor estimate with uncertainty small
  enough to resolve a near-floor gap.

## Sign And Single-Decay Follow-Up

Sign and single decay are both `near_floor` results, not strict
global-faithfulness passes. To turn sign into a clean global result:

- rerun the same 4-member recipe at `2.048M` simulations per member;
- or train a second 4-member `512k` ensemble and evaluate an 8-member
  equal-weight ensemble;
- then fit convex weights only on a separate validation split and report a
  held-out/full-cache NLL.

To turn single decay into a clean global result:

- extend the Flow2 residual ensemble data scale beyond `2.048M` per member;
- or increase ensemble size and evaluate equal-weight and separately fitted
  convex-weighted mixtures on held-out/full-cache NLL;
- or improve the floor and validation estimates enough to decide whether the
  current `~2.1` SE gap is real.

These can be lower priority than Linear6/Banana if the immediate objective is
to transfer the recipe to uncalibrated models. They should still remain on the
remaining-work list until their NLL gaps are statistically indistinguishable
from the floor.

## Decision Rules

Use the same stop/go logic for every model.

| Result | Action |
| --- | --- |
| NLL gap `<= 2` combined SE and diagnostics look sane | Mark the model globally faithful in docs. |
| NLL gap between `2` and `4` combined SE | Treat as near-floor; increase data scale or ensemble size before changing architecture. |
| NLL gap clearly positive after `2.048M` per member | Improve context/target transform before running more data. |
| NLL close to floor but posterior overlays fail | Add panel posterior diagnostics; do not claim global faithfulness from NLL alone. |
| Floor estimator uncertainty is comparable to the NPE gap | Improve the floor estimator before changing the NPE. |

## Immediate Next Work

1. Treat Ordered Two-Exponential Decay as the only remaining unresolved model.
2. Commit and push successful model slices as they finish; for Two-Exponential,
   commit only reusable infrastructure or clearly labeled blocker/probe notes
   until there is a near-floor result.
3. Move the next two-exponential training attempt to a richer posterior family
   or a better invertible target/context; another plain Flow2 scale-up is not
   justified by the current probes.
4. For any new posterior figures, keep the exact/reference layer in the same
   reusable renderer and use deterministic seeded NPE samples.
