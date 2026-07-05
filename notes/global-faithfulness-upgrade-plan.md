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
| Banana | `legacy_pairwise_pass` | Estimate the full-prior floor, then train/evaluate a population NPE in raw coordinates. |
| Label-switching mixture | `legacy_pairwise_pass` | Estimate the sorted-coordinate floor, then train/evaluate a population NPE in symmetry-aware coordinates. |
| Linear6 | `legacy_pairwise_pass` | Estimate the full-prior floor using the linear-Gaussian structure, then train/evaluate a population NPE. |
| Ordered two-exponential decay | `fail` | First build a reliable full-prior floor/evidence pipeline; only then claim or tune global NLL. |

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

Do this first. It is the best candidate for a clean global result because most
of the floor can be computed with model structure instead of brute force.

Target coordinates:

```text
g(z) = (w_1, ..., w_6, sigma)
```

Floor plan:

- Use the linear-Gaussian likelihood conditional on `sigma`.
- Integrate over `log sigma` with one-dimensional adaptive quadrature.
- Conditional on each `sigma`, use the Gaussian posterior for the weights.
- Include the `log sigma -> sigma` Jacobian if reporting density in `sigma`.
- Validate the floor estimator against direct high-precision importance or HMC
  checks on a small validation subset.

Training plan:

- Context: raw 32-point signal plus cheap OLS summaries:
  `(w_hat_1, ..., w_hat_6, log sigma_hat, residual norm, condition diagnostics)`.
- Target: physical `sigma` only if the floor is also in physical `sigma`;
  otherwise use `log sigma` consistently.
- Start with the 4-member Flow2 recipe at `512k` per member.
- Promote to `2.048M` per member if the gap is close but still resolved.

Promotion gate:

```text
ensemble NLL - floor <= 2 combined SE
```

If NLL passes but posterior diagnostics show rare failures, add a panel
Wasserstein distribution over prior-predictive signals before marking the model
globally faithful.

### 2. Banana

This should be the second target because it is two-dimensional and exact-grid
reference calculations are practical.

Target coordinates:

```text
g(theta) = (theta_1, theta_2)
```

Floor plan:

- For each validation pair, compute
  `log p(theta | x) = log p(theta) + log p(x | theta) - log p(x)`.
- Estimate `log p(x)` using a 2D adaptive grid or quadrature over the prior
  support.
- Cross-check the evidence on a subset with dense grids and sampler-based
  bridge/importance estimates.

Training plan:

- Context: raw `x` plus a dewarped approximate inverse summary, for example
  `(x_1, x_2 - b(x_1^2 - c))`, and standardized variants.
- Target: raw coordinates, matching the floor.
- Start with the 4-member Flow2 recipe at `512k` per member.

Diagnostics:

- Exact grid vs NPE vs MCMC on at least one fresh prior-predictive signal.
- A small prior-predictive panel if the NLL gap is small but shape errors are
  visible.

### 3. Label-Switching Mixture

This needs a symmetry-aware target before any NLL number is meaningful.

Target coordinates:

```text
g(z) = (mu_low, mu_high, sigma)
mu_low = min(mu_1, mu_2)
mu_high = max(mu_1, mu_2)
```

Floor plan:

- Compute the posterior density in sorted coordinates by summing raw posterior
  density over the two label permutations:

  ```text
  p_sorted(mu_low, mu_high, sigma | x)
    = p_raw(mu_low, mu_high, sigma | x)
    + p_raw(mu_high, mu_low, sigma | x)
  ```

- Include the `log sigma -> sigma` Jacobian if using physical `sigma`.
- Estimate the evidence `p(x)` using adaptive 3D integration, importance
  sampling around EM modes, or bridge sampling from a converged HMC reference.
- Validate symmetry mass and evidence stability on a small cache before running
  the final validation.

Training plan:

- Train directly in sorted coordinates.
- Restore random label assignment only for raw-coordinate posterior displays,
  not for the NLL target.
- Context: use the existing EM summaries, plus robust raw-data summaries such
  as quantiles or a fixed histogram if they improve validation NLL.
- Start with the 4-member Flow2 recipe at `512k` per member.

Diagnostics:

- Sorted-coordinate exact/reference vs NPE overlay.
- Mode/permutation mass check in raw coordinates as a display diagnostic.

### 4. Ordered Two-Exponential Decay

This is the hard case. Do not spend large training runs before the full-prior
floor/evidence method is credible.

Coordinate decision:

- Choose one NLL target and keep it fixed:
  either ordered log coordinates
  `(log A_1, log k_1, log A_2, log Delta k, log sigma)` or displayed physical
  coordinates `(A_1, k_1, A_2, k_2, sigma)`.
- Physical coordinates are nicer for plots but require more Jacobian bookkeeping.
- Log coordinates are likely more stable for NLL and floor estimation.

Floor plan:

- Start with a `10k` to `50k` validation floor probe, not `1M`.
- For each signal, build a proposal from multi-start profiled two-rate
  least-squares fits plus local curvature.
- Estimate `log p(x)` with adaptive importance, bridge sampling, or SMC/AIS.
- Compare evidence estimates across at least two independent numerical methods
  on a smaller subset.
- Only scale the validation cache after evidence error is clearly below the
  expected NPE-floor gap.

Training plan:

- Use the single-decay Flow2 recipe as the first controlled baseline, but keep
  the model-specific context that worked best so far:
  profiled two-rate least-squares summaries, ridge/profiling diagnostics, and
  raw curve if it helps.
- Train a residual target around the profiled fit if the coordinate decision
  allows an invertible target transform with a known Jacobian.
- If Flow2 at `512k` is far from the floor, escalate in this order:
  `2.048M` data, then Flow3/Flow4, then density ensembles or mixture-of-experts.

Diagnostics:

- MCMC/HMC agreement remains required as a sampler sanity check, but it is not a
  global pass criterion.
- Use posterior predictive overlays and pairwise marginal diagnostics to find
  ridge or mode failures.

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

1. Extract the reusable parts of `train_sign_population_npe.py` into a generic
   population-NPE training/evaluation harness with model hooks for sampler,
   context, target transform, and floor metadata.
2. Implement the Linear6 entropy-floor estimator using the linear-Gaussian
   conditional structure.
3. Run the Linear6 4-member Flow2 `512k` proof.
4. Extend `plot_broad_efficiency_training_curves.py` and the README posterior
   renderer with a `linear6_population` mode instead of adding separate plotting
   scripts.
5. Update the Linear6 README section with the full-prior NLL/floor result and
   one fresh prior-predictive posterior diagnostic.
