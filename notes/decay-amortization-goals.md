# Decay Model Amortization Goals

Date: 2026-06-25

## Purpose

Define an ordered set of amortization goals for the exponential-decay model,
from easiest to hardest, ending with global prior-predictive amortization.

The model is:

```text
theta = (A, k, sigma)
x_j = A exp(-k t_j) + epsilon_j
epsilon_j ~ Normal(0, sigma^2)
```

The strict reference target currently used for the original observed curve
`x_o` is:

```text
mean normalized Wasserstein <= 0.034
```

This target is calibrated from grid/MCMC/HMC agreement at `x_o`. It is useful as
a single-observation target and as a smoke-test benchmark, but it is not assumed
to be representative for arbitrary held-out observations `x_i`.

For amortization panels, the primary target should be observation-specific:

```text
tau_i = calibrated reference noise / solver discrepancy for x_i
ratio_i = D_i / tau_i
```

where `D_i` is the NPE-to-reference posterior discrepancy for observation `x_i`.
Panel claims should be made on `ratio_i`, not on a globally reused `0.034`
threshold.

## Current Status

We have not yet demonstrated global amortization.

What we do have:

- A single-observation local-prior-filtered flow run reached
  `mean normalized Wasserstein = 0.03310` at `x_o`.
- That run used `training_mode = local_prior` and `local_quantile = 0.005`, so it
  was not trained on the full prior predictive.
- Broad prior-predictive Stage 1 NPE did not meet the target at `x_o`.
- Broad multi-`x` checks failed on all tested held-out observations.

Therefore the current result is:

```text
local posterior faithfulness at one x_o, not demonstrated amortization
```

## Common Evaluation Contract

Every amortization claim should use the same basic panel structure:

```text
1. Declare the evaluation distribution mu(x).
2. Train one estimator q_phi(theta | x) once.
3. Draw held-out observations x_i ~ mu, excluded from training.
4. For every x_i, build a grid/HMC/MCMC reference posterior.
5. Compute D_i = mean normalized Wasserstein(q_phi(theta | x_i), reference_i).
6. Compute an observation-specific tolerance `tau_i` from reference noise.
7. Compute `ratio_i = D_i / tau_i`.
8. Report discrepancy, tolerance, ratio, pass fraction, and per-dimension
   diagnostics.
```

For each held-out observation, the claim-grade tolerance should be:

```text
tau_i = max(
  MCMC_i_to_grid_i,
  HMC_i_to_grid_i,
  MCMC/HMC split diagnostics,
  grid resample noise floor
)
```

The grid reference must also record edge mass. If edge mass is not negligible,
the grid must be widened or refined before using `tau_i`.

For development runs, it is acceptable to use cheaper grid-only tolerances:

```text
tau_i_grid = max(grid sample-to-grid, grid sample pairwise)
```

Default panel sizes:

```text
smoke: M = 8
development: M = 32
claim-grade: M = 64 or more
```

Default pass summaries:

```text
primary:   median(ratio_i), q90(ratio_i), pass_fraction(ratio_i <= 1)
secondary: mean(D_i), median(D_i), q90(D_i), max(D_i), tau_i distribution
```

Suggested claim labels:

- `single_x_faithful`: only `x_o` passes.
- `weak_amortized`: median ratio passes but q90 or pass fraction does not.
- `amortized`: q90 ratio is at most 1 and pass fraction is at least 0.90.
- `strong_amortized`: max ratio is at most 1, or within an explicitly
  calibrated MC-noise allowance.

The exact pass rule can be tightened after measuring grid/reference noise across
the panel.

## Goal 0: Build The Amortization Panel Harness

Question:

```text
Can we evaluate one fixed trained NPE on many held-out observations with
references built per observation?
```

Required work:

- Generalize or replace `scripts/evaluate_npe_multi_x.py` so it can evaluate:
  - broad prior-predictive trained models;
  - local-prior-filtered flow models;
  - parameter-region models;
  - future architectures with the same output schema.
- Save one summary JSON per panel with:
  - evaluation distribution metadata;
  - model checkpoint path and training distribution metadata;
  - per-observation reference diagnostics;
  - per-observation tolerance `tau_i`;
  - per-observation NPE discrepancy;
  - per-observation ratio `D_i / tau_i`;
  - aggregate mean, median, q90, max, pass fraction for both `D_i` and
    `ratio_i`.
- Save figures:
  - discrepancy box/strip plot;
  - discrepancy versus region distance;
  - worst-case corner overlays for at least the top 3 failures.

This goal proves no amortization by itself. It creates the measurement surface.

Implementation status:

- Initial script added:
  `scripts/evaluate_decay_amortization_panel.py`.
- Current supported model kinds:
  - `stage1`: loads saved Stage 1 checkpoints by family.
  - `flow_decay`: loads the saved conditional spline-flow checkpoint.
- Current supported panel distributions:
  - `x0`, the original observed curve from `--observed-seed`;
  - `prior_predictive`;
  - `local_x`, using summary-distance filtering around `x_o`.
  - `parameter_region`, using a prior-covariance Mahalanobis ball in
    `z = log(theta)` space.
- Current supported reference priors:
  - full original prior;
  - restricted parameter-region prior for `parameter_region` panels.
- Current grid reference sources:
  - `adaptive`, built per observation from model posterior samples and the true
    theta used to simulate the observation;
  - `x0_mcmc_hmc`, the legacy single-`x_o` grid built from saved MCMC/HMC
    posterior samples. This is only valid for `panel_distribution = x0`.
- Current tolerance mode:
  - `grid_only`, using grid sample-to-grid and grid sample-pairwise noise.
  - `mcmc_hmc`, using per-observation MCMC/HMC-to-grid, sampler split
    diagnostics, MCMC/HMC pairwise discrepancy, and grid resample noise.
- `mcmc_hmc` tolerances only count as valid when MCMC and HMC convergence flags
  pass and HMC has no divergences.
- The evaluator can restrict expensive evaluation to selected original panel
  indices with `--observation-indices`, which is useful for debugging or
  calibrating worst observations.
- The evaluator can evaluate an equal-weight `flow_decay` ensemble with
  `--flow-checkpoints`, concatenating equal posterior sample counts from each
  checkpoint.
- Aggregate summaries include valid-tolerance-only ratio fields so invalid HMC
  or MCMC runs cannot artificially lower median or q90 ratio claims.
- `scripts/npe_flow_decay.py` can train a `local_prior` flow on a saved
  `local_training.region` via `--local-region-summary`, allowing training and
  evaluation to target exactly the same declared local observation region.
- Claim-grade panel runs with production sampler budgets have not been run yet.

Smoke artifacts:

- Stage 1 MDN on a 2-observation prior-predictive panel:
  `runs/01_exponential_decay/07_amortization_panels/00_smoke_stage1_prior/results/decay_amortization_panel_summary.json`.
- Local spline flow on a 2-observation local-`x` panel:
  `runs/01_exponential_decay/07_amortization_panels/00_smoke_flow_local/results/decay_amortization_panel_summary.json`.
- Stage 1 MDN on a 1-observation prior-predictive panel with deliberately tiny
  MCMC/HMC budgets:
  `runs/01_exponential_decay/07_amortization_panels/00_smoke_stage1_prior_mcmchmc/results/decay_amortization_panel_summary.json`.
- Stage 1 MDN on a 1-observation parameter-region panel with a restricted-prior
  grid reference:
  `runs/01_exponential_decay/07_amortization_panels/00_smoke_stage1_parameter_region/results/decay_amortization_panel_summary.json`.

These smoke runs only verify the harness plumbing. They are deliberately too
small and use coarse grids or unconverged sampler budgets, so they are not
amortization evidence.

## Goal 1: Reproduce Single-Observation Local Faithfulness

Evaluation distribution:

```text
mu = delta_{x_o}
```

Training distribution:

```text
theta ~ p(theta)
x ~ p(x | theta)
keep x if d_s(x, x_o) is in the closest 0.5% of prior-predictive summaries
```

Distance:

```text
s(x) = (log A_hat(x), log k_hat(x), log sigma_hat(x))
d_s(x, x_o) = || L^{-1}(s(x) - s(x_o)) ||
```

where `L L^T` is the prior-predictive covariance of `s(x)`.

Target:

```text
D(x_o) <= 0.034
```

This reproduces the current best result and checks that the pipeline remains
stable. It still does not demonstrate amortization.

Implementation status:

- The original saved run remains:
  `runs/01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706/results/npe_flow_decay_summary.json`.
- Directly recomputing the old metric from the saved posterior samples and the
  old MCMC/HMC-sample grid gives:

```text
D = 0.03310209372023741
```

- Re-evaluating the saved checkpoint through the panel harness with a fresh
  180,000-sample posterior draw and the legacy `x0_mcmc_hmc` grid gives:

```text
D = 0.032914963148634575
```

  Artifact:
  `runs/01_exponential_decay/07_amortization_panels/01_flow_x0_reproduction_legacy_reference/results/decay_amortization_panel_summary.json`.
- Re-evaluating the same checkpoint with the harness's default adaptive grid
  gives:

```text
D = 0.0418174267372301
```

  Artifact:
  `runs/01_exponential_decay/07_amortization_panels/01_flow_x0_reproduction/results/decay_amortization_panel_summary.json`.
- Interpretation: Goal 1 is reproduced under the legacy reference used by the
  original claim. The adaptive-reference result is a stricter changed-reference
  check and should not be mixed with the old `0.034` target without
  recalibrating that target.
- The panel harness also reports grid-only `tau` and `D / tau`. For the legacy
  reference reproduction, `tau_grid = 0.00813420776462096` and
  `D / tau_grid = 4.046486652553355`. This does not contradict the original
  pass, because the original `0.034` target was based on MCMC/HMC-to-grid
  agreement, not on grid-resampling noise alone.

## Goal 2: Local Observation-Region Amortization Around x_o

Evaluation distribution:

```text
mu_local_q(x) = prior predictive restricted to d_s(x, x_o) <= r_q
```

where `r_q` is a prior-predictive distance quantile. Start with:

```text
q in {0.001, 0.0025, 0.005, 0.01}
```

Training distribution:

Same as the evaluation region, but with disjoint simulation seeds and held-out
panel observations.

Why this is statistically clean:

```text
p(theta | x, x in region) = p(theta | x)
```

because the region event depends only on `x`. The original prior posterior is
not changed.

Success criterion:

```text
claim-grade panel: M >= 64
q90(ratio_i) <= 1
pass_fraction(ratio_i <= 1) >= 0.90
```

What this proves:

```text
one trained estimator works across a declared neighborhood of x_o
```

What it does not prove:

```text
the estimator works over the full prior predictive
```

Development status:

- The panel harness can now reuse the exact saved local region from:
  `runs/01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706/results/npe_flow_decay_summary.json`.
- A first M=8 grid-only development panel used:

```text
panel_distribution = local_x
region_source = summary
local_quantile = 0.005
posterior_samples = 180000
grid_size = 90
grid_sample_count = 60000
grid_replicates = 8
```

  Artifact:
  `runs/01_exponential_decay/07_amortization_panels/02_flow_local_q0005_m8_grid/results/decay_amortization_panel_summary.json`.
- Accepted local distances were:

```text
min = 0.10149941031288813
median = 0.11984108005030847
max = 0.14341317418656857
radius = 0.1458919877944998
```

- The saved local spline flow did not pass this grid-only development check:

```text
median(D_i) = 0.04804349594721249
q90(D_i)    = 0.06177640151497422
max(D_i)    = 0.08462342232710117

median(D_i / tau_i_grid) = 6.744835334459722
q90(D_i / tau_i_grid)    = 8.988513817933256
pass_fraction            = 0 / 8
```

- Interpretation: this is evidence that the single-`x_o` local-flow result does
  not automatically extend across even the q=0.005 local observation region.
  This is still not claim-grade because `tau_i_grid` is only a grid-resampling
  noise floor. The next check should calibrate `tau_i` with MCMC/HMC for at
  least the worst local observations, then decide whether to retrain or adjust
  the local-region architecture.

- A calibrated M=8 development panel used per-observation MCMC/HMC tolerances:

```text
tolerance_mode = mcmc_hmc
hmc_step_size = 0.0045
mcmc_chains = 8
mcmc_steps = 24000
hmc_chains = 8
hmc_steps = 5000
```

  Artifact:
  `runs/01_exponential_decay/07_amortization_panels/06_flow_local_q0005_m8_mcmchmc_hmc0045/results/decay_amortization_panel_summary.json`.
- All eight observation-specific tolerances were valid:

```text
valid_tolerance_count = 8 / 8
HMC divergences       = 0 for every observation
MCMC/HMC flags        = pass for every observation
```

- Calibrated result:

```text
median(D_i) = 0.04747695887583157
q90(D_i)    = 0.06112943827812506
max(D_i)    = 0.08334397100519687

valid median(D_i / tau_i) = 0.9868496429963596
valid q90(D_i / tau_i)    = 1.2255568602569367
valid max(D_i / tau_i)    = 1.397099605323478
valid pass_fraction       = 4 / 8
```

- Worst calibrated observation:

```text
index = 1
distance_to_x0 = 0.1187152140086689
D_i = 0.08334397100519687
tau_i = 0.059654995740908386
D_i / tau_i = 1.397099605323478
```

- Interpretation: under calibrated tolerances, the saved q=0.005 local spline
  flow is near the boundary but still does not demonstrate local amortization.
  It fails both the q90 ratio target and the pass-fraction target on this small
  development panel. We should not widen the local observation region until a
  stronger q=0.005 estimator passes a calibrated panel.

- A larger same-region candidate was trained with:

```text
training_mode = local_prior
local_region_summary = saved q=0.005 region
train_simulations = 250000
val_simulations = 60000
transforms = 12
hidden_features = 256,256
seed = 20260722
```

  Checkpoint:
  `runs/01_exponential_decay/03_npe_flow_search/20_npe_flow_local_q0005_reuse_region_250k_t12_seed20260722/results/npe_flow_decay_model.pt`.
  The run saved model/samples/figures, but its training summary JSON was not
  written because `local_region_summary` was initially not JSON-serializable;
  the serialization bug is fixed in the script.
- Its saved posterior samples give a single-`x_o` legacy-reference discrepancy
  of:

```text
D(x_o) = 0.03238770585523638
```

  This is slightly better than the earlier `0.03310209372023741` single-point
  result, so the candidate did not fail because it lost faithfulness at `x_o`.
- The same calibrated M=8 panel for this larger candidate gave:

```text
artifact =
runs/01_exponential_decay/07_amortization_panels/07_flow_local_q0005_candidate20_m8_mcmchmc_hmc0045/results/decay_amortization_panel_summary.json

valid_tolerance_count = 8 / 8
valid median(D_i / tau_i) = 1.1315143565007786
valid q90(D_i / tau_i)    = 1.520316916220852
valid max(D_i / tau_i)    = 1.614542349842414
valid pass_fraction       = 1 / 8
```

- Interpretation: this larger model is worse than the saved 150k/t8 local flow
  on the calibrated local panel despite being slightly better at the single
  `x_o`. More capacity and more same-region simulations did not solve the local
  amortization problem by themselves. The next local attempts should change the
  training objective, regularization/early stopping, ensembling, or summary
  representation rather than only scaling this same architecture.

- A three-member equal-weight ensemble of comparable q=0.005 local flows was
  evaluated:

```text
members =
  runs/01_exponential_decay/03_npe_flow_search/09_npe_flow_local_q0005_linear_100k_t8/results/npe_flow_decay_model.pt
  runs/01_exponential_decay/03_npe_flow_search/10_npe_flow_local_q0005_linear_100k_t8_seed20260703/results/npe_flow_decay_model.pt
  runs/01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706/results/npe_flow_decay_model.pt

posterior_samples = 180000 total
tolerance_mode = mcmc_hmc
hmc_step_size = 0.0045
```

  Artifact:
  `runs/01_exponential_decay/07_amortization_panels/08_flow_local_q0005_ensemble_09_10_11_m8_mcmchmc_hmc0045/results/decay_amortization_panel_summary.json`.
- All eight observation-specific tolerances were valid:

```text
valid_tolerance_count = 8 / 8
HMC divergences       = 0 for every observation
MCMC/HMC flags        = pass for every observation
```

- Calibrated ensemble result:

```text
median(D_i) = 0.042651631163329665
q90(D_i)    = 0.04690158213866844
max(D_i)    = 0.050942556513806

valid median(D_i / tau_i) = 0.8856298874979818
valid q90(D_i / tau_i)    = 0.9854664617702457
valid max(D_i / tau_i)    = 0.985919714435703
valid pass_fraction       = 8 / 8
```

- Interpretation: ensembling is the first development result that passes the
  q=0.005 local M=8 calibrated panel. This is not yet a claim-grade local
  amortization result because the panel is too small. The next step should be a
  larger q=0.005 local panel, ideally M >= 32 for development and then M >= 64
  for a claim, before widening to q > 0.005.

## Goal 3: Expand The Observation Region

Question:

```text
How fast does amortized accuracy degrade as the local x-region widens?
```

Evaluation distributions:

```text
mu_local_q for q in {0.005, 0.01, 0.02, 0.05, 0.10}
```

Training:

Train one estimator per region width. Keep architecture and optimizer fixed
where possible so the region-width curve is interpretable.

Report:

```text
q versus median(D_i), q90(D_i), max(D_i), tau_i distribution,
q versus median(ratio_i), q90(ratio_i), max(ratio_i), pass_fraction
```

Useful outcomes:

- If only tiny `q` works, global amortization probably needs a different
  architecture or much larger budget.
- If the pass curve degrades smoothly, we can plan scaling with region width.
- If performance is non-monotonic, diagnose optimization instability,
  summary insufficiency, or data imbalance.

## Goal 4: Parameter-Region Amortization

Evaluation distribution:

```text
z = log(theta)
z ~ p(z | z in A)
x ~ p(x | z)
```

where `A` is a predeclared region in log-parameter space, for example a
Mahalanobis ball under the prior covariance.

Candidate regions:

```text
A_small:  d_prior(z, z_center) <= c_small
A_medium: d_prior(z, z_center) <= c_medium
A_wide:   selected prior box or prior quantile slab
```

Use two versions:

1. `z_center` set to the known synthetic truth for diagnostic work.
2. `z_center` set to a predeclared scientifically relevant region, independent
   of `x_o`, for a non-oracle regional claim.

Important interpretation:

Training on a restricted parameter distribution changes the Bayesian target:

```text
p_A(theta | x) proportional to p(x | theta) p(theta) 1{theta in A}
```

So this is valid regional amortization under a restricted prior. It is not a
claim about the original full-prior posterior unless proposal/prior correction is
added.

Success criterion:

Same observation-specific ratio criterion as Goal 2, but references must use the
same restricted prior `p_A(theta)`.

Implementation status:

- The panel evaluator can now generate `parameter_region` held-out observations
  and compare to a restricted-prior grid reference.
- Dedicated NPE training on the same restricted parameter distribution has not
  been implemented yet, so current parameter-region smoke runs are evaluator
  checks, not fair restricted-prior amortization claims.

What this proves:

```text
the NPE map can be amortized over a controlled part of parameter space
```

Why it is useful:

This separates "global prior too broad" from "conditional density learning is
hard even in a well-controlled parameter region."

## Goal 5: Broad Prior-Predictive Amortization Baseline

Evaluation distribution:

```text
mu(x) = p(x) = integral p(x | theta) p(theta) dtheta
```

Training distribution:

```text
theta ~ p(theta)
x ~ p(x | theta)
```

No `x_o` is used for training or evaluation distribution construction.

Initial training sweep:

- stronger conditional spline flow than Stage 1;
- indirect 3D fit summaries;
- enhanced residual summaries;
- learned embedding over raw curves if summaries saturate;
- ensembles if single seeds are unstable;
- larger simulation budgets in controlled increments.

Panel:

```text
M >= 64 held-out prior-predictive observations
```

Success criterion:

```text
q90(ratio_i) <= 1
pass_fraction(ratio_i <= 1) >= 0.90
```

What this proves:

```text
global amortization over the original prior predictive for the decay model
```

This is the ultimate goal for this model.

## Goal 6: Global Amortization With Calibration

Once Goal 5 passes reference discrepancies, run independent calibration checks:

- SBC ranks over held-out prior-predictive pairs.
- Expected coverage curves.
- TARP if only posterior samples are needed.
- Posterior predictive checks for worst-case panel observations.

Success criterion:

```text
reference discrepancy passes and calibration does not show material bias,
overconfidence, or undercoverage
```

What this adds:

Goal 5 says the estimator matches references relative to observation-specific
reference noise on a panel. Goal 6 says it behaves like a reliable Bayesian
posterior estimator over the same distribution.

## Goal 7: Cost Amortization And Crossover

Only after posterior accuracy passes, measure the computational value:

```text
T_train + M * T_query
versus
M * T_reference_per_observation
```

Report:

```text
M_star = T_train / (T_reference_per_observation - T_query)
```

Do this separately for:

- local observation-region amortization;
- parameter-region amortization;
- global prior-predictive amortization.

Do not claim computational amortization unless the corresponding posterior
accuracy goal has already passed.

## Recommended Execution Order

1. Goal 0: build the panel harness.
2. Goal 1: reproduce the single-`x_o` local pass.
3. Goal 2: prove or disprove local observation-region amortization at `q=0.005`.
4. Goal 3: expand the local region and measure degradation.
5. Goal 4: run parameter-region amortization as a controlled diagnostic.
6. Goal 5: return to full prior-predictive amortization with lessons from Goals
   2-4.
7. Goal 6: add calibration once reference discrepancy passes.
8. Goal 7: quantify speed/cost crossover only after accuracy passes.

## Decision Points

After Goal 2:

- If local observation-region amortization fails, do not widen the region yet.
  Improve architecture, summaries, optimization, or ensemble strategy locally.
- If it passes, widen region gradually.

After Goal 4:

- If parameter-region amortization passes but x-region amortization fails, the
  observation summaries are likely insufficient or noisy.
- If x-region amortization passes but parameter-region amortization fails, the
  restricted-prior target or reference construction needs checking.
- If both fail, focus on conditional density capacity and optimization.

After Goal 5:

- If global prior-predictive amortization fails while smaller regions pass,
  report a measured scaling boundary rather than treating the experiment as a
  total failure.
- If global prior-predictive amortization passes, move immediately to
  calibration and cost crossover before making a final claim.

## Final Claim We Want To Earn

The strongest target statement is:

```text
For the exponential-decay model and declared prior, one trained NPE estimator
q_phi(theta | x), trained once on prior-predictive simulations, produces
posterior samples whose per-observation calibrated reference discrepancy ratios
and calibration diagnostics pass on held-out prior-predictive observations.
```

Anything weaker should be named by its actual scope:

- single-observation posterior faithfulness;
- local observation-region amortization;
- parameter-region amortization;
- restricted-prior amortization;
- global prior-predictive amortization.
