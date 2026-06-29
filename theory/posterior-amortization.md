# Posterior Amortization Definition

## Core Definition

For this repo, an inference method demonstrates posterior amortization only when
a single fitted estimator

```text
q_phi(theta | x)
```

is reused across a declared distribution of observations without retraining,
proposal refitting, reference-chain tuning, or observation-specific density
optimization.

Let `mu(x)` be the observation distribution over which reuse is claimed. This can
be the prior predictive distribution, a restricted benchmark region, or a local
neighborhood around an observed dataset `x_o`, but it must be declared before
evaluation. A posterior estimator is amortized over `mu` if its risk

```text
R_mu(q_phi; D) = E_{x ~ mu} [ D(p(theta | x), q_phi(theta | x)) ]
```

is small on held-out observations according to a posterior discrepancy `D` that
is meaningful for the scientific target.

This is stronger than showing that `q_phi(theta | x_o)` is close to a reference
posterior for one fixed observation. A one-observation result shows local
posterior faithfulness. It does not by itself show amortization.

## What Counts As The Observation Distribution

Use these labels consistently:

- Global prior amortization: `mu(x) = p(x)` under the prior predictive. A single
  estimator must work over held-out observations drawn from the prior predictive.
- Region amortization: `mu` is a declared subset or reweighted version of the
  prior predictive, such as a synthetic benchmark panel or scientifically
  relevant parameter/data region. The restriction must be part of the claim.
- Local amortization: `mu` is concentrated near a target observation `x_o`.
  Local methods can still be amortized over that local region, but only if tested
  on multiple held-out observations from that region.
- Targeted inference: the method adapts simulations, proposals, or training to a
  single `x_o` and is evaluated only at that `x_o`. This can be useful and
  faithful, but it should not be called amortized posterior inference.

Sequential NPE/SNPE usually moves from global amortization toward targeted
inference. If proposals are adapted to `x_o`, the resulting estimator is not a
global amortized posterior unless it is independently tested across the
observation distribution being claimed.

## A Justifiable Repo Measure

The minimum empirical claim should be a panel metric:

```text
x_i ~ mu, i = 1,...,M
D_i = D(p(theta | x_i), q_phi(theta | x_i))
```

Report at least:

```text
mean(D_i), median(D_i), q90(D_i), max(D_i), pass_fraction
```

where the pass threshold is calibrated per model from reference Monte Carlo or
grid uncertainty. The existing decay threshold `0.034` is justified for the
decay grid/reference setup; it should not be inherited blindly by other models.

For tractable toy models in this repo, the preferred `D` is the same
model-specific reference discrepancy already used in the run summaries:

```text
mean normalized Wasserstein to a grid, MCMC, or HMC reference
```

For models without an exact/reference posterior, use calibration diagnostics over
held-out simulated observations:

- SBC ranks for marginal calibration.
- Expected coverage for credible-set calibration and overconfidence.
- TARP when only posterior samples are available.
- L-C2ST for local validation around a specific observation.

Calibration alone is not a full distance between paired posteriors. It is best
treated as a necessary reliability condition when reference posteriors are not
available.

## Amortization Gap

A separate question is whether failures come from amortization rather than from
the density family. Define an observation-averaged amortization gap as:

```text
Gap_mu(D)
  = E_{x ~ mu} D(p(theta | x), q_phi(theta | x))
    - E_{x ~ mu} inf_{psi_x in Q} D(p(theta | x), q_{psi_x}(theta))
```

The second term is an unamortized oracle fit in the same posterior family `Q`,
optimized separately for each observation. In this repo, oracle density fits to
posterior samples at a fixed `x` approximate this second term. If oracle fits are
close but simulation-trained NPE is not, the error is conditional-learning or
amortization error rather than posterior-family incapacity.

The current repo has evidence for this pattern on the exponential-decay problem:
the oracle posterior density fit gets close to the strict target, while
simulation-trained NPE remains worse at the same observation and broad multi-x
checks fail by a wide margin.

## Cost Amortization Is Conditional On Accuracy

There is also a computational use of "amortization": paying training cost once
and reusing the estimator for many observations. This is not sufficient unless
the posterior map is accurate over the claimed `mu`.

When the accuracy criterion passes, report the cost crossover:

```text
T_train + M * T_query < M * T_reference_per_observation
M_star = T_train / (T_reference_per_observation - T_query)
```

Only quote `M_star` for an observation region where the same trained estimator
passes the posterior accuracy and calibration criteria.

## Current Repo Status

The current artifacts do not yet demonstrate faithful global posterior
amortization.

- `scripts/evaluate_npe_multi_x.py` is the closest existing implementation of an
  amortization test: one trained Stage 1 model is queried on multiple held-out
  observations and compared to grid references.
- The recorded multi-x decay results did not pass the target for any tested
  family.
- One-observation NPE flow runs and SNPE/local runs test posterior faithfulness
  or targeted inference at `x_o`; they should not be used as evidence of global
  amortization.
- Legacy pairwise agreement against MCMC/HMC at one observation is useful, but it
  is not an amortization measure.

## Practical Acceptance Criterion

For a future run to claim amortized posterior inference in this repo, require:

1. Declare `mu`: prior predictive, benchmark region, or local region.
2. Train one estimator once, excluding the evaluation observations.
3. Evaluate at least `M` held-out observations from `mu`.
4. For tractable cases, compute per-observation reference discrepancies.
5. Report mean, median, q90, max, and pass fraction against a calibrated
   model-specific target.
6. Run SBC, expected coverage, TARP, or L-C2ST where feasible.
7. Report query time and cost crossover only if the accuracy criterion passes.

Until those criteria are met, the correct wording is "posterior approximation at
`x_o`", "local/targeted NPE", or "candidate amortized estimator", not
"demonstrated amortized posterior inference".

## Sources

- Papamakarios and Murray, "Fast epsilon-free Inference of Simulation Models
  with Bayesian Conditional Density Estimation": https://arxiv.org/abs/1605.06376
- Greenberg, Nonnenmacher, and Macke, "Automatic Posterior Transformation for
  Likelihood-Free Inference": https://proceedings.mlr.press/v97/greenberg19a.html
- Margossian and Blei, "Amortized Variational Inference: When and Why?":
  https://proceedings.mlr.press/v244/margossian24a.html
- Talts et al., "Validating Bayesian Inference Algorithms with Simulation-Based
  Calibration": https://arxiv.org/abs/1804.06788
- Lemos et al., "Sampling-Based Accuracy Testing of Posterior Estimators for
  General Inference": https://proceedings.mlr.press/v202/lemos23a.html
- Linhart, Gramfort, and Rodrigues, "L-C2ST: Local Diagnostics for Posterior
  Approximations in Simulation-Based Inference": https://arxiv.org/abs/2306.03580
- `sbi` L-C2ST guide: https://sbi.readthedocs.io/en/stable/how_to_guide/13_diagnostics_lc2st.html
- `sbi` SBC guide: https://sbi.readthedocs.io/en/stable/advanced_tutorials/11_diagnostics_simulation_based_calibration.html
