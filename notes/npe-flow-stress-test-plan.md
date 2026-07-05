# NPE Flow Stress-Test Plan

Goal: test whether the successful decay-model recipe transfers to harder posteriors with minimal adjustments, while requiring agreement between plain MCMC, HMC, and normalizing-flow NPE.

Successful baseline from the single-exponential decay example:

```text
local amortization near x0
+ conditional spline-flow NPE
+ linear target preconditioning
+ enough local simulations
```

The final posterior must remain a normalizing-flow NPE posterior:

```math
q_\phi(\theta \mid s(x_0))
```

ABC correction may be used only as a diagnostic or reference fallback, not as the final posterior.

The core requirement for every stress test is:

```math
p_{\mathrm{MCMC}}(\theta \mid x_0)
\approx
p_{\mathrm{HMC}}(\theta \mid x_0)
\approx
q_\phi(\theta \mid s(x_0)).
```

MCMC and HMC use the same exact likelihood and prior in the toy problems. NPE may use local amortization and target preconditioning, but its delivered posterior must still be sampled directly from a conditional normalizing flow.

## Common Evaluation

For each case:

- simulate one observed dataset \(x_0\)
- run plain random-walk Metropolis MCMC
- run HMC with the same log posterior
- train a conditional spline-flow NPE
- sample from \(q_\phi(\theta\mid s(x_0))\)
- verify MCMC and HMC convergence using:
  - acceptance rates
  - rank-normalized \(\hat R\)
  - bulk and tail ESS
  - trace plots
- compare all three posteriors using:
  - marginal normalized Wasserstein
  - sliced Wasserstein where useful
  - posterior means, standard deviations, and credible intervals
  - posterior predictive checks
  - mode-mass checks for multimodal cases

For low-dimensional cases, grid references are also useful sanity checks, but the operational agreement target is still three-way: MCMC, HMC, and NPE must agree.

For low-dimensional cases, the target remains:

```text
mean normalized Wasserstein <= 0.034
```

For higher-dimensional cases, the strict 1D marginal target is still reported, but the decision also considers sliced Wasserstein and posterior predictive agreement.

## Case 1: Low-Dimensional Sign Multimodality

Simulator:

```math
x_1 = \theta_1^2 + \epsilon_1,
\qquad
x_2 = \theta_2 + \epsilon_2.
```

Expected posterior:

```math
\theta_1 \approx +\sqrt{x_1}
\quad\text{or}\quad
\theta_1 \approx -\sqrt{x_1}.
```

Purpose:

- verify that conditional flows can keep two separated modes
- measure mode mass error
- use an exact grid reference

## Case 2: Banana / Curved Degeneracy

Simulator:

```math
x_1 = \theta_1 + \epsilon_1,
\qquad
x_2 = \theta_2 + b(\theta_1^2-c) + \epsilon_2.
```

Purpose:

- test nonlinear posterior geometry
- use exact grid reference
- verify spline flow improves over Gaussian/affine approximations

## Case 3: Label-Switching Mixture

Simulator:

```math
x_i \sim
\frac12 \mathcal N(\mu_1,\sigma^2)
+
\frac12 \mathcal N(\mu_2,\sigma^2).
```

Parameter:

```math
\theta=(\mu_1,\mu_2,\log\sigma).
```

Expected posterior symmetry:

```math
(\mu_1,\mu_2)
\leftrightarrow
(\mu_2,\mu_1).
```

Purpose:

- test label-switching multimodality
- measure mass on each label permutation
- compare ordered and unordered variants if needed

## Case 4: Higher-Dimensional Smooth Posterior

Simulator:

```math
y(t)
=
\sum_{j=1}^{d} w_j \phi_j(t)
+
\epsilon.
```

Parameter:

```math
\theta=(w_1,\ldots,w_d,\log\sigma).
```

Purpose:

- isolate dimension from multimodality
- start with \(d_\theta=6\) or \(8\)
- compare NPE to a Gaussian analytic/reference posterior when possible

## Case 5: Two-Exponential Decay

Simulator:

```math
y(t)
=
A_1 e^{-k_1 t}
+
A_2 e^{-k_2 t}
+
\epsilon.
```

Parameter:

```math
\theta=(A_1,k_1,A_2,k_2,\sigma).
```

Variants:

1. ordered rates:

```math
k_1 < k_2
```

2. unordered rates, which allows label switching:

```math
(A_1,k_1)
\leftrightarrow
(A_2,k_2).
```

Purpose:

- closest extension of the current decay example
- tests both higher dimension and multimodality

## Implementation Plan

Create:

```text
scripts/npe_flow_stress_tests.py
```

The script should support:

- case selection
- shared exact log posterior used by MCMC and HMC
- plain random-walk Metropolis MCMC
- HMC with autodiff gradients
- MCMC/HMC diagnostics
- local-prior NPE training
- optional linear target preconditioning
- conditional spline-flow density estimator
- optional grid/reference posterior generation
- overlaid MCMC/HMC/NPE corner plots for low-dimensional cases
- posterior predictive plots when observations are curves
- posterior predictive overlays from all three samplers
- JSON summaries

Create results under:

```text
runs/02_stress_sign/01_npe_flow/<run-name>/results/
runs/03_stress_banana/01_npe_flow/<run-name>/results/
runs/04_stress_label_switch/01_npe_flow/<run-name>/results/
runs/05_stress_linear6/01_npe_flow/<run-name>/results/
runs/06_two_exponential/01_npe_flow/<run-name>/results/
```

Each model folder keeps a consistent parameter vector in its corner plots.

Create final report:

```text
notes/npe-flow-model-results.md
```

## Success Criteria

The investigation is successful if it establishes, with artifacts, which cases the flow-NPE recipe handles with minimal adjustment and which require structural changes.

For each case, report:

- whether MCMC converged
- whether HMC converged
- whether MCMC and HMC agree with each other
- whether NPE agrees with both MCMC and HMC
- best achieved metric
- whether mode mass is correct
- what adjustments were needed
- whether the posterior comes directly from a normalizing flow
- next fix if the case remains difficult
