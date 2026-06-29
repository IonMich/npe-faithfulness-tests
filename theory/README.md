# NPE Theory Notes

This folder collects literature notes and implementation guidance for neural
posterior estimation (NPE), with emphasis on conditional normalizing flows.

Read in this order:

1. [npe-foundations.md](npe-foundations.md): what NPE optimizes, why it learns
   a posterior, and how amortized and sequential variants differ.
2. [posterior-amortization.md](posterior-amortization.md): what counts as
   amortized posterior inference in this repo and how to measure it.
3. [normalizing-flows-for-npe.md](normalizing-flows-for-npe.md): how flows are
   used as conditional posterior density estimators and what design choices
   matter.
4. [implementation-best-practices.md](implementation-best-practices.md):
   practical checklist for this repository and similar simulation-based
   inference projects.
5. [literature-map.md](literature-map.md): annotated reading list with links.

## Short Version

NPE trains a conditional density estimator `q_phi(theta | x)` on simulated pairs
`theta ~ p(theta), x ~ p(x | theta)`. With enough capacity and data, maximum
likelihood training minimizes an average `KL(p(theta | x) || q_phi(theta | x))`,
so the optimum is the Bayesian posterior for every observation in the
simulation distribution.

Normalizing flows are common NPE posterior estimators because they provide both
direct samples and tractable log densities. In practice, this is only useful if
the flow is trained on a well-parameterized target, with enough simulations near
the observed data, and with diagnostics that test calibration and local
posterior faithfulness.

The central implementation lesson is that validation loss and plausible corner
plots are not sufficient. Flow-based NPE can be visually close while still
overconfident or biased enough to fail strict Monte Carlo-level agreement. This
repo already demonstrates that pattern: several stress cases pass against MCMC
and HMC, while the ordered two-exponential case remains a hard local
density-estimation problem.

## Repo-Specific Interpretation

The local custom NPE flow stack in `scripts/npe_flow_stress_tests.py` is aligned
with current SBI practice:

- transform constrained parameters into unconstrained coordinates;
- build problem-specific summaries or residual targets;
- standardize both context and posterior target;
- use local/proposal-focused simulations for a target observation;
- compare against independent MCMC/HMC references and posterior predictive
  overlays.

The open gap is diagnostic maturity. For any result meant to be trusted as more
than a fast approximation, add expected coverage, SBC, TARP or L-C2ST, and prior
or posterior predictive checks. For the two-exponential case, keep MCMC/HMC as
the reference until the NPE approximation passes local diagnostics, not just
training loss.
