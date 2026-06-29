# Proposal 06: Add Two-Exponential Oracle Density Fit

## Claim

Before more NPE architecture tuning, fit NSF/MDN densities directly to exact
two-exponential posterior samples at `x_o`. This separates posterior family
capacity from conditional learning error.

## Literature Signal

- Normalizing flows are flexible density estimators, but finite training and
  parameterization still matter:
  https://jmlr.org/papers/v22/19-1028.html
- The repo's own decay investigation found that oracle posterior density fits
  were much closer to the strict target than simulation-trained NPE, implying
  conditional learning bias.
- The practical SBI workflow recommends isolating whether failures come from
  simulator/data representation, density estimator capacity, or diagnostics:
  https://arxiv.org/abs/2508.12939

## Current Code Touchpoints

- `scripts/oracle_posterior_density_fit.py` already implements this idea for
  the decay model.
- Two-exponential reference samples exist under `runs/06_two_exponential/...`
  and are loaded by `scripts/sbi_two_exp_ordered.py::load_reference`.
- Custom flow code in `scripts/npe_flow_stress_tests.py` already has the NSF
  module and sampling helpers.

## Implementation Sketch

Create:

```text
scripts/oracle_two_exp_density_fit.py
```

Inputs:

```text
--reference-samples <path>
--mcmc-burn-in
--hmc-burn-in
--families gaussian,mdn,nsf
--parameterization raw|ridgecoords|residual
```

Training setup:

- use MCMC or HMC posterior samples after burn-in as training data;
- use the other reference sampler as the evaluation target;
- condition on a constant context or no context;
- train density estimators on posterior samples directly;
- sample from the fitted density and compute normalized Wasserstein against
  MCMC/HMC.

Important variants:

- raw ordered coordinates;
- ridge coordinates;
- known-sigma or fixed-sigma variant if implemented;
- residual-centered coordinates from profile summaries.

## Acceptance Criteria

- Script writes `oracle_two_exp_density_summary.json`.
- Summary reports fit family, parameterization, train reference source, eval
  reference source, validation NLL, and Wasserstein metrics.
- At least one oracle fit is compared to the best simulation-trained NPE run.
- If oracle fit cannot beat the simulation-trained run, stop architecture
  tuning and redesign parameterization.
- If oracle fit passes or nearly passes, focus on conditional learning,
  summaries, embeddings, and calibration.

## First Target

Use the best current two-exponential reference artifact and fit NSF/MDN in the
same diagnostic parameterization used for `pairwise_agreement`. This should be
the next fastest way to understand whether the two-exponential blocker is flow
capacity or NPE conditional learning.
