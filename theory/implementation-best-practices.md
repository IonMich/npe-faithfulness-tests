# Implementation Best Practices For NPE

These notes are written for this repository's NPE experiments, but the checklist
is general for flow-based simulation-based inference.

## 1. Establish The Generative Contract

Before training:

- define `theta`, transformed `z`, prior, simulator noise, and observation
  shape in one place;
- run prior predictive simulations and check whether `x_o` is plausible;
- decide whether inference targets raw parameters, log parameters, ordered
  coordinates, or invariant diagnostics;
- record all constants: true parameters for synthetic tests, simulator seed,
  time grid, noise model, and prior bounds.

If `x_o` is outside the prior predictive distribution, posterior diagnostics are
not enough; the simulator/prior/data representation is misspecified.

## 2. Use Exact References When Available

For toy or tractable models, keep exact references in the loop:

- grid posterior for very low-dimensional problems;
- independent random-walk MCMC and HMC;
- chain diagnostics (`Rhat`, bulk/tail ESS, trace plots);
- posterior predictive overlays from each reference.

Use NPE as a fast surrogate only after it agrees with these references on the
scientific parameterization. This repo's normalized Wasserstein target is a good
example of a hard acceptance criterion.

## 3. Choose Parameterization Before Architecture

Most NPE problems that look like "flow capacity" problems are at least partly
parameterization problems.

Recommended defaults:

- positive parameters in log space;
- ordered parameters through unconstrained base plus positive gaps;
- scale/noise parameters in log space;
- label-switching models with ordered or invariant diagnostics;
- residual or centered coordinates when the posterior is mostly a local
  perturbation around a summary-derived estimate.

Avoid switching parameterizations without updating diagnostics. A diagnostic
that is correct in raw labels can be wrong for sorted labels, and vice versa.

## 4. Design The Observation Context Deliberately

Try three context levels:

1. domain summaries that are cheap and identifiable;
2. raw data or a lightly downsampled version;
3. learned embeddings for high-dimensional observations.

Do not assume raw data is better. Raw context increases the conditional learning
problem and may require more simulations. Do not assume summaries are safe
either; they can hide weak-identifiability directions.

For this repo:

- exponential decay works with local raw or summary contexts, but strict
  MC-level accuracy is still hard;
- label switching needs summaries/diagnostics that respect exchangeability;
- ordered two-exponential runs improved with profile/residual summaries but did
  not pass the strict target.

## 5. Treat Proposals As Statistical Objects

Sequential/local training must preserve target support.

Checklist:

- keep a large enough first round from the prior;
- when using a Gaussian proposal from reference or previous posterior samples,
  inflate covariance and mix in some prior mass;
- record proposal mean, covariance, inflation, prior-mixture weight, and source;
- compute and store `log p(theta) - log r(theta)` for accepted simulations;
- monitor correction ESS and extreme weights;
- never trust a run where the proposal excludes plausible posterior regions.

If local hard selection is used, report pilot size, local quantile, acceptance
rate, context-distance distribution, and candidate cap. These are part of the
statistical method, not incidental metadata.

## 6. Train Flows Conservatively

Use a simple reproducible training loop:

- fixed train/validation split;
- standardized context and target;
- AdamW or Adam with modest learning rate;
- gradient clipping;
- early stopping by validation NLL;
- saved training history;
- saved model and transform state.

Then test robustness:

- repeat 3-5 seeds for the same configuration;
- compare Gaussian/MDN/flow baselines;
- run an oracle density fit on exact posterior samples if available;
- ensemble posterior samples when individual estimators are overconfident.

Validation NLL is useful for catching obvious optimization failures. It is not a
posterior faithfulness metric.

## 7. Diagnose Posterior Faithfulness

Use multiple diagnostics because each sees a different failure mode.

Minimum checks:

- posterior predictive check: simulate from posterior samples and compare to
  `x_o`;
- pairwise agreement with exact references if available;
- marginal and joint metrics, not just one-dimensional means;
- corner plots and trace plots for qualitative inspection;
- prior predictive or misspecification check for `x_o`.

Calibration checks:

- expected coverage for joint over/underconfidence;
- SBC for marginal skew, narrowness, or broadness;
- TARP as a sample-based posterior accuracy check;
- L-C2ST for local validation around a specific observation.

Run calibration on held-out simulated observations, not the training set. For a
single important observation, add a local diagnostic rather than relying only on
average prior-predictive calibration.

## 8. Interpret Failures Productively

When NPE fails against a reference, isolate the source:

- if posterior samples from a directly fitted density also fail, the family or
  parameterization is insufficient;
- if oracle density fitting passes but NPE fails, the problem is conditional
  learning from simulations;
- if posterior predictive checks fail, suspect inference bias or
  misspecification;
- if expected coverage is below nominal, suspect overconfidence;
- if SBC is skewed, suspect biased posterior location;
- if MCMC/HMC disagree, fix the reference before judging NPE.

This repo's exponential-decay investigation points to conditional learning bias:
posterior samples are representable enough, but simulation-trained NPE does not
reach the strict MC-level target. The ordered two-exponential case appears to be
an even harder version of the same issue.

## 9. Repo-Specific Next Experiments

For the ordered two-exponential case:

- run a known-sigma variant to test whether the unknown noise dimension drives
  the failure;
- train an embedding network or lower-dimensional learned summary for the raw
  curve instead of direct high-dimensional conditioning;
- run oracle flow/MDN fits on exact posterior samples to separate family
  capacity from conditional learning;
- add expected coverage and SBC over a panel of two-exponential observations;
- use L-C2ST for the current `x_o` if simulation budget permits;
- keep MCMC/HMC as the reported reference until NPE passes local diagnostics.

For all new runs:

- write a README with command, seed, simulator budget, proposal details,
  diagnostics, pass/fail target, and artifact paths;
- prefer `uv run scripts/...py` for script execution, matching repo
  instructions;
- avoid adding dependencies by editing `pyproject.toml` manually; use
  `uv add <dependency>`.

## 10. "Done" Criteria For A Trustworthy NPE Run

A run is trustworthy enough to present as posterior inference when:

- the observed data pass prior/posterior predictive checks;
- references agree, or calibration diagnostics replace unavailable references;
- NPE passes the declared numeric target on the parameterization that matters;
- calibration does not show material overconfidence or skew;
- posterior conclusions are stable across seeds or ensembles;
- all transforms, proposals, seeds, and training curves are saved.

If any item is missing, label the result as approximate or diagnostic rather
than faithful.

## Sources

- Deistler et al., "Simulation-Based Inference: A Practical Guide":
  https://arxiv.org/abs/2508.12939
- Lueckmann et al., "Benchmarking Simulation-Based Inference":
  https://arxiv.org/abs/2101.04653
- Hermans et al., "A Trust Crisis In Simulation-Based Inference? Your Posterior
  Approximations Can Be Unfaithful": https://arxiv.org/abs/2110.06581
- Deistler, Goncalves, and Macke, "Truncated proposals for scalable and
  hassle-free simulation-based inference": https://arxiv.org/abs/2210.04815
- Talts et al., "Validating Bayesian Inference Algorithms with
  Simulation-Based Calibration": https://arxiv.org/abs/1804.06788
- Lemos et al., "Sampling-Based Accuracy Testing of Posterior Estimators for
  General Inference": https://arxiv.org/abs/2302.03026
- Linhart, Gramfort, and Rodrigues, "L-C2ST: Local Diagnostics for Posterior
  Approximations in Simulation-Based Inference": https://arxiv.org/abs/2306.03580
- Wang et al., "Preconditioned Neural Posterior Estimation for Likelihood-free
  Inference": https://arxiv.org/abs/2404.13557
- sbi diagnostics guide:
  https://sbi.readthedocs.io/en/latest/how_to_guide/14_choose_diagnostic_tool.html
- sbi expected coverage:
  https://sbi.readthedocs.io/en/stable/how_to_guide/15_expected_coverage.html
- sbi SBC:
  https://sbi.readthedocs.io/en/stable/how_to_guide/16_sbc.html
- sbi TARP:
  https://sbi.readthedocs.io/en/latest/how_to_guide/17_tarp.html
- sbi posterior predictive checks:
  https://sbi.readthedocs.io/en/latest/advanced_tutorials/10_diagnostics_posterior_predictive_checks.html
