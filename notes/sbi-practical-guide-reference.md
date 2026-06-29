# Simulation-Based Inference: A Practical Guide

Date added: 2026-06-22

## Source

- Paper: Michael Deistler, Jan Boelts, Peter Steinbach, Guy Moss, Thomas Moreau,
  Manuel Gloeckler, Pedro L. C. Rodrigues, Julia Linhart, Janne K.
  Lappalainen, Benjamin Kurt Miller, Pedro J. Goncalves, Jan-Matthis
  Lueckmann, Cornelius Schroder, Jakob H. Macke. *Simulation-Based Inference:
  A Practical Guide*. arXiv:2508.12939, 2025.
- arXiv: <https://arxiv.org/abs/2508.12939>
- HTML: <https://arxiv.org/html/2508.12939v1>
- Companion code: <https://github.com/sbi-dev/sbi-practical-guide>
- SBI applications explorer: <https://sbi-applications-explorer.streamlit.app/>

BibTeX:

```bibtex
@misc{DeistlerBoelts_simulationbased_2025,
  title = {Simulation-Based Inference: A Practical Guide},
  author = {Deistler, Michael and Boelts, Jan and Steinbach, Peter and Moss, Guy
    and Moreau, Thomas and Gloeckler, Manuel and Rodrigues, Pedro L. C. and Linhart, Julia
    and Lappalainen, Janne K. and Miller, Benjamin Kurt and Goncalves, Pedro J.
    and Lueckmann, Jan-Matthis and Schroder, Cornelius and Macke, Jakob H.},
  year = {2025},
  doi = {10.48550/arXiv.2508.12939},
  archivePrefix = {arXiv}
}
```

## Why this is useful here

This paper is a good operating manual for the work in this repository. It is
less about proposing one new algorithm and more about the full SBI workflow:
define simulator and prior, choose data representation and inference method,
train on simulated pairs, validate the posterior, then analyze the posterior
only after diagnostics are credible.

The current repo is already following much of that shape:

- simulator and prior: exponential decay model with `theta = (A, k, sigma)`
- NPE objective: `q_phi(z | x)` with `z = log(theta)`
- posterior families: diagonal Gaussian, full Gaussian, MDN, affine flow, and
  `sbi` SNPE experiments
- reference checks: grid, MCMC, HMC, Wasserstein distance, quantile error, and
  posterior predictive overlays

## Practical takeaways for this repo

### 1. Keep NPE as the default baseline, but validate hard

The guide recommends starting with NPE unless the problem clearly favors another
method. That matches this repo: NPE is fast at inference time, direct to sample
from, and avoids extra MCMC/VI tuning after training.

The repo results also show the important caveat: a plausible-looking NPE
posterior can still be materially wrong. For this project, every NPE result
should continue to be judged against:

- an independent reference posterior when available
- posterior predictive overlays
- held-out simulation validation loss
- quantile and Wasserstein errors
- calibration diagnostics where feasible

### 2. Add calibration diagnostics, not only pointwise distance

The guide emphasizes diagnostics after training:

- misspecification checks: does the observed `x_o` look plausible under the
  simulator/prior predictive distribution?
- posterior predictive checks: do simulations from posterior samples reproduce
  the observed curve?
- global coverage checks: expected coverage, SBC, TARP
- local checks for a specific observation when the global result is not enough

Concrete project idea:

```text
scripts/check_npe_calibration.py
```

Use calibration pairs `(theta_i, x_i)` from the prior, infer posteriors
`q_phi(theta | x_i)`, and compute SBC or expected coverage for the trained
Stage 1 and scaled Stage 1 models.

### 3. Treat sequential NPE as targeted inference, not amortized inference

The guide frames sequential methods as best suited when simulations are
expensive and the goal is one or a few observations. That matches the repo's
SNPE/local-region experiments. The key warnings are directly relevant:

- establish a non-sequential baseline first
- make the first round large enough to avoid focusing on the wrong region
- track posterior shape across rounds, not only final loss
- remember that sequential training gives up broad amortization

Concrete project idea:

```text
notes/snpe-sequential-results.md
```

Extend the round-by-round report with a convergence table: posterior median,
posterior covariance, Wasserstein-to-reference, and predictive error for each
round.

### 4. Try ensembles when individual NPE models are overconfident

The guide's high-dimensional neuroscience example uses an ensemble of NPE
models and reports that the ensemble improves calibration relative to individual
models. This is worth testing here even in 3D, because the repo already saw
faithfulness gaps from single trained estimators.

Concrete project idea:

```text
scripts/npe_ensemble_decay.py
```

Train 3-5 seeds of the same NPE family, combine posterior samples across seeds,
and compare:

- individual vs ensemble posterior summaries
- individual vs ensemble Wasserstein distance
- individual vs ensemble coverage/SBC
- individual vs ensemble posterior predictive overlays

### 5. Compare method families when the target matters

The guide explicitly recommends using multiple methods for important
applications. Agreement across NPE, NLE, NRE, MCMC/HMC, or grid references is
evidence that the posterior is not an artifact of one inference objective or one
network family.

Concrete project idea:

```text
scripts/snle_decay.py
scripts/snre_decay.py
```

Use `sbi` NLE/NRE as cross-checks on the decay model. If NLE/NRE agree with the
reference while NPE does not, the failure mode is likely direct posterior
regression or density-estimator capacity/training. If all neural SBI methods
miss the target, focus on simulator representation, prior scale, summaries, or
diagnostics.

### 6. Use the companion repo as examples, not as vendored code

The companion repo is useful as a source of patterns:

- reproducible figure-specific experiment directories
- end-to-end examples using `sbi`
- diagnostics and plotting workflows
- `uv`-managed environment

Do not copy it into this repo by default. Pull individual implementation ideas
only when they directly support a local experiment.

## Most relevant sections to reread

- Section 3: the practical SBI workflow
- Section 3.4: posterior diagnostics
- Appendix A1: choosing between NPE, NLE, and NRE
- Appendix A2: sequential methods for targeted inference
- Appendix A5: diagnostic tools

## Local next steps inspired by the guide

1. Add a calibration script for SBC or expected coverage on trained NPE models.
2. Add a prior predictive/OOD check for `x_o` before NPE training.
3. Add ensemble training for the best current NPE family.
4. Try `sbi` NLE or NRE as an independent neural SBI cross-check.
5. Make each experiment note report simulator budget, validation set size,
   posterior predictive result, and calibration result.
