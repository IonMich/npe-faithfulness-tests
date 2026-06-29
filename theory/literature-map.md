# Annotated Literature Map

This is a curated map of sources most relevant to NPE and flow-based NPE.

## Core NPE And SNPE

### Papamakarios and Murray, 2016

"Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
Density Estimation"

Link: https://arxiv.org/abs/1605.06376

Why it matters: introduced the core idea of likelihood-free posterior learning
with conditional density estimation and adaptive simulation rounds. It is the
starting point for modern SNPE.

Implementation takeaway: simulation budget should be spent where posterior mass
is likely, but adaptive proposals require correction. The proposal mechanism is
part of the estimator.

### Lueckmann et al., 2017

"Flexible statistical inference for mechanistic models of neural dynamics"

Link: https://arxiv.org/abs/1711.01861

Why it matters: developed practical SNPE ideas for mechanistic models, including
mixture-density posterior estimators, missing/invalid simulations, and learned
features for time-series observations.

Implementation takeaway: treat summaries and invalid outputs as part of the
statistical workflow, not as preprocessing afterthoughts.

### Greenberg, Nonnenmacher, and Macke, 2019

"Automatic Posterior Transformation for Likelihood-Free Inference"

Link: https://arxiv.org/abs/1905.07488

Why it matters: APT/SNPE-C addressed sequential posterior estimation with
arbitrary proposals and connected SNPE to powerful flow-based density
estimators.

Implementation takeaway: if simulations are drawn from adaptive proposals, use a
method whose objective handles those proposals. Naive retraining can learn the
wrong posterior.

## Normalizing Flow Foundations

### Dinh, Sohl-Dickstein, and Bengio, 2016

"Density estimation using Real NVP"

Link: https://arxiv.org/abs/1605.08803

Why it matters: established tractable invertible transformations with exact
log-likelihood, sampling, and latent inversion.

Implementation takeaway: invertibility and cheap Jacobian determinants are the
engineering constraints that make flows practical.

### Papamakarios, Pavlakou, and Murray, 2017

"Masked Autoregressive Flow for Density Estimation"

Link: https://arxiv.org/abs/1705.07057

Why it matters: MAF became a standard density estimator in SBI and `sbi`.

Implementation takeaway: MAF is a strong baseline for density estimation, but
sampling speed and autoregressive structure matter for direct posterior draws.

### Durkan et al., 2019

"Neural Spline Flows"

Link: https://arxiv.org/abs/1906.04032

Why it matters: rational-quadratic spline transforms improved flow flexibility
while retaining analytic inversion.

Implementation takeaway: NSF is a good default for low-to-medium dimensional
flow-based NPE when simulation budget is sufficient, but watch overfitting.

### Papamakarios et al., 2021

"Normalizing Flows for Probabilistic Modeling and Inference"

Link: https://jmlr.org/papers/v22/19-1028.html

Why it matters: broad review of flow design, expressiveness, and computational
tradeoffs.

Implementation takeaway: choose the flow family based on the operations you
need. NPE needs both density evaluation during training and sampling after
training.

## Flow-Based And Amortized SBI Systems

### Tejero-Cantero et al., 2020

"sbi: A toolkit for simulation-based inference"

Link: https://www.theoj.org/joss-papers/joss.02505/10.21105.joss.02505.pdf

Why it matters: standard Python toolkit implementing NPE/SNPE, NLE/SNLE, and
NRE/SNRE.

Implementation takeaway: use library implementations as baselines, but keep
problem-specific diagnostics and references. Off-the-shelf defaults are not
proof of posterior faithfulness.

### Radev et al., 2020

"BayesFlow: Learning complex stochastic models with invertible neural networks"

Link: https://arxiv.org/abs/2003.06281

Why it matters: emphasizes globally amortized Bayesian inference using
invertible networks plus learned summaries.

Implementation takeaway: learned summaries can be as important as the posterior
flow for high-dimensional observations.

### Dax et al., 2021/2022

"Group equivariant neural posterior estimation"

Link: https://arxiv.org/abs/2111.13139

Why it matters: shows how known symmetries/equivariances can be exploited in
posterior estimation.

Implementation takeaway: when the simulator has symmetries, encode them through
parameterization, standardization, or architecture. Do not force a generic flow
to rediscover exact symmetry from finite simulations.

### Wildberger et al., 2023

"Flow Matching for Scalable Simulation-Based Inference"

Link: https://arxiv.org/abs/2305.17161

Why it matters: extends NPE-style posterior estimation to continuous
normalizing flows through flow matching, motivated partly by scaling limits of
discrete flows.

Implementation takeaway: FMPE is worth considering when discrete flows struggle
in higher dimensions, but it is a different sampler/training stack and should be
benchmarked against NSF/MAF baselines.

## Benchmarks, Diagnostics, And Reliability

### Cranmer, Brehmer, and Louppe, 2020

"The frontier of simulation-based inference"

Link: https://arxiv.org/abs/1911.01429

Why it matters: review of the SBI landscape and the motivation for neural SBI.

Implementation takeaway: NPE is one tool in a workflow. Simulator validity and
diagnostics remain central.

### Lueckmann et al., 2021

"Benchmarking Simulation-Based Inference"

Link: https://arxiv.org/abs/2101.04653

Why it matters: systematic comparison across benchmark tasks; no single method
wins everywhere.

Implementation takeaway: metric choice matters. Report several metrics and use
benchmarks or references when developing a method.

### Hermans et al., 2021/2022

"A Trust Crisis In Simulation-Based Inference? Your Posterior Approximations Can
Be Unfaithful"

Link: https://arxiv.org/abs/2110.06581

Why it matters: documents overconfident and unfaithful SBI posteriors across
methods, including NPE/SNPE.

Implementation takeaway: scientific use needs calibration and coverage checks.
Visual plausibility is not enough.

### Deistler, Goncalves, and Macke, 2022

"Truncated proposals for scalable and hassle-free simulation-based inference"

Link: https://arxiv.org/abs/2210.04815

Why it matters: TSNPE improves sequential NPE robustness with truncated
proposals and scalable coverage tests.

Implementation takeaway: proposal design should reduce wasted simulations while
preserving posterior support and enabling diagnostics.

### Talts et al., 2018

"Validating Bayesian Inference Algorithms with Simulation-Based Calibration"

Link: https://arxiv.org/abs/1804.06788

Why it matters: SBC is a general posterior-sampler validation method.

Implementation takeaway: use simulated parameters and observations to check rank
statistics. SBC detects marginal calibration problems.

### Lemos et al., 2023

"Sampling-Based Accuracy Testing of Posterior Estimators for General Inference"

Link: https://arxiv.org/abs/2302.03026

Why it matters: introduces TARP, a sample-based posterior accuracy test for
generative posterior estimators.

Implementation takeaway: TARP is useful when only posterior samples are
available, but reference-point choices matter.

### Linhart, Gramfort, and Rodrigues, 2023

"L-C2ST: Local Diagnostics for Posterior Approximations in Simulation-Based
Inference"

Link: https://arxiv.org/abs/2306.03580

Why it matters: local validation method for a fixed observation, including a
flow-specialized variant.

Implementation takeaway: use local diagnostics for the actual observation when
average prior-predictive calibration is too weak.

### Deistler et al., 2025

"Simulation-Based Inference: A Practical Guide"

Link: https://arxiv.org/abs/2508.12939

Why it matters: practical end-to-end workflow for applying SBI, including
diagnostics and method choice.

Implementation takeaway: treat SBI as a workflow: prior/simulator checks,
method choice, training, posterior analysis, diagnostics, and iteration.

## Recent Directions Relevant To This Repo

### Wiqvist, Frellsen, and Picchini, 2021

"Sequential Neural Posterior and Likelihood Approximation"

Link: https://arxiv.org/abs/2102.06522

Why it matters: combines normalizing-flow posterior and likelihood
approximations in a sequential likelihood-free algorithm.

Implementation takeaway: hybrid posterior/likelihood learning can avoid some
proposal-correction issues, but doubles the approximation surface.

### Wang et al., 2024

"Preconditioned Neural Posterior Estimation for Likelihood-free Inference"

Link: https://arxiv.org/abs/2404.13557

Why it matters: uses ABC-style preconditioning to focus NPE training on
parameter regions that can generate data close to `x_o`.

Implementation takeaway: local preconditioning is a principled version of what
many hand-built local NPE pipelines attempt. It is relevant to this repo's hard
local two-exponential and decay cases.

## Useful Documentation

- sbi overview and implemented algorithms:
  https://sbi.readthedocs.io/en/latest/
- choosing an inference method:
  https://sbi.readthedocs.io/en/latest/how_to_guide/06_choosing_inference_method.html
- choosing neural nets:
  https://sbi.readthedocs.io/en/latest/how_to_guide/03_choose_neural_net.html
- posterior density builders:
  https://sbi.readthedocs.io/en/latest/reference/_autosummary/sbi.neural_nets.posterior_nn.html
- sequential inference:
  https://sbi.readthedocs.io/en/latest/how_to_guide/02_multiround_inference.html
- diagnostics overview:
  https://sbi.readthedocs.io/en/latest/how_to_guide/14_choose_diagnostic_tool.html
