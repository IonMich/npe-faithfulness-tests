# ABC Faithfulness Repair Plan

Goal: test whether likelihood-free ABC-style correction can make the decay-model posterior as faithful as the MCMC/HMC/grid reference, using the strict target

```text
mean normalized Wasserstein <= 0.034
```

The working diagnosis from the NPE/SNPE investigation is:

```text
pure NPE/SNPE is useful but not yet MC-faithful at finite budget;
the dominant miss is conditional density-estimation bias near x0.
```

This plan treats neural posteriors as proposal mechanisms, not final trusted posteriors.

## Reference

Use the existing reference stack:

- observed decay dataset from `runs/01_exponential_decay/01_mcmc_hmc_reference/00_root_decay_sampler_results/results/mcmc_decay_samples.npz`
- grid posterior built with the same utility as `scripts/compare_decay_samplers.py`
- MCMC/HMC faithfulness target from `scripts/check_faithfulness_target.py`

Primary metric:

```text
mean normalized 1D Wasserstein distance to the grid posterior
```

Pass criterion:

```text
<= 0.034
```

## Methods to Test

### 1. Prior ABC

Sample from the prior:

```math
z_i \sim p(z), \qquad \theta_i = \exp(z_i), \qquad x_i \sim p(x \mid \theta_i).
```

Compute a simulator-only discrepancy:

```math
d_i = \rho(s(x_i), s(x_0)).
```

Keep or weight particles using threshold/kernel ABC:

```math
w_i \propto K_\epsilon(d_i).
```

This is the honest baseline but should be simulation-inefficient.

### 2. SNPE-Proposal ABC

Fit a transparent proposal in log-parameter space from the best existing NPE/SNPE posterior samples:

```math
r(z) = \mathcal N(\mu_r, \alpha^2\Sigma_r).
```

Then simulate:

```math
z_i \sim r(z), \qquad x_i \sim p(x \mid \exp z_i).
```

Correct proposal bias:

```math
w_i \propto
\frac{p(z_i)}{r(z_i)}
K_\epsilon(d_i).
```

This directly tests whether NPE can be trusted as an accelerator while ABC controls the final target.

### 3. SMC-ABC

Use sequential ABC thresholds:

```math
\epsilon_1 > \epsilon_2 > \cdots > \epsilon_T.
```

At each stage, perturb accepted particles and reweight:

```math
w_t(z_i)
\propto
\frac{p(z_i)}
{\sum_j w_{t-1,j} \, K_t(z_i \mid z_{t-1,j})}.
```

This tests whether an adaptive likelihood-free particle method reaches the strict target with fewer wasted simulations than prior ABC.

### 4. Regression-Adjusted ABC

For accepted particles, fit a local weighted regression:

```math
z_i = a + B(s_i - s_0) + \eta_i.
```

Then adjust:

```math
z_i^\star = z_i - \widehat B(s_i - s_0).
```

This tests whether finite-threshold ABC bias is the remaining bottleneck.

## Summary Statistics

The main summary is an indirect-inference summary computed only from simulated data:

```math
s(x) =
\left(
\log \widehat A(x),
\log \widehat k(x),
\log \widehat \sigma(x)
\right).
```

The estimates come from a fast grid least-squares exponential fit over decay rate \(k\). This is not the analytic likelihood; it is a simulator-output summary designed to be close to sufficient for this toy model.

Discrepancy:

```math
\rho(s, s_0)
=
\sqrt{(s-s_0)^\top \widehat\Sigma_s^{-1}(s-s_0)}.
```

The whitening covariance \(\widehat\Sigma_s\) is estimated from prior-predictive simulations.

## Outputs

Create:

- `scripts/abc_faithfulness_decay.py`
- `runs/01_exponential_decay/05_abc_faithfulness/01_abc_faithfulness/results/abc_faithfulness_summary.json`
- `runs/01_exponential_decay/05_abc_faithfulness/01_abc_faithfulness/results/abc_faithfulness_samples.npz`
- `runs/01_exponential_decay/05_abc_faithfulness/01_abc_faithfulness/figures/abc_faithfulness_distance_curve.png`
- `runs/01_exponential_decay/05_abc_faithfulness/01_abc_faithfulness/figures/abc_faithfulness_corner_overlay.png`
- `runs/01_exponential_decay/05_abc_faithfulness/01_abc_faithfulness/figures/abc_faithfulness_predictive_overlay.png`
- `notes/abc-faithfulness-repair-results.md`

## Decision Rules

The repair is successful only if at least one likelihood-free corrected method satisfies all of:

- mean normalized Wasserstein `<= 0.034`
- adequate ESS, not just a few extreme particles
- stable posterior predictive around the observed data
- no use of the analytic likelihood in the inference weights

If no method passes, report the best distance, why it failed, and what bottleneck remains.
