# NPE Stage 1 plan: posterior family comparison on the decay model

Date: 2026-06-22

## Goal

Implement and compare the first set of neural posterior estimators for the same
noisy exponential decay model used by the MCMC/HMC experiments:

```text
y_i = A exp(-k t_i) + Normal(0, sigma)
theta = (A, k, sigma)
z = log(theta)
```

The NPE models learn a conditional density:

```text
q_phi(z | x)
```

where `x` is the fixed-length vector of observed `y` values on the known time
grid, and `z = (log A, log k, log sigma)` is the unconstrained parameter vector.

## Fixed ingredients

- Simulator: same exponential decay simulator and priors as the MCMC/HMC scripts.
- Observation representation: the 40 observed `y` values on the fixed time grid.
- Encoder family: MLP.
- Training objective:

```text
min_phi - mean_i log q_phi(z_i | x_i)
```

with simulation pairs:

```text
z_i ~ p(z)
theta_i = exp(z_i)
x_i ~ p(x | theta_i)
```

## Posterior families in Stage 1

### 1. Conditional diagonal Gaussian

```text
q_phi(z | x) = Normal(mu_phi(x), diag(std_phi(x)^2))
```

This is the smallest baseline. It can represent posterior location and marginal
scale but cannot represent posterior correlations.

### 2. Conditional full-covariance Gaussian

```text
q_phi(z | x) = Normal(mu_phi(x), L_phi(x) L_phi(x)^T)
```

This tests whether one learned affine Gaussian posterior is sufficient for the
decay model.

### 3. Conditional mixture density network

```text
q_phi(z | x) = sum_m pi_phi,m(x) Normal(mu_phi,m(x), L_phi,m(x)L_phi,m(x)^T)
```

This tests whether a mixture of affine Gaussian pieces improves marginal tails or
captures non-Gaussian structure.

### 4. Conditional affine coupling flow

Start from:

```text
u ~ Normal(0, I)
```

Then transform through alternating conditional affine coupling layers:

```text
z = f_K o ... o f_1(u; x)
```

Each layer has neural scale/shift networks conditioned on both part of `z` and an
MLP context encoding of `x`.

## Evaluation

Each trained model is evaluated on the same observed dataset `x_o` used by the
MCMC/HMC experiments.

Metrics:

- training time
- validation negative log likelihood on held-out simulations
- posterior summary for `A`, `k`, `sigma`
- Wasserstein distance to the existing grid reference posterior
- quantile error against the grid reference
- posterior predictive overlay against observed data

## Reference posterior

For this three-dimensional example, use the numerical grid reference already
implemented in `scripts/compare_decay_samplers.py`. This is not scalable to high
dimensions, but it is a useful independent target for Stage 1.

## Expected lessons

- Diagonal Gaussian should be fastest but should miss the curved `A`-`k`
  posterior correlation.
- Full-covariance Gaussian may already work well because the posterior is
  unimodal and close to elliptical in log-parameter space.
- MDN and affine coupling flow should be more flexible, but may need more
  simulations and training time.
- The flow should be most useful once posterior shape is curved, skewed, or
  multimodal; this decay example is a controlled first test, not the hardest case.
