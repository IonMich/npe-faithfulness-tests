# Normalizing Flows For NPE

## Flow Density Model

A normalizing flow starts with a simple base variable:

```text
u ~ N(0, I)
```

and applies an invertible map conditioned on the observation:

```text
theta = f_phi(u; x)
```

The conditional density is available by change of variables:

```text
log q_phi(theta | x)
  = log p_U(f_phi^{-1}(theta; x))
    + log |det d f_phi^{-1}(theta; x) / d theta|
```

This gives NPE two important operations:

- `sample(x_o)`: draw `u`, transform to `theta`;
- `log_prob(theta, x_o)`: evaluate posterior density under the approximation.

These operations are why flows became the default density estimator for many
NPE implementations.

## Main Flow Families

Coupling flows transform part of the vector while conditioning on the remaining
part. They are often fast to sample and evaluate but may need more layers to
couple all dimensions.

Autoregressive flows model each dimension conditionally on previous dimensions.
Masked Autoregressive Flow (MAF) is strong for density estimation and can train
efficiently with parallel density evaluation, but sampling can be slower because
dimensions are generated sequentially.

Neural Spline Flows (NSF) replace affine elementwise transforms with monotone
rational-quadratic splines. In practice, NSF is often more expressive than MAF
or affine coupling at the same number of transforms, but it can train more
slowly and overfit more easily at small simulation budgets.

Flow Matching Posterior Estimation (FMPE) uses continuous normalizing flows
instead of a fixed stack of discrete transforms. It is a useful newer direction
for higher-dimensional posteriors, but discrete MAF/NSF remain the common
baseline in `sbi` and in this repo.

## Parameterization Is Part Of The Model

Flows are easiest to train in unconstrained Euclidean coordinates. Do not ask a
flow to learn hard constraints if a deterministic transform can remove them.

Useful transforms:

- positive parameters: `z = log(theta)`;
- bounded parameters: logit transform to the real line;
- ordered positive rates: use log gaps, e.g. `log(k1)`, `log(k2 - k1)`;
- scale parameters: log scale rather than raw scale;
- label switching: sort or use invariant coordinates for diagnostics, then
  restore labels only when the scientific quantity needs them.

Store every transform with the model artifact. A posterior sample without its
parameterization metadata is easy to misread.

## Conditioning Context

The conditioning variable `x` may be raw simulator output, summary statistics,
or an embedding-network output.

Raw context is attractive because it avoids hand-crafted information loss, but
it increases sample complexity. Summary context can make training easier, but
it can discard target-relevant information. A learned embedding can be better
than both, but then the embedding itself becomes a trained component that needs
diagnostics.

Implementation guidance:

- always standardize context using training-set mean and scale;
- standardize or whiten the target parameterization;
- use domain summaries for pilot/local selection even if the final estimator
  uses raw data or an embedding;
- check whether `x_o` lies inside the simulated context cloud before trusting
  posterior samples;
- save context transforms and observed standardized context in artifacts.

## Architecture Defaults

For low-dimensional targets, start small and increase only when diagnostics
justify it.

Practical starting points:

- `sbi`: `posterior_nn(model="nsf", hidden_features=50-128,
  num_transforms=5, num_bins=8-10)`;
- custom Zuko-style NSF: 5-8 transforms, hidden widths 64-128, 8-12 bins;
- train MDN or Gaussian baselines as sanity checks;
- use ensembles of 3-5 seeds for important runs.

Large flows can hide problems by improving validation likelihood while leaving
local calibration wrong. If extra capacity changes posterior means or tail mass
substantially, run calibration checks rather than picking the best-looking
corner plot.

## Training Details That Matter

Use maximum likelihood on `(theta, x)` pairs, but make the effective training
distribution explicit.

For local or proposal-focused training:

- record the proposal density `r(theta)`;
- preserve a prior-mixture component or other support guarantee;
- correct weights with `p(theta) / r(theta)` when the method requires it;
- track weight effective sample size;
- cap extreme log weights only as a numerical stabilization, and report it.

For optimization:

- use deterministic train/validation splits and fixed seeds;
- monitor train and validation NLL, but do not treat NLL as a posterior
  accuracy metric;
- use early stopping, gradient clipping, and a conservative learning rate;
- compare several seeds before concluding that an architecture works;
- prefer `float64` for reference samplers and stable simulator summaries, but
  `float32` is usually acceptable for neural training on GPU/MPS.

## Sampling And Prior Support

Direct flow samples can land outside hard prior support if the prior support was
bounded and not transformed into unconstrained coordinates. Options:

- transform bounded parameters before training;
- reject or clip samples only if this is reported and the rejected fraction is
  negligible;
- use `sbi` posterior/prior transforms where appropriate;
- inspect posterior mass near boundaries.

If an analytic likelihood is available, a trained NPE posterior can be used as a
proposal for importance sampling or SIR correction. This is no longer pure
likelihood-free NPE, but it can be a good hybrid when exact likelihood
evaluation is cheap enough.

## Repo Notes

The custom flow code in `scripts/npe_flow_stress_tests.py` uses a conditional
NSF, local simulation selection, kernel/proposal weights, target
standardization, optional full target whitening, and optional linear residual
adjustment. These are reasonable techniques, but the experiment history shows
that each transform must be validated per model:

- linear adjustment helps banana-like local geometry;
- disabling it helps sign symmetry and residual-centered two-exponential runs;
- ordered and invariant parameterizations are essential for label-switching
  tests;
- the two-exponential case still fails the strict target, suggesting residual
  conditional-density bias rather than a simple flow-capacity issue.

## Sources

- Dinh, Sohl-Dickstein, and Bengio, "Density estimation using Real NVP":
  https://arxiv.org/abs/1605.08803
- Papamakarios, Pavlakou, and Murray, "Masked Autoregressive Flow for Density
  Estimation": https://arxiv.org/abs/1705.07057
- Durkan et al., "Neural Spline Flows": https://arxiv.org/abs/1906.04032
- Papamakarios et al., "Normalizing Flows for Probabilistic Modeling and
  Inference": https://jmlr.org/papers/v22/19-1028.html
- Greenberg, Nonnenmacher, and Macke, "Automatic Posterior Transformation for
  Likelihood-Free Inference": https://arxiv.org/abs/1905.07488
- Radev et al., "BayesFlow: Learning complex stochastic models with invertible
  neural networks": https://arxiv.org/abs/2003.06281
- Wildberger et al., "Flow Matching for Scalable Simulation-Based Inference":
  https://arxiv.org/abs/2305.17161
- sbi `posterior_nn` API:
  https://sbi.readthedocs.io/en/latest/reference/_autosummary/sbi.neural_nets.posterior_nn.html
- sbi neural-net selection guide:
  https://sbi.readthedocs.io/en/latest/how_to_guide/03_choose_neural_net.html
