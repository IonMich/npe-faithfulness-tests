# Single-Decay Bayes Entropy High-Precision Estimate

## Question

Can we tighten the single-decay Bayes entropy floor used in scaling-law plots to
roughly the same uncertainty as the 1M-example validation NLL estimates?

## Short Answer

Yes. A posterior-centered adaptive Gauss-Hermite estimate over the full 1M
validation cache gives:

```text
H(theta | X) estimate = -3.6386545787958
Monte Carlo SE        =  0.0025331503100
Reported uncertainty  = +/-0.0026
```

The previous working value, `-3.64122 +/- 0.008`, was consistent but materially
less precise.

## Method

For validation pairs `(z_i, x_i)` sampled from `p(z)p(x|z)`, the target is:

```text
E[-log p(z_i | x_i)]
  = E[-log p(x_i | z_i) - log p(z_i) + log p(x_i)].
```

The expensive term is the evidence integral:

```text
p(x_i) = integral p(x_i | z) p(z) dz.
```

The high-precision estimator evaluates that integral by:

1. optimizing the log joint `log p(x_i|z) + log p(z)` for each observation,
2. estimating the local negative Hessian at the posterior mode,
3. using the inverse Hessian as a Gaussian proposal,
4. applying tensor-product Gauss-Hermite quadrature around that proposal.

The true `z_i` is used only as the optimizer initialization. The evidence
integral and posterior density evaluation are still computed from the model
density, not from a shortcut at the true parameter.

## Full Run

```text
uv run scripts/estimate_decay_bayes_entropy_adaptive.py \
  --name adaptive_gh13_full1m \
  --gh-order 13 \
  --flush-every 10000 \
  --report-every 50000 \
  --resume \
  --output-dir runs/01_exponential_decay/15_broad_scaling/202_bayes_entropy_high_precision
```

Result:

```text
examples          = 1,000,000
GH order          = 13
GH points/example = 2,197
mean              = -3.6386545787958
std               = 2.5331503100295
standard error    = 0.0025331503100
median            = -3.6663351104491
q01 / q99         = -9.3997898340397 / 2.4518365518768
```

The estimator flushed every 10k examples and was resumed once from the saved
580k-example checkpoint.

Compact summary JSON:

```text
runs/00_shared_assets/readme_scaling/decay_bayes_entropy_adaptive_gh13_full1m.json
```

## Numerical Checks

On the first 1k validation examples:

| Check | Mean | SE |
| --- | ---: | ---: |
| GH order 9 | -3.597019 | 0.078065 |
| GH order 13 | -3.596907 | 0.078069 |
| GH13, proposal scale 0.75 | -3.597238 | 0.078060 |
| GH13, proposal scale 1.25 | -3.596885 | 0.078070 |

The observed quadrature/proposal sensitivity on this probe is about `0.00035`,
well below the full-cache Monte Carlo SE.

`58,065` of the `1,000,000` BFGS optimizer calls reported `success=false`.
These are status/precision flags in sharp posterior cases rather than a
different integral result: for 100 sampled flagged cases, rerunning from the
prior mean produced the same NLL as true-`z` initialization to numerical
precision, with median absolute difference about `1.8e-14`.

## Impact On Scaling Plot

Using the updated floor:

```text
rightmost excess NLL = -3.6306901640625 - (-3.6386545787958)
                     = 0.0079644147333
```

The free-asymptote raw-loss fit remains:

```text
L_free = -3.6361048428587
alpha  = 0.8234184868241
```

The fitted raw asymptote is `0.00255` above the numerical entropy estimate,
which is comparable to the new `+/-0.0026` uncertainty. Therefore the residual
floor still should not be treated as resolved.
