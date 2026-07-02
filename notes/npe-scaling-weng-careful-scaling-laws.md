# NPE Scaling Weng Careful Scaling Laws Resource

## Question

How should we use Lilian Weng's 2026 scaling-laws discussion when designing
single-decay NPE scaling plots?

Reference: <https://lilianweng.github.io/posts/2026-06-24-scaling-laws/>

## Short Answer

Use the post as a plotting and interpretation checklist, not as a license to
mix changing recipes into one curve. Weng's discussion and the Hestness-style
illustration plot loss/error itself with an irreducible-error level in the same
coordinate system. For this repository, the clean analogue of a data-axis
scaling law is:

```text
hold the NPE recipe fixed, vary only training simulations D, and measure error
above an explicit floor.
```

For the current single-decay estimator, that means a fixed 4-member ensemble of
Flow2 residual neural spline flows, fixed context features, fixed optimizer
recipe, fixed validation cache, and fixed posterior-faithfulness panel.

## Takeaways For This Repo

- Plot error against training data on log-scaled axes.
- Separate three regimes when the data support it: small-data behavior,
  power-law-like improvement, and saturation near an irreducible or evaluation
  floor.
- Report the floor explicitly. For validation NLL, the natural floor is the
  estimated conditional entropy of the population posterior. For panel
  Wasserstein, the practical floor is the numerical/reference/sampling
  evaluation floor for the fixed panel.
- Prefer raw validation NLL for the primary measurement. Continuous-density NLL
  in `z` units can be negative, so a literal log-y plot of raw loss is not
  meaningful. A positive excess-loss diagnostic is useful:

```text
excess_nll(D) = validation_nll(D) - entropy_floor
```

but the subtraction should not drive a headline exponent unless the
entropy-floor uncertainty is smaller than the remaining excess. If the floor is
uncertain at the same scale as the rightmost points, show the uncertainty and
treat the excess-loss exponent as sensitivity analysis. For panel Wasserstein,
the analogous diagnostic is:

```text
excess_W(D) = panel_W(D) - panel_evaluation_floor.
```

- Do not call architecture search, context-feature changes, ensemble-size
  changes, or optimizer retuning a single-axis data scaling law. Those are
  frontier or recipe-improvement experiments.

## Decision For The Flow2 Ensemble Sweep

The first Weng-style update to the single-decay section should use one fixed
recipe:

```text
ensemble_size              = 4
family                     = spline_flow
flow_kind                  = nsf
flow_layers                = 2
flow_residual              = true
flow_randperm              = true
spline_bins                = 8
hidden_dim                 = 80
hidden_layers              = 2
context_features           = raw_decay_fit_summary
batch_size                 = 512
epochs                     = 15
learning_rate              = 0.00325
lr_schedule                = cosine_step
lr_warmup_steps            = 500
weight_decay               = 0.0002
batching_mode              = pre_shuffle
skip_training_validation   = true
```

The x-axis should be per-member training simulations `D`. Total simulator calls
are `4D`, but plotting per-member `D` keeps the estimator recipe fixed and makes
the axis answer the intended question.

## Related Local Resource

The Karpathy training-recipe audit remains the companion reliability checklist:

```text
notes/npe-scaling-karpathy-recipe-audit.md
```
