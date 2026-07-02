# Flow2 Ensemble Data Scaling Results

## Question

Can the current 4-member Flow2 residual NSF ensemble reproduce a Lilian
Weng-style data scaling curve for the single-decay problem?

## Short Answer

Yes, for validation NLL. Holding the 4-member ensemble recipe fixed and scaling
only per-member training simulations gives a clean monotone loss curve. The
most defensible headline is the raw validation NLL trend, not a precisely
floor-subtracted exponent.

A raw-NLL asymptote fit gives exponent `0.82`, but the fitted asymptote is only
`0.00255` above the current Bayes entropy estimate. That gap is comparable to
the entropy-floor uncertainty (`+/-0.0026`), so it should not be interpreted as a
resolved residual training floor.

The Wasserstein panel also improves monotonically, but it is a posterior
diagnostic rather than the primary scaling-law target.

## Fixed Recipe

The sweep held these details fixed:

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

The x-axis is per-member training simulations `D`; total simulator calls are
`4D`.

## Results

The validation loss is exact equal-weight log-mean-exp ensemble NLL on the
fixed 1M validation cache. The entropy-floor estimate is
`-3.63865 +/- 0.0026` in `z` units after the high-precision adaptive
Gauss-Hermite rerun.

| D per member | Total simulator calls | Ensemble NLL | NLL excess | Panel mean W |
| ---: | ---: | ---: | ---: | ---: |
| 64,000 | 256,000 | -3.54993 | 0.08872 | 0.11046 |
| 128,000 | 512,000 | -3.58579 | 0.05286 | 0.08862 |
| 256,000 | 1,024,000 | -3.60923 | 0.02942 | 0.06506 |
| 512,000 | 2,048,000 | -3.62163 | 0.01703 | 0.05188 |
| 1,024,000 | 4,096,000 | -3.62676 | 0.01190 | 0.04179 |
| 2,048,000 | 8,192,000 | -3.63069 | 0.00796 | 0.03613 |

Power-law fits:

```text
Raw-NLL asymptote = -3.63610
Raw-NLL alpha     = 0.823
Raw-NLL R2        = 0.999

Fixed-entropy excess alpha = 0.704
Fixed-entropy log R2       = 0.992
Excess-alpha sensitivity   = 0.631 to 0.806
                              for entropy_floor +/- 0.0026

Panel W floor     = 0.01457
Panel W alpha     = 0.449
Panel W raw R2    = 0.996
```

## Artifacts

- Main loss-scaling plot:
  `runs/01_exponential_decay/15_broad_scaling/201_flow2_ensemble_data_scaling/figures/flow2_ensemble_data_scaling_weng_style.png`
- Wasserstein diagnostic plot:
  `runs/01_exponential_decay/15_broad_scaling/201_flow2_ensemble_data_scaling/figures/flow2_ensemble_panel_w_diagnostic.png`
- Summary JSON:
  `runs/01_exponential_decay/15_broad_scaling/201_flow2_ensemble_data_scaling/results/flow2_ensemble_data_scaling_summary.json`
- Row CSV:
  `runs/01_exponential_decay/15_broad_scaling/201_flow2_ensemble_data_scaling/results/flow2_ensemble_data_scaling_rows.csv`

## Interpretation

This is now the cleanest single-decay data-axis scaling law in the repo. The
curve supports a Weng/Kaplan-style statement for validation NLL:

```text
For the fixed 4-member Flow2 residual NSF recipe, excess population validation
NLL follows a power-law-like decline over 64k to 2.048M simulations per member.
```

The rightmost points are close enough to the Bayes entropy estimate that floor
uncertainty matters. In raw-loss coordinates, the fitted asymptote is consistent
with the Bayes entropy uncertainty band. In excess-loss coordinates, the
subtraction is useful for a positive log-scale diagnostic, but the exponent and
any residual floor claim are not stable until the entropy/oracle floor is known
more tightly than the remaining excess.
