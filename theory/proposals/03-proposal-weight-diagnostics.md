# Proposal 03: Add Proposal Weight ESS And Clipping Diagnostics

## Claim

The custom local/proposal NPE path should explicitly report proposal correction
health. The code computes weights, but it does not yet expose enough diagnostics
to know whether correction is reliable.

## Literature Signal

- APT/SNPE-C exists because proposal correction in sequential NPE is subtle:
  https://arxiv.org/abs/1905.07488
- TSNPE emphasizes robust, testable proposal design:
  https://arxiv.org/abs/2210.04815
- Preconditioned NPE shows that focusing simulations near the observed data can
  help, but the focus mechanism must be checked:
  https://arxiv.org/abs/2404.13557

## Current Code Touchpoints

In `scripts/npe_flow_stress_tests.py::run_npe`:

```text
weights = make_kernel_weights(...)
log_weight_base = prior_logpdf - log_r
weights *= exp(clip(log_weight_base, -20, 20))
weights /= mean(weights)
```

The summary currently stores `local_weight_summary`, but not effective sample
size, clipping fraction, or raw log-weight range.

## Implementation Sketch

Add helper:

```text
def weight_diagnostics(weights, raw_log_weights=None, clipped_log_weights=None):
    w = np.asarray(weights, dtype=np.float64)
    ess = np.square(w.sum()) / np.square(w).sum()
    return {
        "mean": ...,
        "min": ...,
        "q05": ...,
        "median": ...,
        "q95": ...,
        "max": ...,
        "ess": ess,
        "ess_fraction": ess / len(w),
        "raw_log_weight_summary": ...,
        "clipped_log_weight_summary": ...,
        "clip_low_fraction": ...,
        "clip_high_fraction": ...,
    }
```

Store separate diagnostics for:

- kernel weights only;
- proposal weights only;
- final combined weights.

## Acceptance Criteria

- Every `npe_summary` includes `weight_diagnostics`.
- If `--npe-proposal prior`, proposal correction diagnostics say no proposal
  correction was applied.
- If any clip fraction is nonzero, it is visible in the JSON.
- Run README generation includes ESS fraction and clip fraction.

## First Target

Apply to all custom NPE stress-test runs. For the two-exponential proposal
failures, this will answer whether the HMC-Gaussian proposal path failed because
weights collapsed, because the proposal missed support, or because conditional
density estimation remained biased despite healthy weights.
