# NPE UI Extreme Failure Diagnostics

Date: 2026-06-30

## Case

The UI showed a prior-predictive draw where the reference-grid-normalized
Wasserstein distances were enormous:

- Broad spline 4.096M: order `1e7` to `1e11` depending on the sampled draw
  and reference-grid range.
- Broad MDN 512k: order `1e6` to `1e10`.

The visible corner plot did not initially look like a simple grid-edge failure:
the grid posterior was sharp, the true theta dashed lines were near the central
mass, and the edge mass reported by the UI could be tiny.

## Controlled Reproduction

I reproduced the mechanism with a controlled low-noise, high-amplitude signal:

- `theta_true = (A=52.4, k=0.306, sigma=0.045)`
- 100k posterior samples from each selected NPE.
- A UI-style `60^3` reference grid whose range is anchored by broad NPE samples.
- A focused `180^3` grid around the known true theta as a higher-resolution
  reference for the same signal.

Diagnostic artifacts:

- `runs/01_exponential_decay/16_failure_diagnostics/01_ui_extreme_tail_grid/figures/controlled_failure_resolution_and_wasserstein.png`
- `runs/01_exponential_decay/16_failure_diagnostics/01_ui_extreme_tail_grid/figures/controlled_failure_ui_grid_corner.png`
- `runs/01_exponential_decay/16_failure_diagnostics/01_ui_extreme_tail_grid/figures/controlled_failure_focused_grid_corner.png`
- `runs/01_exponential_decay/16_failure_diagnostics/01_ui_extreme_tail_grid/figures/controlled_failure_tail_quantiles.png`
- `runs/01_exponential_decay/16_failure_diagnostics/01_ui_extreme_tail_grid/figures/controlled_failure_signal_predictive.png`
- `runs/01_exponential_decay/16_failure_diagnostics/01_ui_extreme_tail_grid/results/controlled_failure_diagnostics.json`

## Findings

The extreme UI Wasserstein values are largely a reference-grid artifact, but the
underlying posterior is still a real failure case.

For the UI-style `60^3` grid:

- `max_edge_mass = 8.1e-31`, so the usual edge-mass diagnostic says the grid is
  contained.
- The grid step near truth is still too coarse in sharp directions:
  - `A` step near truth: `0.332`
  - `k` step near truth: `0.0432`
  - `sigma` step near truth: `0.00640`
- The grid posterior collapses onto an effectively single `k` slice:
  - grid `k` sd: `9.84e-14`
  - grid `k` median: `0.2968`, while true `k = 0.306`
- The grid compensates by inflating `sigma`:
  - grid `sigma` median: `0.355`, while true `sigma = 0.045`

For the focused `180^3` grid:

- `max_edge_mass = 7.7e-17`.
- The grid step near truth is much finer:
  - `A` step near truth: `0.0176`
  - `k` step near truth: `1.03e-4`
  - `sigma` step near truth: `6.24e-4`
- The grid posterior becomes physically sensible:
  - `A` median: `52.39`
  - `k` median: `0.30605`
  - `sigma` median: `0.0383`

Normalized Wasserstein against the focused grid remains bad:

- Broad spline 4.096M: `418.4`
- Broad MDN 512k: `72.5`

So the giant UI numbers should not be read literally. They are magnified by a
bad reference grid denominator and an under-resolved sharp posterior. However,
even the corrected focused reference shows that both broad amortized models are
far too diffuse and biased for this low-noise corner of the prior. In this
controlled case, the 512k MDN is better than the 4.096M spline under the focused
Wasserstein metric, despite the spline looking better in other aggregate plots.

## Additional Failure Mode

The broad spline has a very small but severe high-`k` tail in this case:

- `k` q99: `0.325`
- `k` q99.9: `27.04`
- `k` max in 100k samples: `27.14`

This tail expands the corner-plot axis and affects tail-sensitive metrics. The
usual q05/q16/median/q84/q95 table hides this because the extreme tail is above
the displayed central quantiles.

## UI Fix Made

The posterior quantiles table now includes explicit `Truth` rows for `A`, `k`,
and `sigma`, instead of relying only on dashed lines in the corner plot and the
small metadata panel.

## Recommended Next Fixes

1. Add reference-grid resolution diagnostics to the UI, not only edge mass:
   grid step near posterior mass, effective occupied bins per parameter, and
   reference posterior sd relative to grid step.
2. Make the sample-mode grid cache key include the actual grid range or disable
   cache reuse when the selected NPE anchor/range changes.
3. For synthetic UI draws where true theta is known, offer a focused reference
   grid centered near the true theta. This should be clearly labeled as a
   diagnostic reference, not as a deployable real-data method.
4. Add tail quantiles such as q99/q99.9/max or a tail warning for NPE samples.
5. Treat this low-noise, high-A regime as a targeted failure set for the next
   model-family and scaling tests.
