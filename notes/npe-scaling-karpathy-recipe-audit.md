# NPE Scaling Karpathy Recipe Audit

## Question

Are the latest single-decay scaling-law tests ignoring any crucial items from
Andrej Karpathy's training recipe?

Reference: <https://karpathy.github.io/2019/04/25/recipe/>

## Short Answer

The current scaling tests follow many of the recipe's high-value controls, but
they are not yet complete enough to support a strong scaling-law claim. The
main missing pieces are not obvious code failures. They are experiment-trust
checks: dumb baselines, tiny-batch overfit tests, systematic outlier inspection,
and a real model-capacity axis.

## What We Are Already Doing Well

- Fixed seeds and nested training prefixes are used for the local and broad
  scaling sweeps.
- Validation/reference data are fixed across scale points.
- Broad sweeps now use a large `1M` validation-NLL cache for final reported NLL.
- Broad posterior faithfulness is no longer judged only at the original `x0`;
  panel marginal Wasserstein is available and used as the primary W plot when
  the panel cache is supplied.
- High-resolution grid-300 audits corrected the earlier coarse-grid `x0`
  Wasserstein interpretation.
- Runs record train/validation curves, early stopping metrics, posterior
  samples, and tail NLL summaries.
- The notes are appropriately cautious about not overclaiming clean global
  Wasserstein scaling from noisy or single-observation audits.

## Crucial Gaps

### 1. Add Dumb Baselines

Karpathy explicitly recommends simple baselines and an input-independent
baseline. The latest scaling scripts do not yet make this a first-class gate.

Needed checks:

- Train/evaluate a prior-only or zeroed-`x` broad NPE baseline.
- Require the real-input model to beat that baseline on:
  - final `1M` validation NLL;
  - panel marginal W;
  - `x0` W audit.
- For local NPE, add a no-context or shuffled-context baseline and verify it
  degrades as expected.

Why this matters: broad NPE can show smooth NLL improvement while still failing
to use the observation in the posterior-relevant way we expect.

### 2. Add Tiny-Batch Overfit Tests

The current loops go directly into real scaling runs. Before spending on larger
MDNs or flows, each model family should pass a tiny-batch overfit test.

Needed checks:

- For MDN, affine flow, and local spline-flow families, train on a tiny fixed
  batch until the training NLL is near the achievable floor.
- Save the tiny batch and trained predictions/posterior samples.
- Check that training loss can decrease sharply and that samples/predicted
  densities concentrate around the training targets.
- Verify loss-at-initialization is finite and in the expected range.

Why this matters: density models can train silently with shape, standardization,
or tail bugs. A scaling curve should not be the first proof that the model
family can fit the task.

### 3. Inspect Tail Failures As Data, Not Only Statistics

The broad sweep now records NLL tail summaries, which is good. But the `1M`
validation cache exposed extreme max-NLL events in some seeds. Karpathy's data
inspection advice says these examples should be surfaced and inspected.

Needed checks:

- Save the top-k worst validation examples by NLL for every run.
- Plot their observations, true parameters, posterior samples, and predictive
  bands.
- Report whether failures cluster by prior tail, high noise, low amplitude,
  fast decay, or numerical edge cases.
- Compare worst-case sets between base and larger MDNs.

Why this matters: a few bad density tails can dominate mean NLL and can also
indicate that the model is learning the typical region while failing important
posterior regimes.

### 4. Complete The Capacity Axis

The broad Chinchilla-style plan correctly defines two axes:

```text
D = broad prior-predictive simulator pairs
P = trainable NPE parameter count
```

The latest evidence is mostly a strong fixed-architecture data sweep plus one
larger-MDN point at `64k`. That larger model is useful and appears to improve
over the base MDN at `64k`, but it is not a `(D, P)` scaling surface.

Needed checks:

- Run at least small/base/large MDNs across several shared data budgets, e.g.
  `64k`, `128k`, `256k`, and `512k`.
- Keep the same panel marginal cache, `1M` validation cache, seeds, optimizer,
  and standardization protocol.
- Fit NLL and panel-W surfaces separately:

```text
NLL(D, P) = NLL_inf + A * D^(-alpha) + B * P^(-beta)
W(D, P)   = W_inf   + C * D^(-gamma) + E * P^(-delta)
```

- Only then discuss compute-optimal behavior under a `D * P` proxy.

Why this matters: without the capacity axis, we can say the fixed base MDN has
a data-scaling trend, but we cannot yet say whether the estimator is data
limited, model limited, or compute-optimally allocated.

### 5. Watch Local-Flow Learning-Rate Scheduling

Karpathy warns against trusting epoch-based learning-rate schedules. The local
flow training loop uses cosine annealing with `T_max=args.epochs`. When dataset
size changes, the number of optimizer steps per epoch changes, so the LR
schedule in optimizer-step units is not invariant across scale points.

Needed checks:

- Repeat representative local scale points with constant LR.
- Or define LR decay in optimizer steps rather than epochs.
- Compare whether the local W/NLL plateau moves under the schedule change.

Why this matters: an apparent local scaling floor can be partly optimization
schedule related rather than purely data/capacity/reference related.

## Priority Order

1. Add broad zeroed-`x` / prior-only baseline and local shuffled-context
   baseline.
2. Add tiny-batch overfit tests for MDN, affine flow, and local spline flow.
3. Save and inspect top-k validation tail failures from the `1M` cache.
4. Run the true MDN capacity grid across shared `D` values.
5. Audit local-flow LR schedule sensitivity.

## Current Interpretation

The current results are credible as staged evidence:

```text
The fixed base MDN shows broad data scaling on NLL and panel marginal W, and
local NPE shows controlled low/mid-data scaling before metric/reference and
seed effects dominate.
```

They are not yet sufficient for the stronger claim:

```text
We have established a robust Chinchilla-style compute-optimal scaling law for
NPE.
```

The missing recipe items above are the next gates before making that stronger
claim.
