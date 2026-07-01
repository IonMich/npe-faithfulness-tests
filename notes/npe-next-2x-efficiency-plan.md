# NPE Next 2x Efficiency Plan

Date: 2026-07-01

## Target

The current broad-prior single-decay NPE record is:

```text
run:
  runs/01_exponential_decay/15_broad_scaling/73_flow3_8m_datascale_capped/
  train8m_lr004_wd2e4_e27_max212000_seed20260901

full_val_nll_z_units = -3.6058692668472965
training_seconds     = 776.213249667082
optimizer_steps      = 212000
train_simulations    = 8192000
architecture         = spline_flow, flow3, bins8, hidden80
batch_size           = 1024
learning_rate        = 0.004
weight_decay         = 2e-4
```

The next 2x wall-time target is therefore:

```text
required full_val_nll_z_units <= -3.6058692668472965
required training_seconds     <= 388.106624833541
```

This is a harder target than the previous 2x step. The current record needed
`212000` optimizer steps. Reaching the same number of useful steps in `388s`
would require about `2x` hot-loop throughput. If throughput does not improve,
the model must reach record NLL in roughly `106000` optimizer steps.

## Why This Is Not A Small-Tweak Goal

The current 8M record's sparse validation curve, converted to cumulative
training time, was:

| optimizer steps | cumulative seconds | sparse val NLL, standardized | sparse val NLL, z-units |
| ---: | ---: | ---: | ---: |
| `40000` | `146.831` | `-2.818768739700` | `-3.488199348939` |
| `80000` | `293.342` | `-2.791456937790` | `-3.460887547029` |
| `120000` | `439.423` | `-2.887918233871` | `-3.557348843111` |
| `160000` | `585.855` | `-2.901834011078` | `-3.571264620317` |
| `200000` | `731.804` | `-2.933829069138` | `-3.603259678377` |
| `212000` | `776.196` | `-2.937506198883` | `-3.606936808122` |

At the target wall time, the current curve is nowhere near the final NLL. The
next win must therefore come from one of three mechanisms:

- a much earlier learning curve;
- a real systems speedup near `2x`;
- a combination, for example `1.4x` faster steps and `1.4x` fewer steps.

## Working Rules

- Do not wait on one long full-scale run unless partial curves show it is the
  likely winner.
- Run proxy screens in parallel when they are independent and the mini is not
  occupied by a credible proof run.
- Record progress curves, not only final summaries. Every promoted run should
  have NLL versus optimizer steps and cumulative seconds.
- Use the unchanged 1M validation cache for the hard metric. Tail-focused or
  proposal-trained models must not win by changing the evaluation distribution.
- Promote to a full proof only when a proxy gives a concrete path to
  `<=388.1066s`, not merely a better final NLL at long time.

## Highest-Value Avenues

### 1. Data-Scale Plus Step Compression

What worked last time was increasing the training pool from `4.096M` to
`8.192M` while keeping roughly the same optimizer-step budget. The next
question is whether larger fresh-data pools can pull the curve earlier enough
to cut the step budget.

Tests:

| candidate | purpose | first gate |
| --- | --- | --- |
| `D=16.384M`, `max_steps=120k`, same flow3 recipe | test another data-scale jump | beat current 8M curve at `80k` and `120k` steps |
| `D=16.384M`, `max_steps=150k`, same recipe | see if larger pool reaches record NLL before `388s` with any throughput gain | full-cache NLL projection within `0.01` of target by `120k` |
| `D=32.768M`, `max_steps=100k-120k`, if memory permits | aggressive Chinchilla-style data scaling | sparse curve materially ahead of 16M at matched steps |

Promotion gate:

```text
At <=120k steps, sparse z-NLL should be at least around -3.59.
Otherwise data scale alone cannot plausibly reach -3.6059 by 388s.
```

Risk: training_seconds excludes data generation, but memory and standardization
cost can still become operationally painful. Keep this honest by reporting
total elapsed separately from training_seconds.

### 2. Conditional Reparameterization Around A Cheap Approximate Posterior

The raw-context spline flow still has to learn the posterior geometry from a
40-point signal. A more NPE-specific improvement is to condition on a cheap
analytic or numerical summary and train the flow on residual coordinates around
an approximate posterior center/scale.

Candidate design:

- For each simulated signal, compute a fast nonlinear least-squares or gridless
  approximate fit for `log A`, `log k`, and `log sigma`.
- Compute local curvature or robust scale features when cheap enough.
- Train the density estimator on residuals:

```text
r = (z - z_hat(x)) / s_hat(x)
q_phi(r | features(x))
```

- At sampling time, transform samples back by `z = z_hat + s_hat * r`.

Why it could be large:

- It makes the conditional posterior closer to a stationary residual problem.
- It may reduce required flow depth and optimizer steps.
- It is analogous to the successful local linear adjustment, but amortized
  across the broad prior.

First tests:

| candidate | scale | gate |
| --- | ---: | --- |
| residual target with current flow3/bins8 | `128k/e20` | beat raw-context control by `>=0.01` NLL |
| residual target with flow2/bins8 or flow3/bins6 | `128k/e20` | similar NLL with materially faster steps |
| residual target promoted | `512k/e25-e35` | show earlier curve, not just better final |

This is probably the most promising architecture-side idea because it changes
the statistical problem rather than only the neural-network size.

### 3. Tail-Stratified Training With Importance Correction

The worst validation failures cluster in high-noise, very-low-`k`, and extreme
`A` regimes. Pure oversampling would silently change the prior and therefore
the posterior target. The correct variant is proposal sampling with known
importance weights or a mixture objective that preserves the true prior
objective.

Tests:

| candidate | scale | gate |
| --- | ---: | --- |
| true-prior 75% + hard-strata 25%, weighted NLL | `128k/e20` | improve mean NLL without worsening top-tail NLL |
| true-prior 50% + hard-strata 50%, weighted NLL | `128k/e20` | test stronger tail pressure |
| curriculum: hard-strata early, true-prior late | `512k/e35` | better early curve on unchanged 1M cache |

Strata:

- high `sigma`;
- very low `k`;
- very high `A`;
- very low `A`;
- combinations of low `k` and high `sigma`.

Promotion gate: unchanged-cache mean NLL must improve at proxy scale, and the
top validation failures should not just move to a different tail.

### 4. Real Hot-Path Speedup

The current recipe is update-limited. A pure systems win would need nearly
`2x` throughput at batch1024/flow3. Generic knobs are mostly exhausted, so the
next work should profile and remove specific overhead.

Tests:

| candidate | gate |
| --- | --- |
| mini full-loop profiler for flow3 record recipe | identify whether time is flow log-prob, backward, optimizer, shuffle, or validation |
| zuko NSF overhead audit | measure cost of constructing/evaluating `self.flow(x)` per batch |
| hand-specialized 3D coupling/NSF hot path | `>=1.4x` steps/s and numerically matching log-prob on smoke tests |
| compile modes and static batch variants | no NLL loss, measured full-loop gain |

Promotion gate:

```text
>=1.4x measured steps/s with no proxy NLL loss is worth combining with an
earlier-curve method. >=1.8x is worth a direct proof attempt.
```

Risk: custom density code can create fake progress. It needs finite-init,
tiny-overfit, inverse/log-det consistency, and fixed-batch log-prob agreement
tests before full-scale training.

### 5. Optimizer And Schedule Changes That Are Actually Large

Small LR nudges and tiny eta floors already failed. The remaining schedule
space should be step-based and judged at fixed step counts.

Tests:

| candidate | scale | gate |
| --- | ---: | --- |
| flat-then-cosine over `100k-140k` total steps | `512k/e35` | bring final NLL forward by `>=30%` steps |
| one-cycle with short warmup | `512k/e35` | better NLL at matched wall time |
| AdamW beta2 sweep, `0.95/0.98/0.999` | `512k/e35` | earlier convergence without tail blow-up |
| Lion or sign-style optimizer, if simple to add | `128k/e20` | only promote if clearly better than AdamW |

Promotion gate: the curve must beat the current 8M record at matched optimizer
steps, not just finish slightly better at the same budget.

### 6. Larger Batch Only With Different Dynamics

Naive `1536/2048` batch scaling was bad. It remains relevant only if paired
with a different optimizer/schedule, because the target is so wall-time hard.

Tests:

| candidate | scale | gate |
| --- | ---: | --- |
| `batch1536`, beta2 sweep, `lr=0.0045-0.0065` | `512k/e25` | beat batch1024 at equal wall time |
| `batch1536`, one-cycle schedule | `512k/e25` | recover the update-count loss |
| `batch2048`, only after batch1536 shows promise | `128k/e20` | avoid repeating known bad full proxy |

Promotion gate: no full run unless matched-wall-time proxy NLL is better, not
merely faster.

## Immediate Experiment Queue

Run these as a matrix, not as serial one-off bets:

1. `128k/e20` residual-target prototype versus raw-context control.
2. `128k/e20` tail-stratified weighted sampler versus true-prior control.
3. `512k/e25` step-schedule matrix for flow3/8M recipe compressed to
   `80k-140k` effective steps.
4. mini profiler on the current flow3 8M recipe for 2 epochs.
5. `16M` data-scale probe capped at `120k` steps, only after the mini is not
   busy with the UI checkpoint job.

Full-proof promotion requires at least one of:

- projected record NLL by `<=388.1066s`;
- `>=1.8x` full-loop throughput with equal proxy NLL;
- or combined `>=1.3x` throughput and `>=1.3x` step reduction with a curve that
  is already ahead by `80k-120k` steps.

## Heartbeat Policy

Use heartbeat mode only for a run whose partial curve makes it a credible proof
candidate. During exploratory phases, keep running diverse short proxy tests and
record their partial curves. Do not let the mini sit on one long run unless the
evidence says that run can plausibly hit the hard target.
