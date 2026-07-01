# NPE 1000s Efficiency Research Avenues

Date: 2026-07-01

## Purpose

This note collects research avenues that could plausibly move the broad-prior
single-decay NPE record from the current `1569.1535412077792 s` proof run toward
a much harder `1000 s` target while preserving or improving the final cached
validation NLL.

The current hard reference is:

```text
run:
  runs/01_exponential_decay/15_broad_scaling/58_partial_epoch_budget/
  batch1024_hidden80_wd1e4_lr004_e74_max294000_seed20260901

full_val_nll_z_units = -3.6040911785998784
training_seconds     = 1569.1535412077792
optimizer_steps      = 294000
batch_size           = 1024
hidden_dim           = 80
weight_decay         = 1e-4
learning_rate        = 0.004
```

For a `1000 s` target, a pure throughput improvement must be roughly
`1569.15 / 1000 = 1.57x` at the same optimizer-step count. If throughput does
not improve, the model must reach the same NLL around `187k` optimizer steps.
The present winner was still far from target at comparable time:

```text
epoch 50, 200000 steps, 1067.923 s:
  sparse val NLL = -3.5715058999190776

epoch 70, 280000 steps, 1494.320 s:
  sparse val NLL = -3.603500719560191

epoch 74, 294000 steps, 1569.153 s:
  sparse val NLL = -3.6054531293044536
  full cache NLL = -3.6040911785998784
```

So a `1000 s` target is not a small stop-earlier tweak. It requires either a
large systems speedup, much better step efficiency, or both.

## What The Last 2x Improvement Actually Used

The improvement from the previous `3140.0351539589465 s` record was a local
recipe optimization, not a broad architecture breakthrough.

Important changes:

- Batch size `512 -> 1024`, reducing batches per epoch from about `8000` to
  `4000`.
- Learning rate `0.002 -> 0.004`, preserving large-batch learning dynamics.
- Hidden width `64 -> 80`, increasing parameters from `45844` to `62356`.
- Weight decay `1e-5 -> 1e-4`, which repeatedly helped proxy quality.
- Manual `pre_shuffle` batching, avoiding DataLoader overhead in the hot loop.
- `torch.compile(..., mode="reduce-overhead")` on the mini proof runs.
- Sparse validation every 5 epochs instead of every epoch.
- A hard optimizer-step cap, `max_optimizer_steps=294000`, to use a partial
  extra epoch while staying under the time ceiling.

The successful run was very close to the wall-time boundary:

```text
training_seconds = 1569.1535412077792
time ceiling     = 1570.0175769794733
margin           = 0.8640357716940343 s
```

That matters for the next goal: the current recipe has essentially no slack for
more validation, more parameters, or more epochs.

## Evidence From Things Already Tried

### Larger Batches

The clean extra factor-of-two batch jump has not been shown to work.

Evidence:

- `batch=2048`, `lr=0.008`, `128k/e25` proxy:
  - `full_val_nll_z_units = -3.320054427624746`
  - same-stage `batch=512`, `lr=0.002`, `128k/e25`:
    `-3.385175106073423`
  - This was much worse, not just a small miss.
- `batch=1536`, hidden80/wd/lr0.006, `256k/e25`:
  `-3.5128844305305917`.
- `batch=1536`, bins10/wd/lr0.006, `256k/e25`:
  `-3.4960499001707275`.
- Full-scale `batch=1088`, hidden80/wd/lr0.0042, `4M/e75`:
  `full_val_nll_z_units = -3.5988654162738762`,
  `training_seconds = 1561.7780475406907`.

Interpretation: naive larger batch reduces useful optimizer-update signal too
much. A larger batch might still work with a different optimizer or schedule,
but simple LR scaling is not enough.

### Small Near-Miss Tweaks

The best proxy tweak after the proof was stronger weight decay:

```text
512k/e35 proxy:
  wd=2e-4, batch1024/hidden80/lr0.004:
    full_val_nll_z_units = -3.562548065493

  baseline wd=1e-4, batch1024/hidden80/lr0.004:
    full_val_nll_z_units = -3.561962272433
```

This is real but too small by itself for a `1000 s` goal. It is a useful
component, not a full avenue.

Other local tweaks were not better:

- `lr=0.0038`, `lr=0.0042`, `lr=0.0045` did not beat the current neighborhood.
- Tiny eta floor `5e-6` and `1e-5` did not beat the current neighborhood.
- Cache-subset checkpoint selection did not improve the final proxy metric and
  added time.
- Hidden width `96` was worse and slower at proxy scale.
- Dense validation selected the same full-cache result as sparse validation.
- Sequential batching was faster locally but quality degraded and mini timing
  did not recover enough.

### Architecture And Representation

Some architecture probes exist, but no major new family has been systematically
optimized for the `1000 s` target.

Evidence:

- The winning model is still `spline_flow`, `flow_layers=4`, `spline_bins=8`.
- `flow3/bins10/wd1e-4` had useful proxy signals in earlier 1M-scale tests but
  was not carried through as a fully optimized `batch1024/hidden80` proof path.
- `raw_decay_summary` failed in an earlier full setting, but representation was
  not systematically revisited under the final near-miss recipe.
- MDN and affine-flow families were explored earlier for broader scaling, but
  not as final 1000s contenders with the new batching/compile/step-cap
  infrastructure.

Interpretation: the architecture space is not exhausted. The current record is
a local optimum around one spline-flow recipe.

## What A 1000s Run Would Need

At the current step rate, `1000 s` corresponds to roughly:

```text
294000 steps / 1569.1535 s = 187.36 steps/s
1000 s * 187.36 steps/s     ~= 187000 steps
```

The current run at `200000` steps was only:

```text
sparse val NLL = -3.5715058999190776
```

The final target is around:

```text
full cache NLL <= -3.6040911785998784
```

So step-efficiency work must pull roughly `0.03` NLL forward by about `90000`
steps. That is a large improvement. If we cannot pull learning forward, the
alternative is preserving `~294k` useful steps while making each step about
`1.57x` faster.

## Evidence Audit From Repo Runs And Local Mini-Tests

This section is deliberately more concrete than the avenue list below. It
separates what the repo has already shown from what is still speculative.

### Karpathy-Derived Controls Now Present

The earlier audit in `notes/npe-scaling-karpathy-recipe-audit.md` highlighted:

- dumb/no-context baselines;
- tiny-batch overfit tests;
- systematic tail-failure inspection;
- real capacity axes;
- step-based LR schedules instead of only epoch-based schedules.

The current broad-scaling tooling now covers several of those controls:

- `scripts/decay_broad_scaling_sweep.py` supports `context_variants`:
  `real`, `zero_x`, and `shuffled_x`.
- It saves `validation_top_failures.{json,npz,png}` through `--tail-top-k`.
- It supports `cosine_step`, `lr_eta_min`, `lr_warmup_steps`,
  `max_optimizer_steps`, `validation_every_epochs`, `early_val_cache`, and
  `batching_mode`.
- `scripts/npe_efficiency_search.py` makes the neighborhood search repeatable
  instead of only hand-mutating commands.

The context ablation run at
`runs/01_exponential_decay/15_broad_scaling/35_karpathy_spline_context_ablation_16k`
confirms the model is not succeeding without the observation:

| variant | full 1M val NLL | best val NLL | seconds | epochs |
| --- | ---: | ---: | ---: | ---: |
| `real` | `-2.7909434935567825` | `-2.7254575448165386` | `15.318` | `67` |
| `zero_x` | `3.5939954187480967` | `3.5915151877274067` | `5.031` | `22` |
| `shuffled_x` | `3.5830055257556` | `3.579686288343862` | `5.163` | `22` |

This is a trust check, not an efficiency breakthrough. It says the conditional
signal matters; it does not say how to reach `1000 s`.

### Full 4M Neighborhood

The full-scale record table shows that the `1569 s` run is a multi-change
recipe, not just an epoch trim or batch-size trick.

| run | full 1M val NLL | best val NLL | seconds | epochs | steps | batch | lr | wd | hidden | params | hot path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| old `3140 s` record | `-3.6026432882882795` | `-3.6049624638686626` | `3140.035` | `90` | n/a | `512` | `0.002` | `1e-5` | `64` | `45844` | dataloader/no compile |
| batch1024 old-width run | `-3.5973806372141084` | `-3.597508068574473` | `2576.769` | `90` | n/a | `1024` | `0.004` | `1e-5` | `64` | `45844` | dataloader/no compile |
| near miss `e73` | `-3.602549703009411` | `-3.6030586915145366` | `1557.983` | `73` | `292000` | `1024` | `0.004` | `1e-4` | `80` | `62356` | pre-shuffle/compile |
| current record `e74_max294000` | `-3.6040911785998784` | `-3.6054531293044536` | `1569.154` | `74` | `294000` | `1024` | `0.004` | `1e-4` | `80` | `62356` | pre-shuffle/compile |
| batch1088 attempt | `-3.5988654162738762` | `-3.5999592499862163` | `1561.778` | `75` | `282375` | `1088` | `0.0042` | `1e-4` | `80` | `62356` | pre-shuffle/compile |

Interpretation:

- `batch=1024` alone was not enough. With old width/decay/hot path it missed
  badly at `-3.59738`.
- The successful recipe needed higher width, stronger weight decay, pre-shuffle
  batching, compile, sparse validation, and a partial final epoch.
- The current winner has only `0.864 s` of margin under the previous hard
  `1570.0176 s` ceiling, so further gains cannot come from adding work.
- `batch=1088` saved a few seconds and one epoch-equivalent of wall time, but
  lost about `0.0052` NLL versus the current record.

### Curve Evidence, Not Just Final NLL

The late-curve shape is the strongest argument that a `1000 s` target is hard.

| run | epoch/steps | elapsed shape | sparse val NLL |
| --- | ---: | --- | ---: |
| current record | `50 / 200000` | `1067.923 s cumulative` | `-3.5715058999190776` |
| current record | `65 / 260000` | about `1387.8 s cumulative` | `-3.599768753541514` |
| current record | `70 / 280000` | about `1494.3 s cumulative` | `-3.603500719560191` |
| current record | `74 / 294000` | `1569.154 s total` | `-3.6054531293044536` |
| near miss `e73` | `65 / 260000` | same recipe family | `-3.5964993195662944` |
| near miss `e73` | `70 / 280000` | same recipe family | `-3.6022299485335796` |
| near miss `e73` | `73 / 292000` | `1557.983 s total` | `-3.6030586915145366` |
| batch1088 | `70 / 263550` | faster epoch, fewer steps | `-3.5968450265059917` |
| batch1088 | `75 / 282375` | `1561.778 s total` | `-3.5999592499862163` |

At the current step rate, `1000 s` corresponds to only about `187k` optimizer
steps. The record run was still near `-3.57` at `200k` steps. Therefore the
next target cannot be reached by simply stopping earlier; it needs either a
real throughput gain, a substantially earlier learning curve, or both.

### Systematic 512k/e35 Proxy Neighborhood

The current systematic proxy matrix is useful because it shows which local
tweaks are real and which were noise. Command:

```bash
uv run scripts/npe_efficiency_search.py report \
  --train-simulations 512000 \
  --epochs 35 \
  --limit 80
```

Top completed proxy rows:

| candidate | full val NLL | seconds | steps | interpretation |
| --- | ---: | ---: | ---: | --- |
| `batch1024_hidden80_wd2e4_lr004` | `-3.562548065493` | `147.123` | `17500` | best local tweak; stronger decay helps, but small effect |
| `batch1088_hidden80_wd1e4_lr0042` | `-3.562179599543` | `141.208` | `16485` | faster but later failed full scale |
| `batch1024_hidden80_wd1e4_lr004` | `-3.561962272433` | `145.329` | `17500` | proxy control for current record family |
| `batch1280_hidden80_wd1e4_lr005` | `-3.561864816618` | `121.224` | `14000` | attractive time, but risks too few updates |
| `batch1024_hidden80_wd5e5_lr004` | `-3.561681129288` | `149.131` | `17500` | weaker than `2e-4` |
| `batch1152_hidden80_wd1e4_lr0045` | `-3.561637045370` | `127.854` | `15575` | faster, not better quality |
| `batch1024_hidden80_wd1e4_lr004_eta5e6` | `-3.561495875387` | `152.149` | `17500` | tiny LR floor did not help |
| `batch1024_hidden96_wd1e4_lr004` | `-3.559437612986` | `194.783` | `17500` | wider model is worse and slower |

Proxy conclusions:

- `wd=2e-4` is the cleanest small hyperparameter improvement, but the gain is
  only `0.000586` NLL over the proxy control.
- Fine LR changes around `0.004` and tiny eta floors did not move enough.
- Dense validation and cache-subset checkpoint selection selected the same
  final metric while costing more time.
- The larger-batch proxy rows can look tempting because they are faster, but
  the full `batch1088` run already showed that proxy time alone is not enough.

### Larger-Batch Evidence

The simple factor-of-two batch idea is contradicted by existing proxies.

| run | full val NLL | seconds | epochs | steps | batch | lr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `flow4_d128k_e25`, batch512 | `-3.385175106073423` | `41.281` | `25` | `6250` | `512` | `0.002` |
| `batch2048_lr008_d128k_e25` | `-3.320054427624746` | `43.585` | `25` | `1575` | `2048` | `0.008` |
| `batch1536_hidden80_wd1e4_lr006_d256000_e25` | `-3.5128844305305917` | `36.595` | `25` | `4175` | `1536` | `0.006` |
| `batch1536_bins10_wd1e4_lr006_d256000_e25` | `-3.4960499001707275` | `57.692` | `25` | `4175` | `1536` | `0.006` |

Interpretation:

- `batch=2048` did not just slightly miss; it was dramatically worse at the
  same small-data proxy scale.
- `batch=1536` can be fast, but the tested variants did not preserve NLL.
- Larger batches remain an avenue only if paired with genuinely different
  optimizer dynamics, not just proportional LR scaling.

### Local Training-Step Microbenchmarks

I ran local forward/backward/AdamW microbenchmarks on random tensors using the
current `hidden80/flow4/bins8` family. These measure hot-loop throughput, not
final learning quality.

Batch/thread sweep for the current architecture:

| compile | threads | batch | steps/s | samples/s |
| --- | ---: | ---: | ---: | ---: |
| none | `1` | `512` | `151.57` | `77603` |
| none | `1` | `1024` | `143.98` | `147432` |
| none | `1` | `1536` | `111.77` | `171673` |
| none | `1` | `2048` | `76.93` | `157557` |
| none | `2` | `512` | `160.50` | `82174` |
| none | `2` | `1024` | `122.03` | `124962` |
| none | `2` | `1536` | `84.39` | `129617` |
| none | `2` | `2048` | `72.88` | `149252` |
| none | `4` | `1024` | `119.87` | `122749` |
| `reduce_overhead` | `1` | `1024` | `143.34` | `146784` |
| `reduce_overhead` | `2` | `1024` | `121.81` | `124731` |

Local interpretation:

- Bigger batches increase samples/s up to about `1536`, but reduce optimizer
  steps/s sharply. Since the full-run curves are update-limited, this explains
  why naive larger batches hurt NLL.
- On this local CPU, `threads=1` was best for `batch>=1024`; the proof runs on
  the mini used `torch_threads=2`. This needs a direct mini retest before
  changing production settings.
- `torch.compile(reduce_overhead)` did not improve the isolated local
  microbenchmark after warmup, but the mini proof runs used it successfully in
  the full loop. Treat compile as machine/full-loop-specific, not settled.

Architecture throughput microbenchmark at `batch=1024`, `hidden=80`:

| architecture | threads | steps/s | samples/s | params |
| --- | ---: | ---: | ---: | ---: |
| `flow3/bins8` | `1` | `192.82` | `197452` | `46767` |
| `flow3/bins10` | `1` | `149.49` | `153076` | `51141` |
| `flow4/bins6` | `1` | `125.75` | `128765` | `56524` |
| `flow4/bins8` current | `1` | `145.37` | `148855` | `62356` |
| `flow4/bins10` | `1` | `113.90` | `116637` | `68188` |
| `flow5/bins8` | `1` | `113.95` | `116683` | `77945` |
| `flow3/bins8` | `2` | `162.46` | `166358` | `46767` |
| `flow4/bins8` current | `2` | `123.82` | `126793` | `62356` |

Architecture interpretation:

- `flow3/bins8` is the only tested architecture with enough raw speed to matter
  for a `1000 s` target: about `1.33x` faster in steps/s locally versus
  `flow4/bins8`.
- `flow3/bins10` is not a strong throughput win despite fewer flow layers,
  because the extra bins cost back much of the saving.
- `flow5` and `bins10` move in the wrong direction for wall time unless their
  NLL gain is unexpectedly large.
- The missing evidence is a controlled `flow3/bins8` proxy under the final
  `batch1024/hidden80/wd` recipe. That is now the most concrete architecture
  test to run before any 1000s proof.

### Validation-Tail Evidence

The current record saves top validation failures against the fixed 1M cache:

```text
full_val_nll_z_summary:
  mean = -3.6040911785998784
  sd   = 2.540111860353947
  q999 = 5.467055353151771
  max  = 30.125315789686635
```

Top-20 failure parameter-percentile summary versus the 1M validation cache:

| parameter | failure percentile quantiles |
| --- | --- |
| `A` | `0.00%, 21.06%, 64.16%, 76.12%, 100.00%` |
| `k` | `0.00%, 13.71%, 62.38%, 85.00%, 99.99%` |
| `sigma` | `36.04%, 98.22%, 99.82%, 99.98%, 100.00%` |

Examples from the worst failures:

| rank | NLL | `A` | `k` | `sigma` | percentile pattern |
| ---: | ---: | ---: | ---: | ---: | --- |
| `0` | `30.125` | `5.3536` | `0.00829` | `0.3002` | decay rate at validation minimum |
| `1` | `26.973` | `2.3035` | `0.01264` | `0.5861` | decay rate at validation floor |
| `2` | `26.636` | `4.4473` | `0.7487` | `6.3070` | noise above `99.97%` |
| `4` | `23.777` | `1.2719` | `0.1285` | `8.6812` | noise near maximum |
| `7` | `17.381` | `109.366` | `6.0356` | `0.4324` | amplitude and decay near maximum |
| `8` | `15.797` | `0.0793` | `0.2893` | `0.4455` | amplitude at validation minimum |

Tail interpretation:

- Many of the worst losses are high-noise regimes; the median sigma percentile
  among the top failures is `99.82%`.
- The two worst examples are extremely slow-decay cases with small-to-moderate
  noise, not high-noise examples. A single oversampling rule for high noise
  would miss them.
- Tail-aware training is credible, but it must be stratified across multiple
  hard regimes: high `sigma`, very low `k`, extreme `A`, and combinations.
- Any biased sampler needs validation on the unchanged 1M cache, because mean
  NLL can improve in one tail while degrading the true prior-predictive
  objective elsewhere.

## Revised Avenue Ranking After The Evidence Audit

| rank | avenue | evidence for | evidence against | next useful local test | verdict |
| ---: | --- | --- | --- | --- | --- |
| `1` | `flow3/bins8` cheaper architecture | local microbench gives `1.33x` steps/s and fewer params | no final-recipe proxy yet; shallower flows may miss tail regimes | `512k/e35` control with `batch1024/hidden80/wd1e-4` and `wd2e-4` | highest architecture priority |
| `2` | mini/full-loop profiling and hot-path optimization | pure throughput needs `1.57x`; last win came from hot-loop changes | isolated local compile result was not enough; custom density code is risky | mini 2-5 epoch profile of winner with thread/compile matrix | highest systems priority |
| `3` | nontrivial step-based schedules | target requires pulling learning forward by about `90k` steps | tiny LR/eta tweaks and cosine-step proxy did not help enough | fixed-step proxy schedules: flat-cosine, one-cycle, warmup, eta floors above `1e-5` | medium-high, but needs bigger changes |
| `4` | large batch with changed optimizer dynamics | local samples/s can be higher at `1536`; would reduce epochs/time | naive `1536/2048` proxies were materially worse; full `1088` missed | beta2/LR/wd matrix at `1536`, maybe LAMB/trust-ratio if implemented | high risk, not dead |
| `5` | tail-aware sampling/curriculum | top failures cluster in identifiable regimes | multiple tail types; biased sampling can hurt mean objective | tail-stratified `128k/256k` proxy with fixed 1M-cache eval | promising as a quality/step-efficiency component |
| `6` | context summaries / learned encoder | simple simulator should have sufficient statistics | earlier summary signals were mixed; can discard posterior info | retest `decay_summary` and learned encoder only under final recipe | medium, not first proof path |
| `7` | validation/checkpoint overhead | sparse validation saved real time | far below the `1.57x` throughput requirement and does not improve learning | keep sparse validation, avoid more checkpoint-selector variants | solved enough for now |
| `8` | warm-start/distillation | could reach target wall time if teacher cost excluded | not fair for from-scratch training unless accounted explicitly | maintain separate leaderboard if used | defer unless target definition changes |

## Concrete Next Local Mini-Tests

Before another full 4M proof, the next local work should be a small but
systematic matrix rather than isolated hand edits:

1. `flow3/bins8` under the final recipe:
   - `batch1024`, `hidden80`, `lr=0.004`, `wd=1e-4`;
   - same with `wd=2e-4`;
   - record NLL at fixed steps and wall time.
2. Direct mini full-loop profiling of the current record recipe:
   - `torch_threads=1,2,4`;
   - compile `none` vs `reduce_overhead`;
   - `batch=1024` only at first;
   - separate training, validation, shuffle, optimizer, and flow log-prob time.
3. Larger-batch optimizer rescue:
   - `batch=1536`, `lr=0.005-0.007`, `wd=1e-4/2e-4`;
   - AdamW beta2 brackets if the script exposes them;
   - promote only if NLL at equal wall time beats the `batch1024` control.
4. Step-schedule matrix:
   - fixed `max_optimizer_steps` proxies, not epoch-only comparisons;
   - flat-then-cosine, one-cycle, warmup, and larger eta floors;
   - judge by NLL at `150k`, `180k`, and `200k` steps.
5. Tail-stratified proxy:
   - build strata for high `sigma`, very low `k`, very low/high `A`, and normal
     interior;
   - evaluate on the unchanged 1M cache plus tail quantiles;
   - reject any candidate that only moves the worst 20 while worsening mean NLL.

Promotion to a full 4M proof should require one of these conditions:

- predicted `<=1000 s` at `294k` useful steps from measured mini throughput;
- or a proxy curve showing the current record's `280k`-step NLL level by about
  `180k-200k` steps;
- or a combined speed and curve gain whose extrapolation leaves real slack,
  not another run that needs the final few seconds to barely pass.

## Detailed Avenue Notes

The ranked table above should drive the next experiments. The sections below
keep the original avenue detail for implementation planning and risk review.

### Systems Profiling And Specialized Spline-Flow Hot Path

Hypothesis: the current Zuko NSF path has avoidable overhead for a tiny 3D
target and fixed context shape. A targeted hot-path optimization could increase
steps/s without changing the learning problem.

Why it could matter:

- The 1000s target can be hit with throughput alone if we can get `1.57x`
  speedup at similar optimizer steps.
- The last win already used batching and compile; the next systems win likely
  requires more specific profiling, not generic flags.

Concrete tests:

- Profile the current winning recipe for 2-5 epochs on the mini using
  `torch.profiler` or `sample`, separating model forward/backward, optimizer,
  shuffle/copy, and validation.
- Compare `torch.compile` modes and dynamic/static shape behavior for the
  exact winning batch.
- Check whether constructing `self.flow(x)` inside every `log_prob` call is a
  material overhead source.
- Try a narrow custom spline-flow or coupling-flow implementation for 3D with
  fixed masks, fixed bins, and no general distribution wrapper overhead.
- Benchmark forward+backward samples/s and verify identical or better proxy
  NLL before any 4M run.

Promotion criteria:

- At least `1.25x` steps/s on the mini with no proxy NLL degradation.
- Ideally `1.45x+` steps/s, which would make the `1000 s` target reachable
  mostly as systems work.

Risk:

- Custom density code is easy to get subtly wrong. It needs finite-init,
  tiny-batch overfit, and numerical agreement tests before scale-up.

### Make Larger Batches Work With Different Optimizer Dynamics

Hypothesis: `batch=1536` or `2048` can provide the needed throughput, but naive
LR scaling loses too much optimizer-update signal. Large-batch-specific
optimizer or schedule changes may recover quality.

What has not been tried:

- AdamW beta sweeps for large batch, especially lower `beta2`.
- One-cycle or flat-plus-cosine schedules tuned by optimizer steps.
- Larger LR with warmup and shorter high-LR plateau.
- LAMB/LARS-style trust-ratio optimization for large batch.
- Gradient-noise or regularization compensation for reduced update count.

Concrete tests:

- Use `512k/e35` and `1M/e35` proxies with `batch=1536` and `2048`.
- Hold wall-clock budgets fixed, not epochs, so candidates compete fairly.
- Test:
  - AdamW `betas=(0.9, 0.95)`, `(0.9, 0.98)`, `(0.9, 0.999)`.
  - LR brackets around `0.0045`, `0.0055`, `0.0065`, `0.008`.
  - Step-based one-cycle with a short warmup and nonzero final LR floor.
  - `wd=2e-4`, because it was the strongest new proxy tweak.

Promotion criteria:

- A large-batch proxy must beat the `batch1024/hidden80/wd1e-4/lr0.004`
  proxy at equal or lower wall time.
- Full-scale promotion only if the curve predicts target NLL by `<=1000 s`,
  not merely a faster miss.

Risk:

- Existing evidence says naive `2048` is very bad. This avenue only makes sense
  if paired with optimizer changes.

### Pull Learning Earlier With Optimizer-Step Schedules

Hypothesis: the winning run gets most of its final NLL very late. A better
step-based schedule can reach the same NLL earlier without changing batch size.

Why it could matter:

- Current `cosine_epoch` is tied to epoch count, and the proof uses a partial
  epoch cap. The LR near the final useful steps is extremely low.
- The NLL crosses the target only near epoch 70. Schedule shape matters.

Concrete tests:

- Use fixed `max_optimizer_steps` proxies, not epoch-only proxies.
- Compare:
  - cosine over `220k`, `240k`, `260k`, `280k`, `294k` steps;
  - flat-then-cosine schedules;
  - one-cycle with `pct_start` around `0.05-0.15`;
  - restart-free cosine with eta floors `1e-5`, `3e-5`, `1e-4`;
  - late LR floor only after the current schedule would nearly stop.
- Combine with `wd=2e-4`.

Promotion criteria:

- At `512k` or `1M`, a schedule should make the `200k`-step checkpoint look
  like the current `280k`-step checkpoint at similar or lower time.
- For a mini proof, require a partial curve that is ahead of the current
  winning run by epoch 50/55, not only a better final proxy.

Risk:

- Several small LR/eta tweaks already failed. This avenue needs larger schedule
  changes, not more tiny eta nudges.

### Cheaper Architectures With Similar Final NLL

Hypothesis: a cheaper density estimator can trade some expressivity for much
faster steps, and regain NLL through more steps or better regularization inside
the `1000 s` budget.

Candidates:

- `flow3 + bins10 + hidden80 + wd=2e-4`
- `flow3 + bins8 + hidden80`, if depth dominates step cost
- `flow4 + bins6`, if spline-bin resolution is not critical late
- MDN revisited with the new infrastructure if its step cost is much lower
- hybrid MDN base plus small flow residual, if implementable simply

Concrete tests:

- Benchmark steps/s and proxy NLL jointly; do not rank by NLL alone.
- For each architecture, compute an efficiency score such as:

```text
proxy_score = proxy_final_nll + lambda * log(training_seconds)
```

or compare NLL at fixed wall-clock checkpoints.

Promotion criteria:

- At least `1.3x` faster steps than current flow4/bins8/hidden80 with proxy NLL
  within `0.003-0.005` of the best proxy.
- Or same speed but clearly better early-NLL curve.

Risk:

- Earlier shallow/changed-flow probes were mixed. Architecture changes need a
  controlled matrix around the final recipe, not isolated one-offs.

### Context Representation And Sufficient-Statistic Features

Hypothesis: the raw 40-point time series makes the density estimator learn a
problem-specific encoder from scratch. Better context features could improve
step efficiency enough to reach target earlier.

Why revisit this despite earlier raw-summary failures:

- The earlier representation probes were not systematically paired with the
  final `batch1024/hidden80/wd/lr` recipe.
- The simulator is simple exponential decay. Good summaries should exist:
  approximate log-linear fit, residual scale, early/late averages, and
  amplitude/decay/noise estimates.

Concrete tests:

- Compare `raw`, `decay_summary`, and `raw_decay_summary` under the final
  recipe and under `wd=2e-4`.
- Add a better summary mode based on robust nonlinear least-squares estimates
  or closed-form log-linear estimates with clipping diagnostics.
- Try a learned small context encoder with fewer output features feeding the
  flow, instead of passing all 40 raw points directly as the context to NSF.

Promotion criteria:

- Faster early learning: must beat current NLL at `100k`, `150k`, and `200k`
  steps on proxy runs.
- No tail degradation on top validation failures.

Risk:

- Bad summaries can discard posterior information, especially for high-noise
  or low-amplitude signals. Tail-failure inspection is required.

### Tail-Aware Training Or Sampling

Hypothesis: final mean NLL is partly held back by hard validation-tail regimes.
If those regimes are underrepresented or learned late, stratified training can
improve NLL sooner.

What is now available:

- Runs save top validation failures.
- We can inspect whether high-NLL examples cluster by amplitude, decay rate,
  noise, or observation shape.

Concrete tests:

- Analyze top-k validation failures from the current winner and near misses.
- Build strata from true `z` or simulation summaries:
  - high noise;
  - low amplitude;
  - fast decay;
  - ambiguous signal-to-noise.
- Try balanced minibatches across strata or modest oversampling of high-loss
  regimes.
- If the training distribution is intentionally changed, use importance weights
  or keep a clearly marked unweighted validation objective.

Promotion criteria:

- Better full-cache mean NLL and better tail quantiles at proxy scale.
- No improvement that comes only from overfitting the early-stop validation set.

Risk:

- Biased training distribution can improve some regimes while hurting the true
  prior-predictive objective. This must be measured on the fixed 1M cache.

### Warm Start, Distillation, Or Teacher-Assisted Training

Hypothesis: a student can reach target NLL within `1000 s` if initialized or
trained from a previous high-quality model.

This is only valid if the target allows amortizing or excluding teacher cost.
If the benchmark is "from scratch training time", this avenue is not directly
fair.

Concrete variants:

- Initialize from a smaller-data or earlier checkpoint and count total training
  time explicitly.
- Distill samples/log-probs from the current best model into a cheaper student,
  then fine-tune on true simulator pairs.
- Use the current model to identify hard regions, then train a new model from
  scratch with a better data curriculum.

Promotion criteria:

- Separate "from scratch" and "using existing teacher" leaderboards.
- Do not compare teacher-assisted training to from-scratch records unless the
  accounting is explicit.

Risk:

- Easy to create a misleading wall-time record by hiding pretraining cost.

## Recommended First Search Plan For A 1000s Goal

### Phase 0: Profile Before More Full Runs

Run a 2-5 epoch mini benchmark of the current winner and record:

- steps/s;
- CPU utilization;
- model forward/backward time;
- optimizer time;
- batching/shuffle time;
- validation time;
- effect of `torch_compile` mode and thread count.

Do not launch a 4M proof until the bottleneck is clear.

### Phase 1: Proxy Matrix Focused On 1000s Mechanics

Use `512k` and `1M` proxies with wall-clock and step checkpoints. Candidate
families:

- current recipe control;
- `wd=2e-4` control;
- large-batch optimizer variants (`1536`, `2048`);
- step-based schedules that improve NLL by `200k` steps;
- cheaper architectures (`flow3`, `bins6`, `flow3/bins10`);
- better context summaries.

Record NLL as a function of optimizer steps and wall time for every candidate.
Promotion should be based on early-curve dominance, not final proxy NLL alone.

### Phase 2: One Mini Proof At A Time

Only promote to 4M mini when a candidate plausibly reaches the target by
`<=1000 s`. During a promoted proof, use heartbeat-only monitoring only if the
partial curve is already strong enough to plausibly hit the hard target.

## What Not To Spend On First

- Another naive `batch=2048/lr=0.008` full run. Existing proxy evidence is too
  weak.
- More dense-validation variants. They already selected the same full-cache
  model and mostly add time.
- More tiny eta-floor nudges around the current schedule. They did not beat the
  proxy control.
- Wider hidden-only changes beyond `80`. `hidden96` was slower and worse in the
  proxy.
- Stopping the current recipe earlier. At `~1000 s`, the NLL is far from target.

## Bottom Line

The current `1569 s` record is a strong local optimum around the
`batch1024/hidden80/wd1e-4/lr0.004` spline-flow recipe. It is not evidence that
the broader system is near optimal.

A `1000 s` target is plausible as a serious research goal, but it should be
treated as a new phase. The evidence-backed order is:

1. test `flow3/bins8` under the final recipe, because it is the only measured
   architecture with enough raw step-speed gain to matter;
2. profile the exact winning loop on the mini and tune the hot path/threading
   before spending another 4M proof run;
3. try larger schedule changes that pull NLL forward by steps, not tiny eta/LR
   nudges;
4. revisit larger batches only with changed optimizer dynamics;
5. use the saved tail failures to design stratified sampling or curriculum
   tests, while judging everything on the unchanged 1M validation cache.
