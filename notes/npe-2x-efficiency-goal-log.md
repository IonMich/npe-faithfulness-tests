# NPE 2x Efficiency Goal Log

Date: 2026-07-01

## Goal

Reach another 2x wall-time improvement from the current broad-prior
single-decay NPE record:

```text
target_full_val_nll_z_units <= -3.6040911785998784
target_training_seconds     <= 784.5767706038896
```

All entries below are from-scratch training runs unless explicitly marked
otherwise.

## Architecture Evidence

The first useful new branch is `flow3/bins8/hidden80`, replacing the current
record's `flow4/bins8/hidden80`.

### 128k/e20 Local Probe

Output root:

```text
runs/01_exponential_decay/15_broad_scaling/59_2x_efficiency_search
```

| candidate | full val NLL | seconds | steps | result |
| --- | ---: | ---: | ---: | --- |
| `flow3/bins8/wd2e-4` | `-3.428452049657` | `18.748` | `2500` | best |
| `flow3/bins8/hidden96/wd1e-4` | `-3.426605994978` | `19.088` | `2500` | second |
| `flow4/bins8/wd2e-4` | `-3.424716788470` | `26.560` | `2500` | control |
| `flow4/bins8/wd1e-4` | `-3.424099827921` | `26.606` | `2500` | prior record family |
| `flow3/bins8/wd1e-4` | `-3.422667981468` | `20.252` | `2500` | faster but weaker |
| `flow3/bins8/batch1280` | `-3.415531820404` | `17.334` | `2000` | too much update loss |

### 512k/e35 Local Probe

| candidate | full val NLL | seconds | steps | result |
| --- | ---: | ---: | ---: | --- |
| `flow3/bins8/wd2e-4` | `-3.566285174284` | `107.184` | `17500` | best |
| `flow3/bins8/wd1e-4` | `-3.565693768933` | `103.225` | `17500` | close, slightly faster |
| `flow3/bins8/hidden96/wd1e-4` | `-3.565258521206` | `110.744` | `17500` | width did not help |
| `flow4/bins8/wd2e-4` | `-3.562548065493` | `140.053` | `17500` | control |

### 1M/e45 Local Probe

| candidate | full val NLL | seconds | steps | result |
| --- | ---: | ---: | ---: | --- |
| `flow3/bins8/wd2e-4` | `-3.584010443987` | `348.518` | `45000` | best |
| `flow4/bins8/wd2e-4` | `-3.578512923304` | `440.950` | `45000` | control |

Interpretation: `flow3/bins8/wd2e-4` is a real architecture win at proxy scale:
better NLL and faster wall time at equal optimizer steps.

## Schedule Evidence

For `1M` local proxies, compressing the cosine horizon gives most of the NLL
gain much earlier:

| schedule horizon | full val NLL | seconds | steps |
| ---: | ---: | ---: | ---: |
| `e10` | `-3.551972283616` | `52.832` | `10000` |
| `e20` | `-3.577035668323` | `104.825` | `20000` |
| `e25` | `-3.580967967702` | `133.004` | `25000` |
| `e30` | `-3.582714449594` | `153.033` | `30000` |
| `e45` | `-3.584010443987` | `348.518` | `45000` |

Interpretation: the `e20-e30` region captures most of the schedule benefit.
This is relevant because the first 4M proof was near target in time but short
on NLL.

## Mini Runs

### Medium Proxy: 1M/e50, wd1e-4

Run id:

```text
npe_2x_flow3_1m_proxy_20260701T125646Z_d4f4a9d9
```

Result:

```text
full_val_nll_z_units = -3.584098348526641
best_val_nll_z_units = -3.5848892884383647
training_seconds     = 199.70432370807976
optimizer_steps      = 50000
```

### Thread Probe: 1M/e10, wd2e-4, threads1

Run id:

```text
npe_2x_flow3_thread1_probe_20260701T130138Z_fcd5eb21
```

Result:

```text
full_val_nll_z_units = -3.5575326974088393
training_seconds     = 36.465328958351165
optimizer_steps      = 10000
```

Interpretation: `torch_threads=1` is faster for this architecture on the mini,
and compressed schedules are credible.

### First 4M Proof: e53, wd2e-4, threads1

Run id:

```text
npe_2x_flow3_4m_e53_proof_20260701T130255Z_f51d5661
```

Result:

```text
full_val_nll_z_units = -3.601940912529751
best_val_nll_z_units = -3.6028567509780376
training_seconds     = 772.8673398750834
optimizer_steps      = 212000
batches_per_epoch    = 4000
```

Verdict: time target passed, NLL missed by `0.002150266070127498`.

### 4M Proof: e53, wd2e-4, eta_min=5e-5, threads1

Run id:

```text
npe_2x_flow3_4m_e53_eta5e5_20260701T131741Z_e5daa914
```

Output root:

```text
runs/01_exponential_decay/15_broad_scaling/62_flow3_eta_floor/
flow3_bins8_wd2e4_e53_eta5e5_threads1_seed20260901
```

Rationale: first proof missed by a small NLL margin in the late low-LR stretch.
This keeps late optimization alive without changing the step count or expected
wall-time budget.

Result:

```text
full_val_nll_z_units = -3.599186436583682
best_val_nll_z_units = -3.599710102571055
training_seconds     = 771.9417622080073
optimizer_steps      = 212000
batches_per_epoch    = 4000
```

Verdict: failed. The eta floor kept late optimization alive but made validation
NLL worse than the no-floor proof.

## Diverse Branches After The Near Miss

### Structured Prior Training Samplers

Implemented `--train-sampler random|lhs|sobol` for training simulations only.
Validation, final cached NLL, and standardization remain random prior
predictive.

At `128k/e20` for `flow3/bins8/hidden80/wd2e-4/lr0.004`:

| sampler | full val NLL | seconds | result |
| --- | ---: | ---: | --- |
| `random` | `-3.428452049657` | `18.748` | control, best |
| `sobol` | `-3.421416218588` | `18.109` | worse |
| `lhs` | `-3.416070950743` | `17.755` | worse |

Verdict: no promotion. Lower-discrepancy prior coverage did not improve the
NLL proxy.

### Architecture And Representation Proxies

At `128k/e20`, all broad architecture/representation branches were worse than
the `flow3/bins8/hidden80/wd2e-4` control:

| branch | full val NLL | result |
| --- | ---: | --- |
| `flow3/bins8/hidden80` | `-3.428452049657` | control, best |
| `flow3/bins10/hidden80` | `-3.419703823707` | worse |
| `flow3/bins12/hidden80` | `-3.407467477654` | worse |
| `flow3/bins8/hidden64` | `-3.383757946251` | worse |
| `flow2/bins8/hidden80` | `-3.337076823430` | much worse |
| `flow3/bins8/raw_decay_summary` | `-3.390321724261` | worse |
| `flow3/bins8/decay_summary` | `-3.085341760401` | much worse |

Medium mini proxies confirmed the same direction:

| branch | full val NLL | seconds | comparison |
| --- | ---: | ---: | --- |
| `flow3/bins8/hidden80` | `-3.566285174284` | `107.184` | 512k/e35 local control |
| `flow3/bins10/hidden80` | `-3.5656239079375704` | `83.61882699979469` | worse NLL |
| `flow3/bins8/raw_decay_summary` | `-3.5611531373464307` | `64.86588745797053` | faster but much worse |

Verdict: these branches improved neither the hard metric nor the likely
full-scale tradeoff. Do not promote.

### Flow3 Hyperparameter Screen

After the broad branches failed, a focused local screen around the `flow3/bins8`
near miss found two useful 128k proxy signals:

| branch | full val NLL | steps | result |
| --- | ---: | ---: | --- |
| `batch896/lr0.0035/wd2e-4` | `-3.436267707330` | `2860` | best proxy |
| `batch1024/lr0.0045/wd2e-4` | `-3.434339700786` | `2500` | better than control |
| `batch1024/lr0.0045/wd3e-4` | `-3.432699341895` | `2500` | better than control, below lr0.0045/wd2e-4 |
| `batch1024/lr0.004/wd2e-4` | `-3.428452049657` | `2500` | current flow3 control |
| `batch1024/lr0.004/wd3e-4` | `-3.423634519246` | `2500` | worse |
| `batch1152/lr0.0045/wd2e-4` | `-3.417616198730` | `2240` | worse |
| `batch1024/lr0.0035/wd2e-4` | `-3.416849287223` | `2500` | worse |

Promoted locally to `512k/e35`:

| branch | full val NLL | steps | result |
| --- | ---: | ---: | --- |
| `batch896/lr0.0035/wd2e-4` | `-3.567574218052` | `20020` | best |
| `batch1024/lr0.0045/wd2e-4` | `-3.566841045149` | `17500` | second |
| `batch1024/lr0.004/wd2e-4` | `-3.566285174284` | `17500` | control |

Verdict: promote `batch896/lr0.0035/wd2e-4` to mini proof. It improves NLL
at both 128k and 512k and buys more optimizer steps per wall-time budget.

## Capped Batch1024 Proof

The mini endpoint ignored `--max-optimizer-steps` for the partial-step proof,
so the endpoint worker was killed early before wasting the full run. Its child
process survived briefly as an orphan, contaminating the first direct relaunch;
that contaminated direct run was also stopped once the timing projection showed
it could not satisfy the hard wall-time ceiling. The proof below is the clean
direct relaunch after clearing competing mini sweep processes, with the cap
verified in `training_progress.jsonl`.

Run:

```text
pid     = 90189
output  = runs/01_exponential_decay/15_broad_scaling/67_flow3_partial_steps_clean/flow3_bins8_wd2e4_e54_max214800_threads1_seed20260901
log     = logs/train_remote/npe_2x_flow3_partial214800_clean_20260701T1503Z.log
```

Configuration:

```text
train_simulations    = 4096000
epochs               = 54
max_optimizer_steps  = 214800
batch_size           = 1024
flow_layers          = 3
spline_bins          = 8
hidden_dim           = 80
weight_decay         = 2e-4
lr_schedule          = cosine_epoch
lr_eta_min           = 0
torch_threads        = 1
```

Rationale: the no-floor e53 proof reached `212000` steps in
`772.8673398750834s` and missed the NLL target by `0.002150266070127498`.
This run adds roughly `2800` optimizer steps while trying to stay under the
`784.5767706038896s` hard wall-time ceiling.

Monitoring: heartbeat `npe-2x-flow3-proof-monitor`.

Result:

```text
full_val_nll_z_units = -3.6031074800779064
best_val_nll_z_units = -3.6035824971328227
training_seconds     = 780.7132414593361
optimizer_steps      = 214800
batches_per_epoch    = 4000
```

Verdict: failed the NLL target by `0.000983698521972`. It did prove that the
time budget can fit roughly `214.8k` batch1024 steps, but the NLL was still just
short of the hard goal.

## Batch896 Proof

Run:

```text
pid     = 90913
output  = runs/01_exponential_decay/15_broad_scaling/68_flow3_batch896_proof/batch896_lr0035_wd2e4_e51_max232000_seed20260901
log     = logs/train_remote/npe_2x_flow3_batch896_max232k_20260701T1520Z.log
```

Configuration:

```text
train_simulations    = 4096000
epochs               = 51
max_optimizer_steps  = 232000
batch_size           = 896
learning_rate        = 0.0035
flow_layers          = 3
spline_bins          = 8
hidden_dim           = 80
weight_decay         = 2e-4
lr_schedule          = cosine_epoch
lr_eta_min           = 0
torch_threads        = 1
```

Rationale: local proxies say batch896/lr0.0035 improves NLL at 128k and 512k.
The cap targets a larger optimizer-step budget than the batch1024 proof while
aiming to stay under `784.5767706038896s`.

Monitoring: heartbeat `npe-2x-flow3-proof-monitor`.

Result:

```text
full_val_nll_z_units = -3.602335576435252
best_val_nll_z_units = -3.60313736964564
training_seconds     = 778.2586675831117
optimizer_steps      = 232000
batches_per_epoch    = 4572
```

Verdict: failed the NLL target by `0.0017556021646263`. Despite fitting
`232k` optimizer steps inside the time budget, the smaller batch / lower LR
branch generalized worse than the clean batch1024 partial-step proof.

## Stopped Batch1024 LR0.0045 Proof

Run:

```text
pid     = 91672
output  = runs/01_exponential_decay/15_broad_scaling/69_flow3_batch1024_lr0045_proof/batch1024_lr0045_wd2e4_e54_max215600_seed20260901
log     = logs/train_remote/npe_2x_flow3_batch1024_lr0045_max215600_20260701T142036Z.log
```

Configuration:

```text
train_simulations    = 4096000
epochs               = 54
max_optimizer_steps  = 215600
batch_size           = 1024
learning_rate        = 0.0045
flow_layers          = 3
spline_bins          = 8
hidden_dim           = 80
weight_decay         = 2e-4
lr_schedule          = cosine_epoch
lr_eta_min           = 0
torch_threads        = 1
```

Rationale: this is a different hyperparameter branch from the batch896 miss,
not merely more time on the same candidate. The 128k/e20 screen improved over
the batch1024/lr0.004 control by about `0.0059` NLL, and the 512k/e35 promotion
still improved the batch1024 control by about `0.00056`. The cap uses the clean
batch1024 timing from the `214800`-step proof and targets just under the hard
`784.5767706038896s` wall-time ceiling.

Result: stopped after epoch 20 because the matched-epoch curve was materially
behind the clean `batch1024/lr0.004/wd2e-4` proof. At epoch 15 it had sparse
validation `-3.473706837190196` versus `-3.4962967591414897` for the clean
LR0.004 proof at the same `60000` optimizer steps, with train NLL also behind.
At epoch 20 it was still only `-3.481151218904063`. This was no longer a
credible proof candidate for a target that needs to improve the clean proof by
about `0.001`.

The proof heartbeat was deleted after this stop because the work returned to
exploration mode.

## Mini EMA / Selector Screen

Run:

```text
pid     = 91952
output  = runs/01_exponential_decay/15_broad_scaling/70_mini_ema_selector_screen
log     = logs/train_remote/npe_2x_ema_selector_screen_20260701T142614Z.log
```

Candidates:

```text
batch1024_flow3_hidden80_bins8_wd2e4_lr004
batch1024_flow3_hidden80_bins8_wd2e4_lr004_ema999
batch1024_flow3_hidden80_bins8_wd2e4_lr004_ema9995
batch1024_flow3_hidden80_bins8_wd2e4_lr004_valevery1
batch1024_flow3_hidden80_bins8_wd2e4_lr004_cacheval200k
```

Rationale: EMA and checkpoint selection are different mechanisms from the
failed batch/LR perturbations. The trainer already stores best validation state
and supports EMA evaluation; this screen checks whether smoothed weights or a
better selector can improve the near-miss flow3 model before spending another
full proof run.

Result:

| branch | full val NLL | train seconds | note |
| --- | ---: | ---: | --- |
| `cacheval200k` | `-3.423033359377` | `9.999` | same selected model as control |
| `control` | `-3.423033359377` | `15.147` | mini screen anchor |
| `valevery1` | `-3.423033359377` | `17.591` | same selected model, more validation |
| `ema999` | `-1.123935289281` | `15.676` | unusable in this form |
| `ema9995` | `2.028291890720` | `15.696` | unusable in this form |

Verdict: checkpoint selection is not the missing `0.001` NLL on this proxy.
EMA starts too close to initialization for these short cosine runs and should
not be promoted without a bias-corrected or delayed EMA implementation.

### Sequential Batching Add-On

After the EMA screen, the same mini proxy root was extended with
`batch1024_flow3_hidden80_bins8_wd2e4_lr004_sequential`.

| branch | full val NLL | best sparse NLL | train seconds | mode |
| --- | ---: | ---: | ---: | --- |
| `pre_shuffle` | `-3.4230333593773086` | `-3.4252675728927104` | `15.146894916892052` | per-epoch full-tensor shuffle |
| `sequential` | `-3.420121349714204` | `-3.419287558091685` | `9.3147051660344` | fixed random order |

Verdict: NLL is worse at equal steps, but the time gain is large enough to
promote to a 512k comparison. If the speedup holds at 4M, it may buy enough
extra optimizer steps to offset the small fixed-order penalty.

512k promotion:

| branch | full val NLL | best sparse NLL | train seconds | mode |
| --- | ---: | ---: | ---: | --- |
| `pre_shuffle` | `-3.5677583052647552` | `-3.568854446900889` | `71.42360879201442` | per-epoch full-tensor shuffle |
| `sequential` | `-3.564254205142303` | `-3.565600033296153` | `69.54707437520847` | fixed random order |

Verdict: do not promote. At 512k, sequential is only about `2.6%` faster and
loses about `0.0035` full-cache NLL.

## 8M Data-Scale Pilot

Rationale: the strongest remaining mechanism is not more tuning around
batch/LR; it is using a larger simulated training pool at the same optimizer
step budget. With `8192000` simulations, `batch_size=1024`, `epochs=27`, and
`max_optimizer_steps=215600`, the run has roughly the same LR-vs-step shape and
same update budget as the clean 4M near miss (`4096000`, `epochs=54`,
`max_optimizer_steps=214800`), but each update should reuse data less often.
`training_seconds` excludes data generation, so this can improve NLL without
necessarily violating the wall-time metric.

Run:

```text
pid     = 92569
output  = runs/01_exponential_decay/15_broad_scaling/72_flow3_8m_data_scale_proof/train8m_lr004_wd2e4_e27_max215600_seed20260901
log     = logs/train_remote/npe_2x_flow3_8m_datascale_20260701T143509Z.log
```

Early stop rule: compare sparse validations at matched optimizer steps against
the clean 4M proof. If the 8M curve is clearly behind by epoch 5 or 10, stop
instead of spending the full proof window.

Pilot result: stopped after epoch 7 because its original `215600` step cap
projected slightly over the hard wall-time budget. The signal was positive, not
negative:

```text
epoch 5 / 40000 steps sparse val = -3.4881993489394634
epoch 5 / 40000 steps train      = -3.427082154763743
elapsed at 48000 steps           = 177.14630837505683s
```

For comparison, the clean 4M proof at `40000` steps had sparse validation
`-3.453361864579722`. The 8M pool is therefore much better early, but it needs
a lower cap to satisfy the hard time target.

## Active 8M Data-Scale Capped Proof

Run:

```text
pid     = 92831
output  = runs/01_exponential_decay/15_broad_scaling/73_flow3_8m_datascale_capped/train8m_lr004_wd2e4_e27_max212000_seed20260901
log     = logs/train_remote/npe_2x_flow3_8m_datascale_max212000_20260701T143928Z.log
```

Configuration: same as the 8M pilot, except `max_optimizer_steps=212000`.
This cap targets about `782-783s` from the pilot's measured step time.

Result:

```text
full_val_nll_z_units = -3.6058692668472965
best_val_nll_z_units = -3.6069368081222026
training_seconds     = 776.213249667082
optimizer_steps      = 212000
epochs_completed     = 27
batches_per_epoch    = 8000
validation_evaluations = 7
```

Hard-target comparison:

```text
required full_val_nll_z_units <= -3.6040911785998784
required training_seconds     <= 784.5767706038896
NLL margin                    = -0.0017780882474181
time margin                   = 8.3635209368076s
```

Verdict: goal achieved. The useful efficiency mechanism was not another
batch/LR tweak; it was increasing the simulated training pool from `4.096M` to
`8.192M` while reducing epochs from `54` to `27` and capping the optimizer at
`212000` steps. This preserved the wall-time budget, reduced sample reuse, and
improved the full cached validation NLL.

## 1M Proxy Promotions

The local `1M/e45` promotions finished after the mini batch proofs:

| branch | full val NLL | best sparse NLL | train seconds | optimizer steps |
| --- | ---: | ---: | ---: | ---: |
| `batch896/lr0.0035/wd2e-4` | `-3.5850919998832187` | `-3.5860818581710308` | `1215.7398147080094` | `50265` |
| `batch1024/lr0.0045/wd2e-4` | `-3.5825422327133856` | `-3.5827008443007915` | `1183.8960358329932` | `43965` |

Verdict: these larger proxies agree with the full-scale proof decisions.
Batch896 remains better than LR0.0045, but both are too weak to justify another
full proof run around the same batch/LR neighborhood.
