# NPE Next 2x Efficiency Goal Log

Date: 2026-07-01

This file is the raw chronological audit log. For future decisions, start with
`notes/npe-next-2x-efficiency-decision-diary.md`, then use this file only for
detailed run context. Before reporting a record, run:

```text
uv run scripts/select_npe_efficiency_record.py --require-saved-model
```

## Hard Target

Current broad amortized NPE record:

```text
full_val_nll_z_units = -3.6058692668472965
training_seconds     = 775.6109559582546
```

Next 2x target:

```text
full_val_nll_z_units <= -3.6058692668472965
training_seconds     <= 387.8054779791273
```

The winning model must be rerun with saving enabled.

## Frontier Reconstruction

Best comparable completed run remains:

```text
runs/01_exponential_decay/15_broad_scaling/74_ui_best_8m_checkpoint/
train8m_lr004_wd2e4_e27_max212000_seed20260901

full_val_nll_z_units = -3.6058692668472965
best_val_nll_z_units = -3.6069368081222026
training_seconds     = 775.6109559582546
optimizer_steps      = 212000
```

Best completed run under the new `387.805s` ceiling before this search:

```text
full_val_nll_z_units = -3.584098348526641
training_seconds     = 199.70432370807976
```

So the new goal is not a small time shave; it needs a materially better early
learning curve, a systems speedup, or both.

## Residual Target Transform

Implemented and smoke-tested:

```text
--target-transform linear_residual
```

This fits a ridge baseline `z_hat(x)` and trains the density model on
`(z - z_hat) / scale`, while evaluating exact log probability in the original
standardized `z` coordinates.

128k/e20 screen:

| branch | full val NLL | seconds | verdict |
| --- | ---: | ---: | --- |
| flow3 control | `-3.4284520496567925` | `28.597` | baseline |
| residual flow3 raw | `-3.2599978482110936` | `28.771` | reject |
| residual flow2 raw | `-3.166701238694533` | `20.368` | reject |
| residual flow3 raw+summary | `-3.180987763799235` | `28.831` | reject |
| residual full Gaussian | `-2.626588766754195` | `7.080` | reject |

Verdict: linear residualization makes this density problem worse. Do not
promote.

## Tail-Weighted Loss

Implemented and smoke-tested:

```text
--loss-weight-mode tail_balanced
```

It upweights high-noise, very-slow-decay, and extreme-amplitude regimes found
in top validation failures, then normalizes mean weight to one.

128k/e20 screen:

| branch | full val NLL | seconds | verdict |
| --- | ---: | ---: | --- |
| flow3 control | `-3.4284520496567925` | `28.597` | baseline |
| tail weight 1 | `-3.3938557277659616` | `21.480` | reject |
| tail weight 3 | `-3.3518588386636217` | `21.290` | reject |
| tail weight 6 | `-3.289174912322565` | `21.380` | reject |

Verdict: the weighted objective hurts the true prior-predictive mean NLL.

## Schedule Screen

Added `--lr-schedule one_cycle` and smoke-tested it.

128k/e20 screen:

| branch | full val NLL | seconds | verdict |
| --- | ---: | ---: | --- |
| cosine_epoch control | `-3.4284520496567925` | `28.597` | baseline |
| one_cycle | `-3.4104019616123877` | `25.587` | reject |
| cosine_step warmup | `-3.4132973722823103` | `25.265` | reject |
| constant | `-3.229154916895792` | `24.708` | reject |

Verdict: the existing cosine-epoch schedule remains best in this proxy.

## Data Scale

Mini 16M/batch1024 probe:

```text
runs/01_exponential_decay/15_broad_scaling/75_next2x_datascale/
flow3_16m_e7_max106k_seed20260901

full_val_nll_z_units = -3.5889474300762854
training_seconds     = 421.84660008270293
optimizer_steps      = 112000
```

The old mini server ignored the requested `106000` step cap and ran all
`112000` steps. Even with the extra steps, data scale alone missed the target
by about `0.0169` NLL.

The mini server was then synced/restarted with the current scripts, and a cap
smoke verified `max_optimizer_steps=3` produced exactly `3` optimizer steps.

32M/batch1024 probe was stopped after more than five minutes with no epoch
output, low CPU utilization, and rising memory. The 16M result already showed
data-scale-alone was far short, so keeping the mini occupied was not justified.

## Small-Batch Update-Rich Branch

128k/e20 screen:

| branch | full val NLL | seconds | steps |
| --- | ---: | ---: | ---: |
| batch1024/lr0.004 control | `-3.4284520496567925` | `28.597` | `2500` |
| batch768/lr0.0035 | `-3.4432334565254887` | `84.452` | `3340` |
| batch512/lr0.002 | `-3.4367450750736435` | `106.166` | `5000` |
| batch512/lr0.003 | `-3.462232639969512` | `105.416` | `5000` |
| batch512/lr0.004 | `-3.47014557893519` | `20.690615833038464` | `5000` |
| batch256/lr0.0015 | `-3.459565248883053` | `97.2015699580079` | `10000` |
| batch256/lr0.002 | `-3.4755029444824896` | `97.0236890000524` | `10000` |

512k/e20 follow-up:

| branch | full val NLL | seconds | steps | verdict |
| --- | ---: | ---: | ---: | --- |
| batch512/lr0.003 | `-3.563442950915142` | `152.0563710409915` | `20000` | not better than prior 512k flow3 control |
| batch512/lr0.004 | `-3.5635502244894943` | `182.67664737498853` | `20000` | not better than prior 512k flow3 control |
| batch256/lr0.002 | `-3.563196849231287` | `224.12502020795364` | `40000` | not better than prior 512k flow3 control |

Mini 8M/batch512/lr0.003 probe:

```text
full_val_nll_z_units = -3.5877881968516547
best_val_nll_z_units = -3.5873886304030864
training_seconds     = 340.8179555842653
optimizer_steps      = 128000
```

16M/batch512/lr0.004 follow-up:

```text
full_val_nll_z_units = -3.5409124644738874
best_val_nll_z_units = -3.5436412053237407
training_seconds     = 351.84611604129896
optimizer_steps      = 128000
```

Verdict: the small-batch branch gives a useful wall-time profile, but lower
data reuse at 16M is much worse. The 8M/batch512 curve is still useful because
it improved sharply late, from `-3.5684682088027446` at 112k steps to
`-3.5873886304030864` cached validation at 128k steps.

## Sampler Screen

Existing 128k/e20 low-discrepancy sampler screen:

| branch | full val NLL | seconds | verdict |
| --- | ---: | ---: | --- |
| random control | `-3.4284520496567925` | `18.74829212500481` | baseline |
| LHS | `-3.4160709507433613` | `17.75537249998888` | reject |
| Sobol | `-3.4214162185875616` | `18.10887970798649` | reject |

Verdict: low-discrepancy prior draws did not improve the broad validation NLL.

## System Throughput

Mini 1M/8k-step benchmark on the current flow3/batch512 recipe:

| compile | threads | training seconds | verdict |
| --- | ---: | ---: | --- |
| reduce_overhead | 1 | `20.6` | tied best |
| reduce_overhead | 2 | `22.7` | slower |
| none | 1 | `20.5` | tied best |
| none | 2 | `22.8` | slower |
| default | 1 | `20.5` | tied best |
| default | 2 | `22.9` | slower |

Verdict: no hidden compile-mode speedup at this slice. Keep mini proof runs at
one torch thread.

## AdamW Beta Screen

Implemented:

```text
--adam-beta1
--adam-beta2
--adam-eps
```

128k/e20 flow3/batch1024 screen:

| beta1 | beta2 | full val NLL | seconds | verdict |
| ---: | ---: | ---: | ---: | --- |
| `0.9` | `0.95` | `-3.428554227402016` | `24.022178457933478` | neutral |
| `0.9` | `0.98` | `-3.4344096573672735` | `24.014827957958914` | small positive |
| `0.9` | `0.99` | `-3.4310740821729144` | `23.864356917096302` | small positive |
| `0.8` | `0.98` | `-3.4168946353135548` | `27.68144554202445` | reject |
| `0.95` | `0.98` | `-3.4457910642729246` | `28.33386604092084` | promote |
| `0.9` | `0.97` | `-3.43117427093248` | `27.812317749951035` | small positive |

512k beta2-only promotion:

```text
beta1 = 0.9
beta2 = 0.98
full_val_nll_z_units = -3.5644
```

Verdict: beta2 alone did not transfer at 512k. The high-beta1 interaction is
the only optimizer signal large enough to combine with the late-improving
8M/batch512 curve.

Active mini candidate:

```text
runs/01_exponential_decay/15_broad_scaling/86_next2x_8m_beta_high_b1/
batch512_lr003_beta095_098_e9_max144k_seed20260901

train_simulations = 8192000
batch_size        = 512
learning_rate     = 0.003
adam_beta1        = 0.95
adam_beta2        = 0.98
max_steps         = 144000
target seconds    <= 387.8054779791273
```

Final result:

```text
full_val_nll_z_units = -3.5923937922610483
best_val_nll_z_units = -3.5932105736861675
training_seconds      = 372.1566766249016
optimizer_steps       = 144000
epochs_completed      = 9
```

Verdict: substantial under-ceiling near miss, but still short of the hard
target by about `0.01348` full validation NLL.

## Plain Batch512 Weight Decay Variant

Motivation: the 144k frontier had about `15.65s` of remaining wall-time
budget and the local proxy did not strongly separate `wd1e-4` from `wd2e-4`.
This was cheap enough to promote because it preserved the known-good 8M
training/data regime.

```text
run_id = npe_next2x_plain_batch512_lr003_wd1e4_144k_probe_20260701T185620Z_05274b7e
output = runs/01_exponential_decay/15_broad_scaling/97_next2x_plain_batch512_wd_sweep/
         batch512_lr003_wd1e4_e9_max144k_seed20260901

full_val_nll_z_units = -3.5933724077415903
best_val_nll_z_units = -3.593134279740855
training_seconds      = 374.00410649972036
optimizer_steps       = 144000
epochs_completed      = 9
```

Verdict: the best under-ceiling full NLL so far, but still short of the hard
target by about `0.01250`. The last epoch gained about `0.0194` cached NLL in
`41.45s`; the remaining wall-time margin is only about `13.80s`, so simple
extension is not sufficient without a better schedule or architecture.

## Fast Architecture Rejection

Local proxy:

```text
output = runs/01_exponential_decay/15_broad_scaling/104_next2x_flow2_proxy/
         flow2_h80_bins8_lr003_128k

full_val_nll_z_units = -3.3745
training_seconds      = 13.1
```

Verdict: flow2 is fast but much too weak at the 128k proxy. Reject as a
frontier architecture for this target.

## Frontier-Centered HPO Wave

The first HPO driver was anchored around the earlier high-beta branch, which
was useful then but stale after the `batch512/lr0.003/default Adam` near miss.
The HPO driver now has a `frontier` profile centered on the current under-time
best:

```text
base batch_size     = 512
base learning_rate  = 0.003
base adam_beta1     = 0.9
base adam_beta2     = 0.999
base weight_decay   = 2e-4
base flow           = NSF, hidden80, flow3, bins8
```

Active systematic mini wave:

```text
pid = 6343
log = logs/npe_next2x_frontier_hpo_wave1_20260701.log
output = runs/01_exponential_decay/15_broad_scaling/105_next2x_frontier_hpo_mini/wave1

stage1 = 28 trials at 128k simulations, 20 epochs
stage2 = promote top 6 to 512k simulations, 20 epochs
stage3 = promote top 2 to 2048k simulations, 20 epochs
```

The wave mixes anchors around LR/WD/capacity with random search over batch
size, LR, Adam betas, weight decay, hidden width/depth, NSF depth/bins,
activation, residual/randperm/pass variants, and a small number of schedule
eta-floor/constant alternatives.

## Parallel Local Proxy Screens During Frontier HPO

Cheap family screen at 128k simulations, 20 epochs, frontier optimizer:

| family | full val NLL | seconds | verdict |
| --- | ---: | ---: | --- |
| `mdn` | `-2.918468208682` | `9.09` | reject |
| `affine_flow` | `-2.943500058048` | `11.68` | reject |
| `full_gaussian` | `-2.610126700824` | `5.44` | reject |
| `diag_gaussian` | `-2.556667056722` | `4.18` | reject |

Conclusion: simpler posterior heads are faster but much too inaccurate.

Frontier context-feature screen at 128k simulations, 20 epochs, `wd1e-4`:

| context features | full val NLL | seconds | verdict |
| --- | ---: | ---: | --- |
| `raw_decay_summary` | `-3.437640531269` | `26.23` | reject |
| `decay_summary` | `-3.135157625470` | `25.81` | reject |
| `asinh` | `-3.364345055764` | `26.82` | reject |
| `asinh_decay_summary` | `-3.379384350067` | `26.98` | reject |

Conclusion: raw trajectory features remain the frontier context representation.

Capacity/batch interaction probes:

| config | train sims | full val NLL | seconds | verdict |
| --- | ---: | ---: | ---: | --- |
| `h128/lr0.003/wd1e-4` | `128000` | `-3.466928627200` | `24.49` | not enough |
| `h112/lr0.0035/wd1e-4` | `128000` | `-3.474190496521` | `23.54` | promote |
| `b640/h112/lr0.00325/wd1e-4` | `128000` | `-3.466045537522` | `20.90` | not enough |
| `h112/lr0.0035/wd1e-4` | `512000` | `-3.564528346454` | `75.48` | tied with base |
| `base/lr0.003/wd1e-4` | `512000` | `-3.564344723044` | `72.32` | baseline |

Conclusion: the width signal does not survive a fair 512k comparison; do not
promote width alone unless the mini HPO finds a stronger interaction.

Residual current-regime promotion:

| config | train sims | full val NLL | seconds | verdict |
| --- | ---: | ---: | ---: | --- |
| `residual/lr0.003/wd2e-4` local | `512000` | `-3.577748928740` | `93.60` | NLL promising, local speed concerning |
| `residual/lr0.003/wd2e-4` mini HPO | `512000` | `-3.5813` | `65.6` | strongest candidate |

The mini timing projects roughly `118k` residual optimizer steps inside the
`387.8054779791273s` proof budget. This is the current best candidate for an
8M proof run if stage3 confirms that the NLL gain survives longer training.

Mini HPO final promotion summary:

| stage | config | train sims | steps | full val NLL | seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| stage1 | `residual/lr0.003/wd2e-4` | `128000` | `5000` | `-3.5106155833808623` | `16.83900283323601` |
| stage2 | `residual/lr0.003/wd2e-4` | `512000` | `20000` | `-3.58131880641882` | `65.5918905842118` |
| stage3 | `residual/lr0.003/wd2e-4` | `2048000` | `80000` | `-3.6069234390229425` | `259.41404474992305` |
| stage3 | `plain/lr0.00325/wd2e-4` | `2048000` | `80000` | `-3.5889033997166595` | `206.80114633310586` |

Conclusion: residual NSF is the only HPO branch with enough NLL headroom. The
stage3 residual run already beats the hard NLL target at 80k steps while staying
below the wall-time target, so it justifies an 8M proof run.

Active no-save proof run:

```text
run_id = npe_next2x_residual_8m_e7_max112k_proof_20260701T195044Z_72d31cec
output = runs/01_exponential_decay/15_broad_scaling/111_next2x_residual_8m_proof/
         residual_lr003_wd2e4_e7_max112k_seed20260901

train_simulations = 8192000
batch_size        = 512
learning_rate     = 0.003
adam_beta1        = 0.9
adam_beta2        = 0.999
flow_residual     = true
flow_kind         = nsf
max_steps         = 112000
epochs            = 7
target NLL        <= -3.6058692668472965
target seconds    <= 387.8054779791273
```

Heartbeat monitor:

```text
automation_id = npe-residual-8m-proof-monitor
cadence       = every 8 minutes
```

If the no-save proof succeeds, rerun the same config with model saving enabled
under `112_next2x_residual_8m_saved`, then mark the goal complete only if the
saved rerun also satisfies the NLL/time target.

No-save proof result:

```text
full_val_nll_z_units = -3.6067369137399634
best_val_nll_z_units = -3.606071348680064
training_seconds      = 366.0543805831112
optimizer_steps       = 112000
epochs_completed      = 7
validation_evaluations = 7
model_pt              = null  # expected: no-save proof
```

Per-epoch cached curve:

| epoch | steps | cached val NLL | elapsed seconds |
| ---: | ---: | ---: | ---: |
| 1 | `16000` | `-3.3775309281478374` | `52.338298291899264` |
| 2 | `32000` | `-3.4343432145247905` | `104.95209341682494` |
| 3 | `48000` | `-3.4883917527327983` | `156.9879312501289` |
| 4 | `64000` | `-3.517833109391734` | `209.23124950006604` |
| 5 | `80000` | `-3.551785822404429` | `261.4462041668594` |
| 6 | `96000` | `-3.5755501942763774` | `313.8248220831156` |
| 7 | `112000` | `-3.606071348680064` | `366.05349637474865` |

This no-save proof satisfies the hard NLL/time target, but the goal still
requires a saved/rerun proof model.

Active saved rerun:

```text
run_id = npe_next2x_residual_8m_e7_max112k_saved_20260701T195842Z_e533979d
output = runs/01_exponential_decay/15_broad_scaling/112_next2x_residual_8m_saved/
         residual_lr003_wd2e4_e7_max112k_seed20260901

save_models = true
same config as the no-save proof
```

Heartbeat monitor was updated to watch the saved rerun and verify that the
saved `model_pt` path is non-null and exists before marking the goal complete.

Saved rerun result:

```text
full_val_nll_z_units = -3.6067369137399634
best_val_nll_z_units = -3.606071348680064
training_seconds      = 364.69443004205823
optimizer_steps       = 112000
epochs_completed      = 7
validation_evaluations = 7
model_pt              = runs/01_exponential_decay/15_broad_scaling/112_next2x_residual_8m_saved/residual_lr003_wd2e4_e7_max112k_seed20260901/runs/n8192000_seed20260901/results/spline_flow_model.pt
model_file_bytes      = 510290
```

Saved per-epoch cached curve:

| epoch | steps | cached val NLL | elapsed seconds |
| ---: | ---: | ---: | ---: |
| 1 | `16000` | `-3.3775309281478374` | `52.181838042102754` |
| 2 | `32000` | `-3.4343432145247905` | `104.29635854205117` |
| 3 | `48000` | `-3.4883917527327983` | `156.20382920792326` |
| 4 | `64000` | `-3.517833109391734` | `208.30282929213718` |
| 5 | `80000` | `-3.551785822404429` | `260.6540729170665` |
| 6 | `96000` | `-3.5755501942763774` | `312.68674254231155` |
| 7 | `112000` | `-3.606071348680064` | `364.69320324994624` |

Final verdict: achieved. The saved rerun satisfies the hard target
(`-3.6067369137399634 <= -3.6058692668472965` and
`364.69443004205823 <= 387.8054779791273`) and has a saved model artifact.

Validation checks:

```text
uv run python -m py_compile scripts/npe_stage1_decay.py \
  scripts/decay_broad_scaling_sweep.py \
  scripts/submit_remote_broad_scaling.py \
  scripts/train_remote_server.py \
  scripts/npe_hpo_successive_halving.py
```

Result: passed.

## Faster Saved Proof Correction

The HPO stage3 run at `2,048,000` training simulations was not only a screen:
it already satisfied the target faster than the later conservative 8M saved
proof. The 8M saved proof remains valid, but it is not the best efficiency
record. To make the faster result a proper saved artifact, reran the exact
stage3 residual configuration with model saving enabled.

Saved faster rerun:

```text
run_id = npe_next2x_residual_2m_e20_saved_20260701T231520Z_06a4488f
output = runs/01_exponential_decay/15_broad_scaling/113_next2x_residual_2m_saved/
         residual_lr003_wd2e4_e20_seed20260901

train_simulations     = 2048000
epochs                = 20
batch_size            = 512
learning_rate         = 0.003
adam_beta1            = 0.9
adam_beta2            = 0.999
flow_residual         = true
flow_kind             = nsf
optimizer_steps       = 80000
validation_evaluations = 5
full_val_nll_z_units  = -3.6069234390229425
best_val_nll_z_units  = -3.6068519311080425
training_seconds      = 260.0026829591952
model_pt              = runs/01_exponential_decay/15_broad_scaling/113_next2x_residual_2m_saved/residual_lr003_wd2e4_e20_seed20260901/runs/n2048000_seed20260901/results/spline_flow_model.pt
model_file_bytes      = 510226
```

Faster saved per-evaluation cached curve:

| epoch | steps | cached val NLL | elapsed seconds |
| ---: | ---: | ---: | ---: |
| 1 | `4000` | `-3.1670230584273784` | `13.12652570893988` |
| 5 | `20000` | `-3.393739576829478` | `65.1709000421688` |
| 10 | `40000` | `-3.452586765779063` | `130.22404058417305` |
| 15 | `60000` | `-3.561262484086558` | `195.08205258427188` |
| 20 | `80000` | `-3.6068519311080425` | `260.00243137497455` |

Correct final efficiency record for this target:

```text
full_val_nll_z_units = -3.6069234390229425
training_seconds      = 260.0026829591952
saved_model           = true
```

This supersedes the conservative 8M saved proof for the efficiency claim.

## Guardrail Against Misreporting the Record

The final record must be selected by a mechanical eligibility/ranking check,
not by the most recent completed proof run. Added:

```text
scripts/select_npe_efficiency_record.py
```

Required command before future "best efficiency" claims:

```text
uv run scripts/select_npe_efficiency_record.py --require-saved-model
```

Current output:

```text
eligible_count = 2
BEST seconds=260.002682959 full_nll=-3.60692343902294 train=2048000 steps=80000 model_exists=True
#2   seconds=364.694430042 full_nll=-3.60673691373996 train=8192000 steps=112000 model_exists=True
```

Rule: quote the fastest eligible saved run unless explicitly discussing a
different constraint. A slower passing run is backup evidence, not the record.

Partial result before stopping:

| epoch | steps | cached val NLL | elapsed training seconds |
| ---: | ---: | ---: | ---: |
| 1 | `16000` | `-3.3892780499587505` | `41.98283800017089` |
| 2 | `32000` | `-3.4079900937209575` | `83.34318841714412` |
| 3 | `48000` | `-3.4363249497542827` | `124.69281075010076` |
| 4 | `64000` | `-3.3815287308822124` | `166.08002466708422` |

Verdict: stopped. At matched 64k steps this was far worse than the 9-epoch
run (`-3.4905163006911724`) and had no plausible path to the target.

Active mini probe:

```text
run_id = npe_next2x_plain_batch512_lr004_144k_probe_20260701T185201Z_8a014ef8
output = runs/01_exponential_decay/15_broad_scaling/96_next2x_plain_batch512_lr004_144k/
         batch512_lr004_e9_max144k_seed20260901

train_simulations = 8192000
batch_size        = 512
learning_rate     = 0.004
max_steps         = 144000
epochs            = 9
target NLL        <= -3.6058692668472965
target seconds    <= 387.8054779791273
```

Partial result before stopping:

| epoch | steps | cached val NLL | elapsed training seconds |
| ---: | ---: | ---: | ---: |
| 1 | `16000` | `-3.3322507100234477` | `41.54319087509066` |
| 2 | `32000` | `-3.3888748841414897` | `83.0034182080999` |
| 3 | `48000` | `-3.4235755162368267` | `124.40995750017464` |
| 4 | `64000` | `-3.4359515862594097` | `166.05877225007862` |

Verdict: stopped. lr0.004 was far behind lr0.003 at matched steps
(`-3.4359515862594097` versus `-3.4905163006911724` at 64k).

Active mini probe:

```text
run_id = npe_next2x_plain_batch512_lr003_wd1e4_144k_probe_20260701T185620Z_05274b7e
output = runs/01_exponential_decay/15_broad_scaling/97_next2x_plain_batch512_wd_sweep/
         batch512_lr003_wd1e4_e9_max144k_seed20260901

train_simulations = 8192000
batch_size        = 512
learning_rate     = 0.003
weight_decay      = 0.0001
max_steps         = 144000
epochs            = 9
target NLL        <= -3.6058692668472965
target seconds    <= 387.8054779791273
```

Local focused proxy after the 144k near miss:

| branch | full val NLL at 128k proxy | verdict |
| --- | ---: | --- |
| lr0.003 / wd5e-5 | `-3.4544` | worse |
| lr0.003 / wd3e-4 | `-3.4503` | worse |
| lr0.00275 / wd2e-4 | `-3.4455` | worse |
| lr0.00325 / wd2e-4 | `-3.4572` | worse |

Verdict: nearby lr/wd nudges do not improve the cheap proxy. If wd1e-4
misses at full scale, seed/data variance is a better next branch than more
manual lr/wd tweaks.

Final result:

```text
full_val_nll_z_units = -3.5923937922610483
best_val_nll_z_units = -3.5932105736861675
training_seconds      = 372.1566766249016
optimizer_steps       = 144000
epochs_completed      = 9
```

Verdict: new best under the wall-time ceiling, but misses the hard NLL target
by about `0.01348`. The measured per-step time leaves roughly 6k additional
steps before the ceiling, so the next probe is the same recipe with a
10-epoch cosine horizon capped at 150k optimizer steps.

Active mini probe:

```text
run_id = npe_next2x_plain_batch512_lr003_150k_probe_20260701T184745Z_60a7938c
output = runs/01_exponential_decay/15_broad_scaling/95_next2x_plain_batch512_150k/
         batch512_lr003_e10_max150k_seed20260901

train_simulations = 8192000
batch_size        = 512
learning_rate     = 0.003
max_steps         = 150000
epochs            = 10
target NLL        <= -3.6058692668472965
target seconds    <= 387.8054779791273
```

## Systematic HPO And Architecture Screen

Implemented a custom successive-halving HPO driver:

```text
scripts/npe_hpo_successive_halving.py
```

The driver records every command/config/result in JSONL, ranks by full cached
validation NLL, and promotes the top candidates from 128k to 512k. I also
exposed real Zuko NSF conditioner knobs end-to-end:

```text
--flow-activation
--flow-residual
--flow-randperm
--flow-passes
--flow-kind
```

Mini HPO wave:

```text
runs/01_exponential_decay/15_broad_scaling/90_next2x_arch_hpo_mini/focused_arch_wave1
```

128k stage:

| candidate | full val NLL | best val NLL | seconds | steps | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| residual NSF | `-3.508508994441076` | `-3.510335321916148` | `16.910784708335996` | `5000` | promote |
| coupling passes=2 | `-3.4699623770425756` | `-3.470453377259776` | `13.313571209087968` | `5000` | promote but weak |
| batch896/flow4/bins10 random | `-3.464104717772647` | `-3.4659951882491558` | `19.325224625412375` | `2860` | promote but weak |
| batch768/randperm random | `-3.4611100611421306` | `-3.461049433244273` | `11.037247208878398` | `3340` | promote but weak |
| randperm anchor | `-3.451691908` | `-3.453580494` | `13.289` | `5000` | reject |
| GELU/SILU/ELU anchors | `-3.3578` / `-3.2893` / `-3.2901` | | `15-18` | `5000` | reject |

512k promoted stage:

| candidate | full val NLL | best val NLL | seconds | steps | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| residual NSF | `-3.5753046043888768` | `-3.5763462739119976` | `65.24507066560909` | `20000` | clear winner |
| batch896/flow4/bins10 random | `-3.5602940777459584` | `-3.559773798478648` | `73.7052089581266` | `11440` | reject |
| coupling passes=2 | `-3.5590772223270375` | `-3.5629910187850444` | `51.64035850018263` | `20000` | reject |
| batch768/randperm random | `-3.553558784451886` | `-3.5556100564132183` | `42.45726912515238` | `13340` | reject |

Local architecture probes:

| branch | full val NLL | seconds | verdict |
| --- | ---: | ---: | --- |
| Zuko MAF, 128k | `-3.404099350` | `9.297` | reject |
| Zuko GF, 128k | no finite validation summary | | reject |
| Zuko NAF, 128k | stopped after >2 min without summary | | reject for wall-time target |
| residual NSF, lr 0.0035 | `-3.507145992` | local parallel timing inflated | tied with lr 0.004 |
| residual NSF, lr 0.0045 | `-3.498390711` | local parallel timing inflated | worse |
| residual NSF + randperm | `-3.497406309` | local parallel timing inflated | worse |
| residual NSF + passes=2 | `-3.476243192` | local parallel timing inflated | worse |
| residual NSF, hidden80/layers1 | `-3.491080753` | `79.828` | worse |
| residual NSF, hidden64/layers1 | `-3.462499935` | `75.264` | reject |
| residual NSF, batch1024 | `-3.482928147` | `18.312` | worse |
| residual NSF, Adam betas 0.9/0.999 | `-3.5101` | local parallel timing | tied/slightly better 128k proxy |
| residual NSF, Adam betas 0.95/0.98 | `-3.5044` | local parallel timing | neutral |
| residual NSF, one-cycle LR | `-3.5048` | local parallel timing | neutral |

Verdict: residual NSF is the first new architecture signal large enough to
promote. It improves NLL at 128k and 512k, but it is slower per step, so it
needs a capped 8M probe to determine whether the NLL gain beats the wall-time
penalty.

Active promoted mini probe:

```text
run_id = npe_next2x_residual_8m_112k_probe_20260701T183108Z_25af17ac
output = runs/01_exponential_decay/15_broad_scaling/93_next2x_residual_8m_probe/
         residual_batch512_lr004_8m_max112k_seed20260901

train_simulations = 8192000
batch_size        = 512
learning_rate     = 0.004
adam_beta1        = 0.98
adam_beta2        = 0.99
flow_residual     = true
max_steps         = 112000
epochs            = 7
validation_every  = 2
target NLL        <= -3.6058692668472965
target seconds    <= 387.8054779791273
```

Partial curve after launch:

| epoch | steps | cached val NLL | elapsed training seconds |
| ---: | ---: | ---: | ---: |
| 1 | `16000` | `-2.9936486439834087` | `52.41405741730705` |
| 2 | `32000` | `-3.2909156041274517` | `105.1482807090506` |
| 4 | `64000` | `-3.4591444687972515` | `210.43854829203337` |

The run started behind the old batch512 8M curve but recovered by epoch 4.
It is no longer an obvious early-stop, so it should be allowed to finish.

Final result:

```text
full_val_nll_z_units = -3.580043688569828
best_val_nll_z_units = -3.5816584782729595
training_seconds      = 368.21293275011703
optimizer_steps       = 112000
epochs_completed      = 7
```

Verdict: under the wall-time ceiling, but worse than the old batch512 8M
128k run (`-3.5873886304030864` at `340.8179555842653s`). Reject residual
NSF for the current target.

## Next Plain Batch512 Extension

The old batch512/lr0.003 8M curve improved late:

| epoch | steps | cached val NLL | elapsed seconds |
| ---: | ---: | ---: | ---: |
| 6 | `96000` | `-3.5482836918960063` | `255.68032870907336` |
| 7 | `112000` | `-3.5684682088027446` | `298.13124895934016` |
| 8 | `128000` | `-3.5873886304030864` | `340.81711641699076` |

One additional epoch projects to about `383s`, still below
`387.8054779791273s`. The last two epoch gains were about `0.020` and
`0.019` NLL, which is close to the remaining gap to the target. This is a
better-justified mini probe than more residual variants.

Active mini probe:

```text
run_id = npe_next2x_plain_batch512_lr003_144k_probe_20260701T183940Z_5ab599cc
output = runs/01_exponential_decay/15_broad_scaling/94_next2x_plain_batch512_144k/
         batch512_lr003_e9_max144k_seed20260901

train_simulations = 8192000
batch_size        = 512
learning_rate     = 0.003
adam_beta1        = 0.9
adam_beta2        = 0.999
flow_residual     = false
max_steps         = 144000
epochs            = 9
target NLL        <= -3.6058692668472965
target seconds    <= 387.8054779791273
```

## Completed Next 2x Efficiency Proof

The latest hard target was another 2x improvement over the saved 4-member
residual NSF ensemble:

| Item | Previous saved record | New saved proof |
| --- | ---: | ---: |
| Training wall seconds | `119.0` | `57.37` |
| Speedup | | `2.074x` |
| Exact full validation NLL | `-3.6129153125` | `-3.61336271875` |
| Target wall seconds | | `59.5` |
| Target NLL | | `-3.6129153125` |
| Saved ensemble members | `4` | `4` |

Proof artifact:

```text
runs/01_exponential_decay/15_broad_scaling/146_next8x_rawfit_512k10_mixed_lr_timed_proof/
  mixed_lr_rawfit_512k_e10_seeds2_6_3_5/results/ensemble4_proof_summary.json
```

The winning recipe is a 4-member residual NSF ensemble with `raw_fit_summary`
context, 512k simulations per member, 10 epochs, batch size 512, skipped
training-time validation, and mixed learning rates:

| Seed | LR | Exact individual NLL |
| ---: | ---: | ---: |
| `20260902` | `0.00325` | `-3.59383205078125` |
| `20260903` | `0.00335` | `-3.60132440625` |
| `20260905` | `0.00335` | `-3.59381384765625` |
| `20260906` | `0.00325` | `-3.59825075` |

Validation command:

```text
uv run scripts/select_npe_efficiency_record.py --target-nll -3.6129153125 \
  --target-seconds 59.5 --require-saved-model --top 8
```

Selector result:

```text
eligible_count = 1
BEST kind=ensemble seconds=57.37 full_nll=-3.61336271875 saved_models=4
```
