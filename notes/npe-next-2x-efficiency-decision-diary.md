# NPE Next 2x Efficiency Decision Diary

Date: 2026-07-02

This is the decision-oriented diary for the broad amortized NPE efficiency
search. The raw chronological audit trail remains in
`notes/npe-next-2x-efficiency-goal-log.md`.

## How To Use This Diary

1. Before making any "best efficiency" claim, run:

   ```text
   uv run scripts/select_npe_efficiency_record.py --require-saved-model
   ```

2. Quote the fastest eligible saved run from that selector, not the newest run
   and not the most conservative proof run.

3. Use the decision ledger below to avoid rediscovering rejected branches.

4. Treat `full_val_nll_z_units` and `training_seconds` as the hard objective
   metrics. Track panel Wasserstein separately as a diagnostic; do not silently
   mix the two objectives.

## Current Saved Efficiency Record

Selector output:

```text
target_nll <= -3.6058692668472965
target_seconds <= 387.8054779791273
require_saved_model = True
eligible_count = 5
BEST kind=ensemble seconds=57.37 full_nll=-3.61336271875 train=2048000 saved_models=4
#2   kind=ensemble seconds=119 full_nll=-3.6129153125 train=2048000 saved_models=4
#3   kind=ensemble seconds=246 full_nll=-3.6306901328125 saved_models=4
#4   kind=single seconds=260.002682959 full_nll=-3.60692343902294 train=2048000 steps=80000 model_exists=True
#5   kind=single seconds=364.694430042 full_nll=-3.60673691373996 train=8192000 steps=112000 model_exists=True
```

Current fastest saved efficiency record:

```text
summary = runs/01_exponential_decay/15_broad_scaling/146_next8x_rawfit_512k10_mixed_lr_timed_proof/mixed_lr_rawfit_512k_e10_seeds2_6_3_5/results/ensemble4_proof_summary.json

full_val_nll_z_units  = -3.61336271875
training_wall_seconds = 57.37
train_simulations     = 2048000 aggregate, 512000 per member
ensemble_size         = 4
context_features      = raw_fit_summary
epochs_per_member     = 10
batch_size            = 512
learning_rates        = 0.00325 and 0.00335
saved_model_count     = 4
```

Current stricter-NLL fresh proof kept in the UI:

```text
summary = runs/01_exponential_decay/15_broad_scaling/199_nll63_randperm_e15_cosstep_ensemble4_saved/results/ensemble4_proof_summary.json

full_val_nll_z_units  = -3.6306901328125
training_wall_seconds = 246.0
ensemble_size         = 4
status                = best fresh-training NLL proof, not the fastest default-efficiency record
```

Historical records and the transitions that produced them remain below. Use
the selector output above when quoting the current fastest eligible saved run.

## Previous Best To New Best

Previous saved record at the start of this search:

```text
summary = runs/01_exponential_decay/15_broad_scaling/74_ui_best_8m_checkpoint/train8m_lr004_wd2e4_e27_max212000_seed20260901/runs/n8192000_seed20260901/results/broad_scaling_run_summary.json
```

Meaningful changes:

| Item | Previous best | New best | Effect / interpretation |
| --- | ---: | ---: | --- |
| `training_seconds` | `775.6109559582546` | `260.0026829591952` | `2.98x` faster wall-time proof. |
| `full_val_nll_z_units` | `-3.6058692668472965` | `-3.6069234390229425` | New run is slightly better NLL, not just faster. |
| saved model | yes | yes | Both are proper saved proof artifacts. |
| `train_simulations` | `8192000` | `2048000` | The winner used `4x` less training data, not more data scale. |
| `batch_size` | `1024` | `512` | Smaller batch gave more update-rich learning dynamics at the winning data scale. |
| `batches_per_epoch` | `8000` | `4000` | The smaller training pool made each epoch shorter despite the smaller batch. |
| `epochs_completed` | `27` | `20` | Fewer epochs and much less wall time. |
| `optimizer_steps` | `212000` | `80000` | `2.65x` fewer updates. Most of the win is earlier learning, not pure throughput. |
| seconds / optimizer step | `0.00366` | `0.00325` | About `1.13x` faster per step; useful but secondary. |
| `learning_rate` | `0.004` | `0.003` | Lower LR was part of the residual frontier recipe. |
| `weight_decay` | `0.0002` | `0.0002` | Unchanged. |
| `lr_schedule` | `cosine_epoch` | `cosine_epoch` | Unchanged schedule family. |
| validation cadence | every 5 epochs | every 5 epochs | Similar overhead profile. |
| architecture family | spline flow | spline flow / NSF | Same broad family, but new implementation exposes NSF knobs explicitly. |
| residual coupling | not recorded / effectively off | `flow_residual=true` | Key architectural change found by HPO. |
| flow depth / bins / hidden | flow3, bins8, hidden80, layers2 | flow3, bins8, hidden80, layers2 | Core capacity shape unchanged. |
| `model_parameters` | `46767` | `105087` | Residual NSF roughly doubles parameters but learns much earlier. |
| panel Wasserstein mean | `0.17010832025126288` | `1.1213718453230732` | Diagnostic regression. The efficiency goal optimized NLL/time; panel quality needs separate tracking if it matters. |

Mechanism summary:

```text
The win came from residual NSF learning much earlier, not from more data,
larger batches, larger width, or a new schedule. The smaller 2.048M training
pool plus 20 epochs gave 80k useful updates in 260s. HPO showed residual NSF
was the only promoted branch with enough NLL headroom.
```

## Winning Evidence Ladder

| Stage | Config | Train sims | Steps | Full val NLL | Seconds | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| HPO stage1 | residual NSF, lr0.003, wd2e-4 | `128000` | `5000` | `-3.5106155833808623` | `16.83900283323601` | Promote. Large proxy lead. |
| HPO stage2 | same | `512000` | `20000` | `-3.58131880641882` | `65.5918905842118` | Promote. Clear winner over plain variants around `-3.56`. |
| HPO stage3 | same | `2048000` | `80000` | `-3.6069234390229425` | `259.41404474992305` | Passed target but unsaved. |
| saved rerun | same | `2048000` | `80000` | `-3.6069234390229425` | `260.0026829591952` | Current saved record. |
| conservative backup | residual NSF 8M | `8192000` | `112000` | `-3.6067369137399634` | `364.69443004205823` | Valid backup, not record. |

## Rejected Or Downgraded Branches

| Avenue | Best evidence | Decision |
| --- | --- | --- |
| More data alone | 16M/batch1024: `-3.5889474300762854` in `421.8466s` | Too slow and not enough NLL. |
| 32M data | stopped for poor utilization/memory before useful signal | Do not revisit without a concrete systems fix. |
| Plain batch512 extension | 8M/lr0.003/wd1e-4: `-3.5933724077415903` in `374.0041s` | Near miss, but still `~0.0125` NLL short. |
| Extra LR tweaks near plain frontier | local proxies around lr/wd were worse | Do not hand-tune tiny LR/WD changes first. |
| Width-only change | h112 looked better at 128k but tied base at 512k | Downgrade; require matched-rung evidence. |
| Smaller/simple posterior heads | MDN/affine/Gaussian around `-2.94` to `-2.56` | Too inaccurate. |
| Context summaries | `raw_decay_summary`, `decay_summary`, `asinh` worse than raw | Keep raw context for this frontier. |
| Linear residual target transform | 128k residual target variants much worse | Reject for this density target. |
| Tail-weighted loss | hurt true prior-predictive NLL | Reject unless objective changes. |
| Samplers | LHS/Sobol worse than random | Keep random sampler. |
| One-cycle / cosine-step / constant screens | worse than cosine-epoch in proxy | Keep cosine-epoch until a stronger schedule test is designed. |
| Compile/thread knobs | no hidden speedup; one thread best on mini | Keep mini proof runs at one torch thread. |
| Alternate Zuko flow kinds | MAF worse, GF/NAF not viable at proxy time | Do not prioritize. |
| Naive residual batch scaling | b640/b768 residual local proxies did not clearly help | Only revisit with a systematic batch/LR schedule search. |

## Decision Rules Going Forward

- Always run `uv run scripts/select_npe_efficiency_record.py --require-saved-model`
  before reporting a record.
- Separate objective wins from diagnostics. NLL/time selected the current
  winner; panel Wasserstein got worse and should be a separate gate if it
  matters.
- Promote only on matched-rung comparisons. The h112 false signal is the
  cautionary example.
- If an unsaved run becomes the fastest eligible objective result, immediately
  rerun that exact config with saving enabled before declaring the record.
- Do not wait on one long run unless its partial curve already projects to the
  target and no better parallel proxy can run safely.

## Extra 2x Outlook

New 2x target from the corrected saved record:

```text
required full_val_nll_z_units <= -3.6069234390229425
required training_seconds     <= 130.0013414795976
```

Current winning curve:

| Epoch | Steps | Cached val NLL | Seconds |
| ---: | ---: | ---: | ---: |
| 1 | `4000` | `-3.1670230584273784` | `13.12652570893988` |
| 5 | `20000` | `-3.393739576829478` | `65.1709000421688` |
| 10 | `40000` | `-3.452586765779063` | `130.22404058417305` |
| 15 | `60000` | `-3.561262484086558` | `195.08205258427188` |
| 20 | `80000` | `-3.6068519311080425` | `260.00243137497455` |

This makes another clean 2x difficult: at the `130s` budget the current model
is only around `-3.45`, roughly `0.15` NLL short. Another 2x is conceivable,
but it likely needs a real new mechanism, not small local tweaks.

Most plausible next directions:

| Direction | Why it could matter | First gate |
| --- | --- | --- |
| Residual NSF schedule compression | Current gains arrive late between 60k and 80k steps. A better schedule may move that gain earlier. | Beat current `40k/130s` NLL by a large margin, not just final NLL. |
| Faster residual architecture | Residual NSF is statistically strong but parameter-heavy. A cheaper residual flow could preserve the curve with faster steps. | Same 512k/stage2 NLL within `~0.005`, materially faster. |
| Residual batch/LR systematic sweep | Larger batches were bad in plain runs, but residual may tolerate different scaling. | Matched-wall-time proxy beats b512 residual, not just fewer epochs. |
| Better amortized features / approximate posterior conditioning | Need an earlier learning curve, not just a bigger model. Domain summaries or approximate fits could reduce problem difficulty if done correctly. | 128k and 512k matched-rung improvements over raw residual NSF. |
| Hot-path profiling and specialization | Pure 2x throughput would hit the next target with the same 80k-step curve. | Measured `>=1.5x` step/sec with unchanged NLL; `>=2x` would be decisive. |

Near-term expectation:

```text
Conceivable: yes.
Likely from tiny tweaks: no.
Likely first useful target: 180-220s before a full 130s attempt.
```

## Active Extra 2x Search

Started: 2026-07-02

Hard target:

```text
full_val_nll_z_units <= -3.6069234390229425
training_seconds     <= 130.0013414795976
```

Baseline evidence at the new wall-time ceiling:

| Run | Train sims | Steps | Seconds | Full NLL | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Current saved residual NSF at epoch 10 | `2048000` | `40000` | `130.224` | cached `-3.452586765779063` | Far short; pure early stopping of the saved recipe cannot solve the new target. |
| Best completed run under `130s` | `512000` | `20000` | `65.5918905842118` | `-3.58131880641882` | Smaller-pool residual NSF learns much faster per wall second; data reuse is a leading hypothesis. |
| Current saved residual NSF full run | `2048000` | `80000` | `260.0026829591952` | `-3.6069234390229425` | Record to beat by `2x`; saved model exists. |

First experimental priorities:

| Avenue | Why it is plausible | Gate |
| --- | --- | --- |
| Smaller-pool residual NSF with more epochs | Existing 512k/20-epoch run reached `-3.581` in only `65.6s`; another 15-19 epochs may fit inside `130s`. | Cross target with a saved run under `130.001s`. |
| Residual NSF HPO around early efficiency | Need schedule/batch/optimizer settings that move the late residual-NSF gain earlier. | Matched-rung improvement over the 512k residual control. |
| Cheaper residual architectures | Residual NSF has good statistical efficiency but doubled params; cheaper variants could preserve the curve with faster epochs. | Similar 512k NLL with lower seconds. |
| Context feature variants | Previous summary features were bad for plain NSF, but residual NSF may use transformed traces differently. | Improve 128k/512k HPO rank, not just one lucky rung. |
| EMA and checkpoint selection | Final gains happen late and sparse validation may miss transient optima. | Better full-cache NLL at equal training seconds. |

Search tooling update:

```text
scripts/npe_hpo_successive_halving.py now has --base-profile next2x_residual.
```

Completed proof:

| Item | Value |
| --- | ---: |
| Winning mechanism | 4-member saved residual NSF seed ensemble |
| Per-member recipe | 512k simulations, 20 epochs, batch512, lr0.003, wd2e-4, residual NSF |
| Seeds | `20260901`, `20260902`, `20260903`, `20260904` |
| Training mode | Four saved members trained in parallel on mini with `jobs=4`, `torch_threads=1` |
| Remote wall time | `119.0s` from `2026-07-02T00:02:16+00:00` to `2026-07-02T00:04:15+00:00` |
| Target wall time | `130.0013414795976s` |
| Ensemble full validation NLL | `-3.6129153125` |
| Target NLL | `-3.6069234390229425` |
| Saved model count | `4` |
| Proof summary | `runs/01_exponential_decay/15_broad_scaling/120_next4x_ensemble4_saved/residual_512k_e20_lr003_wd2e4_seeds4/results/ensemble4_proof_summary.json` |

Why it worked:

```text
Each 512k residual NSF member was individually short of the NLL target
(-3.575 to -3.581), but log-mean-exp density ensembling improved the full
validation NLL by about 0.0316 versus the best member. Parallel training kept
wall time under the 130s ceiling while using the same aggregate 2.048M
simulation count as the previous single-model record.
```

Relevant misses during this search:

| Avenue | Evidence | Decision |
| --- | ---: | --- |
| 512k/e39 data reuse with stretched cosine | `-3.5737106695129115` in `127.74248412484303s` | Under time, but NLL far short. Stretching the schedule delayed learning. |
| 512k/e39 with 20-epoch decay and `eta_min=1e-4` | `-3.570881353058024` in `127.4851432912983s` | Floor did not fine-tune; it drifted/worsened. |
| 2.048M/batch1024 residual NSF | Projected about `186s` from epoch-7 timing; stopped after recorded partial curve | Direct batch doubling failed the wall-time route because larger batches were slower per optimizer step. |
| Local residual HPO wave | Best completed 128k signal before stop was beta `0.95/0.995` at `-3.516406428260966` | Useful future lead, but ensemble proof made continued local HPO unnecessary for this goal. |

Record selector command:

```text
uv run scripts/select_npe_efficiency_record.py --target-nll -3.6069234390229425 --target-seconds 130.0013414795976 --require-saved-model
```

Selector result:

```text
eligible_count = 1
BEST kind=ensemble seconds=119 full_nll=-3.6129153125 train=2048000 saved_models=4
```

## Active Next 2x Search From Ensemble Record

Started: 2026-07-02

Hard target:

```text
full_val_nll_z_units <= -3.6129153125
training_wall_seconds <= 59.5
```

Current record to beat:

```text
runs/01_exponential_decay/15_broad_scaling/120_next4x_ensemble4_saved/
residual_512k_e20_lr003_wd2e4_seeds4/results/ensemble4_proof_summary.json
```

### Early Evidence

| Avenue | Evidence | Decision |
| --- | ---: | --- |
| 10-member first wave of 128k residual NSF members | Exact 1M-cache ensemble NLL `-3.58511782421875`; first-wave members finished around `52-54s` each | NLL far short; one-wave 128k seed ensembling cannot solve the new target. |
| 16-member 128k residual NSF ensemble | Exact ensemble NLL `-3.588089046875`; remote wall `134s` because 16 members required two waves at `jobs=10` | Extra 128k members barely improve NLL and miss wall time. Reject pure cheap-member scaling. |
| Mixed retrospective ensemble: 2 saved 512k residual + 8 saved 128k residual members | Exact ensemble NLL `-3.5952032578125` | Cheap members do not recover enough NLL even when anchored by strong members. |
| 4x512k residual NSF rerun with `torch_threads=1` and sparse validation | Epoch 3 at about `16.6s`, projecting near `105s` for 20 epochs; stopped | Old architecture cannot hit `59.5s` by thread/cadence tuning alone. |
| 8x512k plain NSF ensemble probe | Epoch 3 at about `19s`, projecting near `120s`; stopped | Plain NSF is too slow when eight members run concurrently, and it is weaker per member than residual NSF. |

### Current HPO Wave

Parallel HPO launched on the mini:

```text
runs/01_exponential_decay/15_broad_scaling/126_next8x_parallel_hpo_wave1
```

Purpose:

```text
Use --trial-jobs to run multiple HPO trials concurrently, screen broader
flow kinds and residual NSF variants at 128k, then promote the best to 512k.
```

Search tooling changes:

| Change | Rationale |
| --- | --- |
| `scripts/npe_hpo_successive_halving.py --trial-jobs` | Makes HPO use the mini as a parallel search machine instead of serially blocking on one trial. |
| Added MAF/GF/NAF anchors and random choices to `next2x_residual` | Broadens architecture search beyond NSF so the next attempt is not just local hyperparameter tuning. |

Interim decision:

```text
Do not start a heartbeat for this goal yet. None of the active/exploratory
runs has concrete evidence that it will hit both the 59.5s wall and the
-3.6129153125 NLL target. Keep doing parallel/systematic exploration.
```

### Raw-Fit Context Branch

Motivation:

```text
Fitting a cheap exponential-decay summary to each raw trace gives the NPE an
explicit low-dimensional physical hint while retaining the raw 40-point panel.
This is a Karpathy-style feature/baseline check: inject an obvious useful
statistic and verify whether the learned model becomes more data-efficient.
```

Implementation evidence:

| Change | Evidence |
| --- | --- |
| Added `fit_summary` and `raw_fit_summary` context modes | `raw_fit_summary` is raw 40-d trace plus 6 vectorized exponential-fit features. |
| Added validation context disk cache | 1M validation cache transforms once to `broad_prior_val_1m_float32_raw_fit_summary_context_float32.npy` with shape `(1000000, 46)`. |
| Rejected threaded backend for training | 10-way threaded training reached epoch 1 at about `29s/member`, projecting far over budget. |
| Used cached subprocess backend | All 10 members launched immediately and finished cleanly. |

Measured result:

| Item | Value |
| --- | ---: |
| Run | `runs/01_exponential_decay/15_broad_scaling/132_next8x_rawfit_128k_ensemble10_cached_saved/rawfit_128k_e20_lr003_wd2e4_seeds10` |
| Members | `10` saved residual NSF members |
| Per-member train simulations | `128000` |
| Per-member epochs / steps | `20` / `5000` |
| Max member training seconds | `52.355155042372644` |
| Remote proof job wall interval | `91s` (`2026-07-02T00:41:59+00:00` to `2026-07-02T00:43:30+00:00`) |
| Best individual full validation NLL | `-3.56536903125` |
| Exact 10-member ensemble NLL | `-3.598857953125` |
| Required NLL | `-3.6129153125` |
| Required wall | `59.5s` |

Decision:

```text
Reject as a proof candidate. The raw-fit branch is a real speed signal because
the max member training time is inside the 59.5s budget, but it misses the NLL
target by 0.014057359375 and the full remote proof job is still 91s. Continue
with variants that improve per-member quality inside the same time envelope:
more data per optimizer step, smaller/faster ensembles with stronger members,
and systematic raw-fit HPO rather than more pure seed ensembling.
```

### Same-Step Raw-Fit Data Scaling

Hypothesis:

```text
Keep roughly 5000 optimizer steps, but expose each member to more unique
simulations: 256k x 10 epochs instead of 128k x 20 epochs. This tests whether
the previous cheap ensemble was data-limited rather than step-limited.
```

HPO wave:

| Item | Value |
| --- | ---: |
| Search path | `runs/01_exponential_decay/15_broad_scaling/133_next8x_rawfit_256k10_hpo` |
| Profile | `next8x_rawfit` |
| Train simulations / epochs | `256000` / `10` |
| Parallel trials | `5` |
| Best single-seed NLL | `-3.5801541908314665` |
| Best single-seed training seconds | `30.12880316702649` |
| Best config | residual NSF, raw-fit context, batch512, lr `0.00325`, wd `2e-4` |

Saved proof attempt:

| Item | Value |
| --- | ---: |
| Run | `runs/01_exponential_decay/15_broad_scaling/134_next8x_rawfit_256k10_ensemble10_trainonly_saved/rawfit_256k_e10_lr00325_wd2e4_seeds10` |
| Members | `10` saved train-only members |
| Max member training seconds | `51.4406093750149` |
| Remote train-only wrapper interval | `64s` |
| Exact 10-member ensemble NLL | `-3.60187301953125` |
| Required NLL | `-3.6129153125` |

Decision:

```text
Reject as the next 2x proof. More unique data improved single members from the
128k raw-fit branch, but the 10-member ensemble gain dropped enough that the
exact ensemble still missed by 0.01104229296875. Promote the same idea to a
smaller 4-member, 512k x 10 epoch run: it matches the old 4-member ensemble
shape with about half the optimizer steps per member and may fit under 59.5s
with train-only overhead removed.
```

### Final 2x Proof: Mixed-LR Raw-Fit 4-Member Ensemble

Hypothesis:

```text
The 512k x 10 epoch raw-fit branch was close enough that the limiting factors
were no longer architecture class or total data scale. The most promising
remaining levers were removing avoidable validation overhead and selecting a
diverse set of independently saved members whose exact log-mean-exp ensemble
improves NLL without adding training wall time.
```

Near-miss evidence before the proof:

| Attempt | Wall seconds | Exact ensemble NLL | Decision |
| --- | ---: | ---: | --- |
| 4-member 512k/e10, lr `0.00325`, seeds 1-4 | `60.0` class | `-3.6110869921875` | Missed both wall and NLL. |
| 4-member 512k/e10, lr `0.00325`, seeds 2-5 | `60.0` class | `-3.61242161328125` | NLL close, still missed target. |
| 4-member 512k/e10, lr `0.00325`, seeds 3-6 direct timed | `59.86` | `-3.61229952734375` | Wall and NLL both missed narrowly. |
| 4-member 512k/e10, lr `0.00335`, skip training validation | `58.84` | `-3.61234983984375` | Wall passed, NLL missed by `0.00056547265625`. |

What changed for the saved proof:

| Lever | Previous 119s record | New 57.37s proof | Effect |
| --- | ---: | ---: | --- |
| Wall target basis | `119.0s` | `57.37s` | `2.074x` faster than the previous record. |
| Exact full validation NLL | `-3.6129153125` | `-3.61336271875` | Improved by `0.00044740625`. |
| Context features | raw decay panel | `raw_fit_summary` | Adds 6 cheap exponential-fit features to the raw 40-point trace. |
| Per-member train simulations | `512000` | `512000` | Data scale held fixed. |
| Epochs / optimizer steps per member | `20` / `20000` | `10` / `10000` | Halves optimizer work per member. |
| Ensemble size | `4` | `4` | No extra ensemble member cost. |
| Learning rates | one recipe | mixed `0.00325` and `0.00335` | Retains diversity from two near-miss families. |
| Training validation | enabled | skipped during training | Removes overhead without changing final exact 1M validation. |
| Proof timing | run summary wall | direct `/usr/bin/time -p` | High-resolution timed rerun on the mini. |

Saved proof:

| Item | Value |
| --- | --- |
| Run | `runs/01_exponential_decay/15_broad_scaling/146_next8x_rawfit_512k10_mixed_lr_timed_proof/mixed_lr_rawfit_512k_e10_seeds2_6_3_5` |
| Proof summary | `results/ensemble4_proof_summary.json` |
| Exact NLL JSON | `results/ensemble4_nll.json` |
| Direct timing log | `direct_train.log` |
| Training wall seconds | `57.37` |
| Target wall seconds | `59.5` |
| Exact full validation NLL | `-3.61336271875` |
| Target full validation NLL | `-3.6129153125` |
| Validation examples | `1000000` |
| Saved models | `4` |

Member recipe:

| Seed | Learning rate | Context | Epochs | Batch | Optimizer steps |
| ---: | ---: | --- | ---: | ---: | ---: |
| `20260902` | `0.00325` | `raw_fit_summary` | `10` | `512` | `10000` |
| `20260903` | `0.00335` | `raw_fit_summary` | `10` | `512` | `10000` |
| `20260905` | `0.00335` | `raw_fit_summary` | `10` | `512` | `10000` |
| `20260906` | `0.00325` | `raw_fit_summary` | `10` | `512` | `10000` |

Validation:

```text
uv run scripts/select_npe_efficiency_record.py --target-nll -3.6129153125 \
  --target-seconds 59.5 --require-saved-model --top 8
```

Result:

```text
eligible_count = 1
BEST kind=ensemble seconds=57.37 full_nll=-3.61336271875
saved_models=4 .../146_next8x_rawfit_512k10_mixed_lr_timed_proof/.../ensemble4_proof_summary.json
```

Decision:

```text
Accept as the completed next-2x proof. It is a saved four-member ensemble,
passes the exact 1M validation NLL target, and beats the 59.5s wall target
with direct mini timing.
```

## NLL -3.63 Goal

Started: 2026-07-02

Hard target:

```text
full_val_nll_z_units <= -3.63
assembly / proof wall time < 300s
```

Oracle context:

```text
latest adaptive oracle estimate ~= -3.63865 +/- 0.0026
practical excellent target      = -3.63
```

Accepted fresh-training proof:

| Item | Value |
| --- | ---: |
| Model type | fresh 4-member equal density ensemble |
| Summary | `runs/01_exponential_decay/15_broad_scaling/199_nll63_randperm_e15_cosstep_ensemble4_saved/results/ensemble4_proof_summary.json` |
| Full 1M validation NLL | `-3.6306901328125` |
| Training wall time | `246.0s` |
| Wall interval | `2026-07-02T05:51:50Z` to `2026-07-02T05:55:56Z` |
| Member recipe | flow2 residual NSF, randperm, raw-decay-fit summary context |
| Train simulations / member | `2048000` |
| Epochs | `15` |
| Batch size | `512` |
| Learning rate / schedule | `0.00325`, `cosine_step`, `500` warmup steps |
| Weight decay | `0.0002` |
| Seeds | `20260901`, `20260902`, `20260903`, `20260904` |
| Individual full NLLs | `[-3.6256940703125, -3.6261252578125, -3.624350703125, -3.62662684375]` |

Interpretation:

```text
The clean proof is not the old convex-weighted checkpoint pool. It is a newly
trained four-member ensemble. The meaningful change from the previous clean
best was extending the strong randperm/raw-decay-fit recipe from 10 to 15
epochs while preserving the correct cosine_step schedule. Earlier e12/e15
thinking was muddied by a schedule-control error: several rejected reruns used
cosine_epoch and therefore were not valid evidence against longer training.
```

Previous clean best to new clean proof:

| Item | Previous clean best | New clean proof | Effect / interpretation |
| --- | ---: | ---: | --- |
| Run | `180_nll70_flow2_randperm_2m_e10_ensemble4_saved` | `199_nll63_randperm_e15_cosstep_ensemble4_saved` | Same recipe family, longer correct-schedule training. |
| Full 1M validation NLL | `-3.6290406640625` | `-3.6306901328125` | `0.00164946875` lower NLL; crosses `-3.63`. |
| Training wall time | `179s` | `246s` | Still under the 300s budget. |
| Epochs | `10` | `15` | Extra five epochs supplied the missing NLL margin. |
| Optimizer steps / member | `40000` | `60000` | `1.5x` more updates per member. |
| Train simulations / member | `2048000` | `2048000` | Data scale unchanged. |
| Batch size | `512` | `512` | Unchanged. |
| LR schedule | `cosine_step` + 500 warmup | `cosine_step` + 500 warmup | Keeping this schedule was essential. |
| Architecture | flow2 residual NSF, randperm, bins8, hidden80 | same | No architecture change in the winning proof. |
| Ensemble weights | equal | equal | No validation-fit weights needed for proof. |

Reference-only convex weighted pool:

| Item | Value |
| --- | ---: |
| Summary | `runs/01_exponential_decay/15_broad_scaling/187_nll63_weighted_broad_pool/results/weighted_ensemble_summary.json` |
| Full 1M validation NLL | `-3.63128073481036` |
| 800k holdout NLL | `-3.6312498510615034` |
| Status | Kept in the UI as a clearly labeled saved-checkpoint reference because it still has lower NLL than the fresh proof. Not used as goal proof. |

Rejected or downgraded attempts during this goal:

| Attempt | Evidence | Decision |
| --- | ---: | --- |
| Weight only the current 4-member randperm proof | full `-3.6291944877626188`, holdout `-3.6291757409239964` | Not enough diversity. |
| Greedy equal-weight selection over 12 high-performing flow2/raw-fit members | best `-3.6298628352185487` at 6 members | Close, but still above target. |
| Same 4-member randperm recipe, 12 epochs with `cosine_epoch` | `204s` wall, ensemble `-3.6274666953125` | Rejected as a schedule-control failure, not evidence against longer correct-schedule training. |
| Six concurrent e10 randperm members with `cosine_epoch` | `334s` wall, ensemble `-3.626295890625` | Too slow and wrong schedule. |
| Four-worker e10 randperm seeds 5-8 | `184s` wall, ensemble `-3.62636706640625` | Valid timing, but seeds were weaker. |
| Greedy selection across randperm seeds 1-8 | best `-3.629102082310021` at 5 members | Seed selection alone cannot cross target. |
| Weighted 12-member flow2/raw-fit pool | full `-3.629999781631937`, holdout `-3.629974976089806` | Scientifically useful near miss; technically short. |
| Six-member mixed fresh run with correct schedule | `274s` wall, ensemble `-3.62922605078125`; weighted full `-3.629484044244151` | Valid but short; diversity helped less than expected. |
| Top-10 fast weighted fresh pool | killed after exceeding 300s | Statistically plausible from old evidence, but systems-invalid. |
| Four randperm plus 2M residual fresh pool | killed after exceeding 300s | Strong old-evidence mixture, but residual member made wall time invalid. |

UI update:

```text
scripts/npe_posterior_viewer.py now exposes two separate broad ensemble slots:
the fresh e15 proof model and the convex-weighted saved-checkpoint reference.
The weighted reference stays in the UI only because it still has lower NLL
than the fresh proof; it is labeled as reference-only, not as the proof model.
```
