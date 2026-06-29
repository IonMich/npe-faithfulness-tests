# Broad Scaling NPE Wall-Time Investigation

Date: 2026-06-29

## Active Run

Investigated this broad-prior MDN sweep:

```bash
uv run scripts/decay_broad_scaling_sweep.py \
  --preset pilot \
  --output-root runs/01_exponential_decay/15_broad_scaling/07_mdn_panel512k \
  --train-simulations 64000,128000,256000,512000 \
  --seeds 20260901,20260902,20260903 \
  --val-simulations 100000 \
  --validation-cache runs/01_exponential_decay/15_broad_scaling/validation_cache/broad_prior_val_1m_float32.npz \
  --panel-marginal-cache runs/01_exponential_decay/15_broad_scaling/panel_marginal_cache/decay_panel16_grid180_marginals.npz \
  --panel-posterior-samples 20000 \
  --skip-existing
```

Configuration:

| Setting | Value |
| --- | ---: |
| Device | CPU |
| MDN hidden dim | 128 |
| MDN hidden layers | 3 |
| MDN components | 5 |
| Parameters | 44,722 |
| Epoch cap | 90 |
| Batch size | 512 |
| Early-stop validation pairs | 100,000 |
| Final cached NLL pairs | 1,000,000 |
| Panel signals | 16 |
| Posterior samples per panel signal | 20,000 |

## Code Path

The sweep is serial:

1. `scripts/decay_broad_scaling_sweep.py` parses the preset and command-line overrides.
2. It loads the cached `x0` reference and panel marginal reference.
3. It builds one standardization sample and one early-stop validation sample.
4. For each seed, it samples one nested training pool of size `max(train_simulations)`.
5. For each train size, `run_one(...)` trains one MDN with `stage1.train_one_model(...)`.
6. After training, each run evaluates final cached NLL, samples the original `x0`, evaluates the 16-signal panel, writes per-run JSON/NPZ, and eventually aggregates plots/CSV/JSON.

The training loop is in `scripts/npe_stage1_decay.py::train_one_model`. The MDN log density is in `MixtureDensityPosterior.log_prob`, which calls `batched_mixture_log_prob`.

## Completed Run Timings

The run completed successfully. Aggregate wall time before final aggregation was `2280.8 s`; summed reported training time was `2151.9 s`. That means training accounted for about `94%` of the measured run time before aggregation.

Per-run rows:

| Seed | Train signals | Epochs | Training seconds | Final 1M NLL | Panel mean W |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20260901 | 64,000 | 90 | 38.3 | -2.9539 | 0.5354 |
| 20260902 | 64,000 | 90 | 59.9 | -2.9334 | 0.4655 |
| 20260903 | 64,000 | 90 | 54.1 | -2.8933 | 0.6133 |
| 20260901 | 128,000 | 90 | 68.0 | -3.2217 | 0.3495 |
| 20260902 | 128,000 | 90 | 103.3 | -3.1760 | 0.3759 |
| 20260903 | 128,000 | 84 | 64.4 | -3.1560 | 0.3985 |
| 20260901 | 256,000 | 90 | 307.2 | -3.3437 | 0.2965 |
| 20260902 | 256,000 | 90 | 175.8 | -3.3235 | 0.3450 |
| 20260903 | 256,000 | 90 | 123.1 | -3.3437 | 0.2362 |
| 20260901 | 512,000 | 90 | 454.5 | -3.4108 | 0.2340 |
| 20260902 | 512,000 | 90 | 361.0 | -3.4329 | 0.2184 |
| 20260903 | 512,000 | 85 | 342.4 | -3.3955 | 0.2374 |

Median by train size:

| Train signals | Seeds | Median training seconds | q16-q84 seconds | Median panel W | Median final NLL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64,000 | 3 | 54.1 | 43.4-58.0 | 0.5354 | -2.9334 |
| 128,000 | 3 | 68.0 | 65.5-92.0 | 0.3759 | -3.1760 |
| 256,000 | 3 | 175.8 | 140.0-265.2 | 0.2965 | -3.3437 |
| 512,000 | 3 | 361.0 | 348.4-424.6 | 0.2340 | -3.4108 |

## Profiling Observations

A 5-second `sample` of the active Python process landed in the training loop, not simulator generation or file I/O. The heavy native frames were consistent with the custom MDN density path:

- `exp` and `log` kernels
- SiLU forward/backward kernels
- small batched triangular solves from `torch.linalg.solve_triangular`
- BLAS `strsm`
- tensor view/indexing overhead from constructing and slicing tiny Cholesky tensors

This matches the static code path: `batched_mixture_log_prob` repeatedly builds `3 x 3` lower-triangular matrices and solves tiny triangular systems for every batch and mixture component.

## Benchmarks

Small synthetic MDN step benchmarks on this machine showed:

| Backend | Batch size | Approx throughput |
| --- | ---: | ---: |
| CPU | 512 | 111k samples/s |
| CPU | 1024 | 147k samples/s |
| CPU | 2048 | 185k samples/s |
| CPU | 4096 | 146k samples/s |
| MPS | 512-4096 | 15k-17k samples/s |

Conclusion: do not switch this MDN sweep to MPS. CPU is the right backend for this workload on this machine.

The machine has 8 CPU cores, 16 GB RAM, and PyTorch reported `torch.get_num_threads() == 4`. Under live load, a small benchmark suggested `torch.set_num_threads(2)` can be better for larger batches, especially if running more than one training process.

The final cached NLL pass is secondary but still improvable. The sweep currently evaluates the final 1M cache with the training batch size (`512`). A standalone benchmark on 200k cached pairs took:

| Eval batch size | Seconds for 200k pairs |
| ---: | ---: |
| 512 | 1.59 |
| 16,384 | 0.80 |

This is not the main bottleneck, but separating training batch size from final-eval batch size is low risk.

## Wall-Time Reduction Opportunities

### 1. Parallelize independent runs

Expected gain: high.

The largest opportunity is that every `(seed, train_simulations)` cell is independent, but the sweep runs them serially. The observed process generally used only a fraction of the available CPU. Two concurrent workers should plausibly approach a 2x wall-time reduction for future large sweeps, subject to memory pressure. Three or more workers may be risky on a 16 GB machine because the live process peaked around several GiB.

Best implementation:

- add a `--jobs` option or a simple claim-file mechanism;
- force `torch.set_num_threads(1 or 2)` per worker;
- keep per-run output directories unchanged;
- aggregate after all workers finish.

### 2. Tune training batch size and thread count

Expected gain: medium to high, but needs an A/B run.

The current `batch_size=512` is not obviously compute-optimal. Synthetic training throughput improved at larger batch sizes, especially around `1024-2048`, and `torch.set_num_threads(2)` looked better than the default 4 for large batches under live load.

Risk: increasing batch size changes the number of optimizer updates per epoch. Treat this as a controlled experimental change, not a drop-in replacement inside an existing scaling run. A robust version would report both epochs and optimizer steps, or switch the sweep to a fixed-step budget.

### 3. Keep CPU, avoid MPS for this MDN

Expected gain from MPS: negative.

MPS was much slower for the small full-covariance MDN kernels tested here. MPS may still help larger neural flow workloads, but it should not be used for this broad MDN training path without a fresh end-to-end A/B benchmark.

### 4. Add a separate final NLL eval batch size

Expected gain: low to medium.

Final cached NLL currently uses `args.batch_size`, which is tuned for training dynamics rather than evaluation throughput. Add something like:

```text
--eval-batch-size 16384
```

and use it in `evaluate_val_nll_z_summary(...)`.

This should save seconds per run and reduce overhead, but it will not change the main training bottleneck.

### 5. Decimate or shrink early-stop validation

Expected gain: medium.

The sweep evaluates the 100k early-stop validation tensor every epoch. Since the final reported NLL comes from the 1M cache, the 100k set is only for checkpoint selection. Options:

- validate every 2-5 epochs;
- use a smaller early-stop set, such as 20k, while keeping the 1M final NLL;
- keep 100k only for final checkpoint audits.

Risk: best validation epochs in the completed rows were usually late, around epochs 76-85, so the epoch cap should not be aggressively reduced without checking quality.

### 6. Optimize the MDN log-prob implementation

Expected gain: medium, higher engineering risk.

The profiler points at tiny `3 x 3` triangular solves and tensor-view overhead. A specialized 3D Cholesky log-prob could avoid generic `torch.linalg.solve_triangular` and some repeated tensor construction. This needs numerical equivalence tests against the current implementation because it touches the density objective directly.

## Recommended Next Step

For the next broad sweep, prioritize infrastructure rather than changing the model:

1. Add `--jobs`, `--torch-threads`, and `--eval-batch-size`.
2. Run a short controlled timing/quality A/B for `batch_size=512,1024,2048`.
3. Keep CPU as the default backend.
4. Only then repeat large 512k-style sweeps.

## Implemented Infrastructure

Implemented after this investigation:

- `scripts/decay_broad_scaling_sweep.py --jobs N` now parallelizes by seed.
  Each worker still trains all requested `D` values serially for that seed, so
  the per-seed training pool stays nested. The parent aggregates once at the
  end.
- `--torch-threads` limits PyTorch CPU threads per worker and also sets the
  usual BLAS thread-count environment variables for child workers.
- `--eval-batch-size` separates final cached-NLL evaluation throughput from the
  training batch size.
- `--skip-x0-reference` allows fresh-machine panel-W/NLL scaling runs without
  the large local `300^3` x0 grid cache.
- `scripts/train_remote_server.py` runs a generic `/train` endpoint on a remote
  machine. It does not encode scaling-law logic; it updates the repo, runs
  optional setup commands, and launches a requested repo-local training command.
- `scripts/submit_remote_broad_scaling.py` keeps the broad-scaling logic local
  by constructing the cache-prep commands and the broad sweep command, then
  submitting that JSON request to `/train`.

The default broad-scaling train request intentionally runs panel-W and NLL
first. x0-specific W can be added later by transferring or regenerating the x0
grid cache on the remote machine and changing the submitted training command.
