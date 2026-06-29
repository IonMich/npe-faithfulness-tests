# Plain MCMC MPS performance investigation

Date: 2026-06-22

## Context

The noisy exponential decay random-walk Metropolis sampler uses PyTorch tensors and
can run on CPU or Apple's MPS backend. The original default selected MPS when
available, but direct benchmarks showed that MPS was slower for the default
workload:

- 8 chains
- 24,000 MCMC steps
- 40 observed time points
- 3 inferred parameters: `A`, `k`, `sigma`

## Root cause

The default workload is too small and too sequential for MPS to pay off.

Each Metropolis step depends on the previous accepted state, so the loop cannot be
fully vectorized across time. With only 8 chains and 40 observations, each step
launches many tiny PyTorch/MPS operations over very small tensors. MPS dispatch
and synchronization overhead dominates the actual arithmetic.

This is different from neural-network training or very large batched simulation,
where each kernel does enough work to amortize GPU dispatch overhead.

## Changes made

The sampler now:

- synchronizes MPS/CUDA before and after the timed loop for honest timings
- supports `--sampler-variant baseline`
- supports `--sampler-variant pregenerated`, which draws proposal and acceptance
  random numbers up front
- supports `--sampler-variant low-overhead`, which also uses a lower-overhead
  equivalent log-posterior expression
- defaults to `--sampler-variant low-overhead`
- uses CPU for `--device auto` in this plain-MCMC script; MPS remains available
  through `--device mps`

## Benchmark commands

```bash
uv run scripts/benchmark_mcmc_devices.py \
  --output runs/01_exponential_decay/01_mcmc_hmc_reference/03_mcmc_device_benchmarks/results/mcmc_device_variant_benchmark.json
```

```bash
uv run scripts/benchmark_mcmc_devices.py \
  --variants low-overhead \
  --steps 2000 \
  --burn-in 500 \
  --chains 4096 \
  --repeats 2 \
  --output runs/01_exponential_decay/01_mcmc_hmc_reference/03_mcmc_device_benchmarks/results/mcmc_device_scale_chains4096.json
```

## Default workload results

For 8 chains and 24,000 steps:

| Device / variant | Median sampler time |
| --- | ---: |
| MPS float32 baseline | 4.781 s |
| CPU float64 baseline | 0.968 s |
| MPS float32 pregenerated | 3.802 s |
| CPU float64 pregenerated | 0.926 s |
| MPS float32 low-overhead | 3.660 s |
| CPU float64 low-overhead | 0.825 s |

The low-overhead MPS variant improved MPS time by about 23% versus baseline, but
CPU remained about 4.44x faster than low-overhead MPS for the default workload.

## Large-chain result

For 4,096 chains and 2,000 steps:

| Device / variant | Median sampler time |
| --- | ---: |
| MPS float32 low-overhead | 0.881 s |
| CPU float64 low-overhead | 1.549 s |
| CPU float32 low-overhead | 1.217 s |

At this larger chain count, MPS becomes useful because each step has enough
parallel work to amortize dispatch overhead.

## Practical policy

For the current pedagogical plain-MCMC example, CPU is the right default.

Use MPS only when deliberately testing backend behavior or when running many
parallel chains or a much heavier likelihood. For future NPE/neural-network
training experiments, MPS is still likely worth testing because those workloads
are much more GPU-shaped than this sequential random-walk sampler.
