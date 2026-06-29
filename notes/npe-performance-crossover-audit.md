# NPE Performance And Amortization Audit

Date: 2026-06-25

## Question

Do the current experiments show a case where neural posterior estimation (NPE) beats MCMC/HMC by speed, amortization, scaling, or GPU use at comparable posterior discrepancy?

## Investigation Plan

1. Inventory existing MCMC, HMC, NPE, ABC, and device benchmark summaries.
2. Separate five claims:
   - speed to the same posterior discrepancy,
   - amortized accuracy over a region of observations,
   - scaling curves and crossover behavior,
   - MPS/GPU utilization,
   - literature-backed regimes where NPE should win.
3. Extract only comparable numbers where the discrepancy target is explicit.
4. Add a small post-training flow sampling and training-kernel benchmark, because the saved summaries did not isolate amortized sampling speed.
5. Define the next experiment needed to test the crossover claim fairly.

## Current Local Evidence

### 1. Speed To Same Discrepancy

There is no current case where NPE is faster end-to-end than MCMC/HMC at the same discrepancy level.

For the exponential-decay model, the strict grid-reference comparison is:

| Method | Device | Discrepancy | Runtime |
| --- | --- | ---: | ---: |
| MCMC | CPU | 0.03348 | 1.47 s |
| HMC | CPU | 0.03162 | 6.67 s |
| best NPE flow | CPU | 0.03310 | 421.68 s total, including 311.25 s training and 103.23 s data generation |

The successful NPE run reaches essentially the same faithfulness target, but it is about two orders of magnitude slower end-to-end on this toy model.

For the stress-test families, the NPE discrepancy is agreement against converged MCMC/HMC rather than an analytic/grid posterior. Successful NPE runs exist for sign, banana, label-switch, and linear6, but MCMC/HMC are still faster:

| Family | Best NPE diagnostic metric | NPE total | MCMC | HMC |
| --- | ---: | ---: | ---: | ---: |
| sign | 0.02691 | 220.13 s | 0.74 s | 14.39 s |
| banana | 0.01844 | 129.66 s | 1.50 s | 14.78 s |
| label-switch | 0.02868 | 285.34 s | 1.21 s | 23.77 s |
| linear6 | 0.03301 | 337.51 s | 5.86 s | 24.07 s |
| ordered two-exponential | 0.05317, target not met | 442.74 s | 15.31 s | 50.31 s |

Conclusion: NPE currently demonstrates posterior matching on several cases, not end-to-end speed dominance.

### 2. Amortization Across A Region Of Signals

The current repo does not yet show faithful amortization at the 0.034 target over a region of observations.

The decay multi-x runs tested 8 observations. None passed:

| Run | Best family by median discrepancy | Median | Min | Max | Pass fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| `08_npe_multi_x` | full Gaussian | 0.321 | 0.134 | 0.738 | 0.0 |
| `09_npe_multi_x_scaled` | MDN | 0.191 | 0.0966 | 0.387 | 0.0 |

The successful NPE flow runs are local-to-one-observation. They should not be treated as evidence that the learned map `x -> p(theta | x)` is faithful across a broad region.

### 3. Scaling Curves And Crossover

There are no current scaling curves that fairly compare MCMC/HMC and NPE at fixed discrepancy while varying:

- number of target observations `M`,
- parameter dimension `d_theta`,
- simulator or likelihood cost,
- posterior geometry difficulty,
- NPE training budget.

We have isolated runs across model families, but not enough controlled variation to estimate a crossover scale. The likely crossover condition is:

```text
NPE wins when:
  NPE_train_time + M * NPE_query_time
    <
  M * per_observation_MC_time
```

For the current toy decay setup, using the best NPE run and MCMC as baseline:

```text
M* approx 421.7 s / 1.47 s = 287 observations
```

That estimate is optimistic because the current broad amortized multi-x NPE does not meet the target. For HMC the rough crossover is even larger:

```text
M* approx 421.7 s / 6.67 s = 63 observations
```

but again only if the trained NPE remained faithful across those observations, which is not yet demonstrated.

### 4. MPS/GPU Evidence

Current evidence is mixed.

For random-walk MCMC on the decay toy model, MPS was worse for small chain counts and only became faster than CPU float64 at a highly batched 4096-chain workload:

| Chains | MPS median | CPU float64 median | MPS speedup vs CPU float64 |
| ---: | ---: | ---: | ---: |
| 8 | 4.12 s | 0.98 s | 0.24x |
| 64 | 0.78 s | 0.32 s | 0.41x |
| 512 | 0.82 s | 0.81 s | 0.99x |
| 4096 | 0.88 s | 1.55 s | 1.76x |

For trained NPE flow sampling, a small benchmark showed modest MPS speedups:

| Checkpoint | Samples | CPU median | MPS median | MPS speedup |
| --- | ---: | ---: | ---: | ---: |
| decay successful flow | 50,000 | 0.795 s | 0.570 s | 1.39x |
| banana legacy pairwise-pass flow | 50,000 | 0.313 s | 0.218 s | 1.43x |

For synthetic one-epoch flow training kernels with the saved architectures:

| Architecture | Items | CPU median | MPS median | MPS speedup |
| --- | ---: | ---: | ---: | ---: |
| decay flow | 65,536 | 1.160 s | 0.655 s | 1.77x |
| banana flow | 65,536 | 0.854 s | 0.504 s | 1.70x |

Conclusion: MPS can accelerate the neural flow kernels after warmup, but the repo does not yet contain an end-to-end CPU-vs-MPS NPE A/B run at fixed discrepancy. The calibrated sign stress NPE run and the legacy pairwise-pass stress NPE runs used MPS, while the successful decay NPE flow used CPU.

## Literature Context

The literature supports NPE in regimes that are different from the current toy experiments:

- Papamakarios and Murray, "Fast epsilon-free Inference of Simulation Models with Bayesian Conditional Density Estimation" (NeurIPS 2016), introduced SNPE for likelihood-free simulators and reported cases where learning a posterior representation used fewer simulations than Monte Carlo ABC needed for one approximate posterior sample: https://arxiv.org/abs/1605.06376
- Greenberg, Nonnenmacher, and Macke, "Automatic Posterior Transformation for Likelihood-free Inference" (ICML 2019), introduced APT/SNPE-C with arbitrary adaptive proposals and flow-compatible posterior estimators, emphasizing high-dimensional time series and image settings: https://arxiv.org/abs/1905.07488
- Lueckmann et al., "Benchmarking Simulation-Based Inference" (AISTATS 2021), benchmarked neural SBI and classical ABC methods across tasks and metrics. The useful lesson is not that NPE always wins, but that method performance is task-dependent and needs benchmarked diagnostics: https://proceedings.mlr.press/v130/lueckmann21a.html
- Deistler, Goncalves, and Macke, "Truncated proposals for scalable and hassle-free simulation-based inference" (NeurIPS 2022), emphasized that NPE provides a directly sampleable amortized posterior and reported robust performance on established benchmarks and challenging neuroscience models: https://arxiv.org/abs/2210.04815
- DINGO-style gravitational-wave inference is a strong applied example where NPE-like amortized inference is used to make repeated parameter estimation much faster than traditional event-wise inference workflows: https://link.aps.org/doi/10.1103/PhysRevLett.127.241103

The literature expectation is therefore:

NPE should win when the likelihood is unavailable or expensive, many observations must be analyzed, simulations are parallelizable, and the trained conditional density estimator is accurate over the requested observation region. It should not be expected to beat HMC on a tiny, differentiable, closed-form likelihood with three to seven parameters and cheap likelihood evaluations.

## Concrete Next Experiment

Build a controlled crossover benchmark, not another one-off posterior plot.

### Required Benchmark

Use one model where the true posterior can still be checked, but make the computational knobs explicit:

1. Start with a tractable "expensive-but-known" simulator/likelihood so HMC remains a valid reference.
2. Add a cost multiplier `C` to the simulator/likelihood, for example by evaluating a larger time grid, a basis expansion, or a small ODE solve.
3. Generate `M` target observations across a controlled prior region.
4. For each target observation:
   - run MCMC and HMC until R-hat/ESS and normalized Wasserstein to reference meet the target,
   - query a single amortized NPE trained once,
   - measure discrepancy and wall time.
5. Plot total wall time versus `M` at fixed discrepancy and report the crossover `M*`.
6. Repeat for increasing `d_theta` and increasing cost `C`.
7. Run NPE CPU and MPS end-to-end A/B tests with the same seeds, simulation budget, architecture, and target discrepancy.
8. Add SBC/coverage checks over the target region so "amortized" means calibrated, not just visually plausible.

### First Implementation Target

Implement this first for an expensive extension of the current exponential-decay family:

- keep a known likelihood and grid/HMC reference,
- vary `M in {1, 4, 16, 64, 256}`,
- vary cost multiplier `C in {1, 10, 100}`,
- compare MCMC, HMC, local NPE, and amortized NPE,
- require the same target discrepancy, initially 0.034.

This will directly answer whether NPE has a measured crossover in this repo. If it does not cross over even at large `M` and `C`, then the next target should be a genuinely likelihood-free simulator where MCMC/HMC are not applicable without an emulator or synthetic likelihood.
