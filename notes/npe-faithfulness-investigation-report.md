# NPE faithfulness investigation report

Goal: explain why finite-budget NPE/SNPE has not reached

```text
mean normalized Wasserstein <= 0.034
```

and identify what helps.

## Executive result

The target is strict but meaningful. The posterior density family is not fundamentally incapable. The dominant failure is learning the conditional map from simulated observations to posterior density with enough accuracy. In this toy problem, the remaining error is small visually but large relative to the MC-level target.

No pure likelihood-free NPE/SNPE fix reached `0.034` in these runs.

Best non-likelihood-corrected result remains about:

```text
0.0848 to 0.0878
```

which is about `2.5x` the strict target.

The only method that reached MC-level faithfulness earlier was exact-target correction using the analytic likelihood, which is not a pure likelihood-free NPE solution.

## 1. Target sanity check

Script:

```bash
uv run scripts/check_faithfulness_target.py
```

Results:

| Diagnostic | Mean normalized W |
| --- | ---: |
| Full MCMC to grid | 0.0335 |
| Full HMC to grid | 0.0316 |
| MCMC chain half 1 to grid | 0.0468 |
| MCMC chain half 2 to grid | 0.0347 |
| HMC chain half 1 to grid | 0.0323 |
| HMC chain half 2 to grid | 0.0329 |
| Grid sample to grid, median | 0.0060 |
| Grid sample pairwise, median | 0.0077 |

Interpretation:

- `0.034` is a strict target.
- It is not below the grid-sampling noise floor.
- It is roughly HMC-level and full-MCMC-level accuracy.
- One MCMC half is above target, so `0.034` should be treated as a strict MC-level target, not a loose visual-accuracy threshold.

## 2. Oracle posterior density fit

Script:

```bash
uv run scripts/oracle_posterior_density_fit.py
```

This removes the simulator/amortization problem. The model sees posterior samples directly at `x0` with a constant context.

| Model | Mean normalized W | Target ratio | Pass |
| --- | ---: | ---: | --- |
| Diagonal Gaussian | 0.0405 | 1.19x | no |
| Full Gaussian | 0.0388 | 1.14x | no |
| MDN | 0.0348 | 1.02x | no |
| Empirical diagonal Gaussian in `z` | 0.0405 | 1.19x | no |
| Empirical full Gaussian in `z` | 0.0399 | 1.17x | no |

Interpretation:

- The MDN can almost hit the target when given posterior samples directly.
- The posterior family is not the main limitation.
- The main limitation is learning `x -> p(theta | x)` from simulated data, not representing `p(theta | x0)` once posterior samples are available.

This also explains why the plots look close: the gap from `0.0348` to `0.034` is tiny, while simulation-trained NPE is stuck around `0.085+`.

## 3. Summary-context training

Script:

```bash
uv run scripts/npe_summary_context_decay.py
```

Context summary:

- 8 binned curve means.
- Rough noise from first differences.
- Rough global scale.
- Early-minus-late mean difference.

Results:

| Run | Context | Mean normalized W | Target ratio | Pass |
| --- | --- | ---: | ---: | --- |
| Broad MDN, 100k train | summary | 0.1149 | 3.38x | no |
| Hard local 0.5%, MDN | summary | 0.1038 | 3.05x | no |
| Hard local 0.5%, diagonal Gaussian | summary | 0.1397 | 4.11x | no |

Comparison:

- Scaled broad raw-`x` MDN at `x0`: `0.1564`.
- Broad summary MDN: `0.1149`.
- Best hard local raw-`x` MDN: `0.0867`.
- Hard local summary MDN: `0.1038`.

Interpretation:

- Summaries help broad amortization.
- Summaries do not beat raw curves in the local regime.
- The summaries probably discard small but target-relevant information once we are already focused near `x0`.

## 4. Kernel-weighted local training

Script:

```bash
uv run scripts/npe_summary_context_decay.py --mode kernel
```

Kernel:

```math
w(x) = \exp(-d(x,x_0)^2 / 2h^2)
```

with bandwidth set to the 0.5% prior-predictive summary-distance quantile.

| Run | Context | Train ESS fraction | Mean normalized W | Target ratio | Pass |
| --- | --- | ---: | ---: | ---: | --- |
| Kernel MDN, 200k train | summary | 0.091 | 0.1171 | 3.44x | no |
| Kernel MDN, 200k train | raw `x` | 0.091 | 0.2756 | 8.11x | no |

Interpretation:

- Smooth kernel weighting did not fix hard-region instability.
- Raw-context kernel weighting was much worse, likely because the model sees many broad-prior curves with tiny weights and optimization becomes less focused than hard local training.
- The best local result remains hard local raw-`x` MDN at `0.0867`.

## 5. Sequential SNPE

Script:

```bash
uv run scripts/snpe_sequential_decay.py
```

Best custom sequential results:

| Method | Best round | Mean normalized W | Target ratio | Pass |
| --- | ---: | ---: | ---: | --- |
| MDN, inflation 2.5 | 4 | 0.0878 | 2.58x | no |
| Diagonal Gaussian, inflation 2.5 | 2 | 0.0848 | 2.49x | no |
| Full Gaussian, inflation 2.5 | 3 | 0.1322 | 3.89x | no |

The correction ESS fractions were mostly healthy, so the miss is not primarily importance-weight collapse. It is residual conditional-density estimation error.

## 6. Alternate SBI methods

Script:

```bash
uv run scripts/sbi_alternate_decay.py
```

| Method | Simulations | Mean normalized W | Target ratio | Pass |
| --- | ---: | ---: | ---: | --- |
| sbi SNLE, MAF likelihood | 25k | 0.2534 | 7.45x | no |
| sbi SNRE, resnet classifier | 25k | 1.3371 | 39.33x | no |

Previous standard `sbi` SNPE-C checks:

- `sbi` MDN became numerically unstable in round 3.
- `sbi` MAF completed but stayed around `0.43`.

Interpretation:

- Off-the-shelf alternate SBI did not fix this at the tested budgets.
- SNLE has to model a 40D observation likelihood under broad prior simulation, which is hard.
- SNRE was much worse in this configuration.

## What Was Fixed

No pure NPE/SNPE run was fixed all the way to `0.034`.

What did improve:

- Summary context improved broad MDN from `0.1564` to `0.1149`.
- Local focusing improved from broad values to about `0.0867`.
- Sequential proposal focusing reached about `0.0878`.

What did not improve:

- Narrowing the hard local region beyond 0.5%.
- Increasing MDN size/data at the 0.5% local region.
- Kernel-weighted local training.
- Off-the-shelf `sbi` SNPE/SNLE/SNRE at the tested budgets.

## Current diagnosis

The most defensible explanation is:

```text
The neural density family can almost represent the posterior at x0.
The strict target is fair but very tight.
The main remaining error is conditional density estimation bias:
learning x -> p(theta | x) from simulated observations is not accurate enough at finite budget.
```

This is why posteriors look close in corner plots but fail the `0.034` target.

## Recommended next path

For reliable inference on this tractable example:

1. Use HMC/MCMC/grid as the faithful reference.
2. Use NPE/SNPE as a fast approximate surrogate, not as MC-faithful yet.
3. If MC-level NPE is required, the next experiments should target the conditional-learning bias directly:
   - Train ensembles and average posterior densities.
   - Use simulation-based calibration over a local benchmark panel, not one `x0`.
   - Try larger/residual context networks and lower learning rates.
   - Train directly on posterior samples when available, or use exact-target correction when the likelihood is tractable.

In this toy problem, exact-target correction is the only tested path that reaches MC-level faithfulness. In genuinely likelihood-free problems, the closest practical path so far is local/sequential NPE, but it remains approximate at the tested budgets.
