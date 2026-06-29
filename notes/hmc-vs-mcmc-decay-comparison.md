# HMC vs random-walk MCMC for the decay model

Date: 2026-06-22

## Implemented scripts

- `scripts/mcmc_decay_inference.py`: random-walk Metropolis in log-parameter space.
- `scripts/hmc_decay_inference.py`: Hamiltonian Monte Carlo in log-parameter space.
- `scripts/compare_decay_samplers.py`: overlays posterior corners, overlays posterior
  predictive distributions, and compares both samplers against a numerical grid
  reference posterior.

## HMC tuning result

The first HMC defaults were too aggressive. The stable tuned defaults are:

- step size: `0.009`
- leapfrog steps: `10`
- chains: `8`
- steps: `5,000`
- burn-in: `1,000`

The tuned CPU HMC run had:

- acceptance rate: `0.992`
- divergences after burn-in: `0`
- maximum absolute energy error after burn-in: `0.329`
- all R-hat values below `1.01`
- all bulk/tail ESS values above `400`

## Posterior consistency

The posterior summaries from random-walk MCMC and HMC are nearly identical.

MCMC medians:

- `A`: `5.2914`
- `k`: `0.5712`
- `sigma`: `0.3405`

HMC medians:

- `A`: `5.2906`
- `k`: `0.5711`
- `sigma`: `0.3407`

Pairwise MCMC-vs-HMC sample distances:

| Parameter | KS statistic | Wasserstein | HMC median - MCMC median |
| --- | ---: | ---: | ---: |
| `A` | `0.00396` | `0.00162` | `-0.00088` |
| `k` | `0.00825` | `0.00044` | `-0.00010` |
| `sigma` | `0.00499` | `0.00027` | `0.00016` |

## Faithfulness probe

Because this example is only three-dimensional and has a tractable likelihood, the
comparison script builds a 90 x 90 x 90 grid over log-parameter space and uses the
grid posterior as an independent numerical reference.

Mean normalized Wasserstein distance to grid reference:

- random-walk MCMC: `0.03348`
- HMC: `0.03162`

Both are very close to the grid posterior; HMC is slightly closer on this metric,
but the difference is small.

## Performance

On the default small workload, MPS is slower than CPU for both samplers.

Random-walk MCMC low-overhead device benchmark:

- CPU float64 median sampler time: `0.825 s`
- MPS float32 median sampler time: `3.660 s`

HMC canonical run:

- CPU float64 sampler time: `6.667 s`
- MPS float32 sampler time: `22.376 s`

For raw time-to-converged-output, random-walk MCMC is faster on this small example.
For median bulk ESS/sec across parameters, HMC is competitive or better, but for
the weakest parameter (`sigma`) random-walk MCMC has higher ESS/sec in this tuned
run.

The main practical conclusion is that CPU random-walk MCMC remains the fastest
pedagogical baseline here, while HMC is a useful comparison because it reaches the
same posterior with gradient-informed proposals and excellent diagnostics.
