# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| two_exp_ordered | True | True | 0.0532 | False | 0.0193 | 0.0429 | 0.0532 |

## two_exp_ordered

- Summary JSON: `runs/06_two_exponential/01_npe_flow/12_npe_flow_stress_tests_two_exp_ordered_residual/results/two_exp_ordered_summary.json`
- Corner overlay: `runs/06_two_exponential/01_npe_flow/12_npe_flow_stress_tests_two_exp_ordered_residual/figures/two_exp_ordered_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/06_two_exponential/01_npe_flow/12_npe_flow_stress_tests_two_exp_ordered_residual/figures/two_exp_ordered_trace.png`
- Predictive plot: `runs/06_two_exponential/01_npe_flow/12_npe_flow_stress_tests_two_exp_ordered_residual/figures/two_exp_ordered_predictive.png`
- Runtime seconds: MCMC 15.31, HMC 50.31, NPE train 225.45
- MCMC acceptance: 0.342
- HMC acceptance: 0.992
