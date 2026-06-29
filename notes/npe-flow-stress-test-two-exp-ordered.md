# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| two_exp_ordered | False | True | 0.4146 | False | 0.0537 | 0.3939 | 0.4146 |

## two_exp_ordered

- Summary JSON: `runs/06_two_exponential/01_npe_flow/01_npe_flow_stress_tests_two_exp_ordered/results/two_exp_ordered_summary.json`
- Corner overlay: `runs/06_two_exponential/01_npe_flow/01_npe_flow_stress_tests_two_exp_ordered/figures/two_exp_ordered_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/06_two_exponential/01_npe_flow/01_npe_flow_stress_tests_two_exp_ordered/figures/two_exp_ordered_trace.png`
- Predictive plot: `runs/06_two_exponential/01_npe_flow/01_npe_flow_stress_tests_two_exp_ordered/figures/two_exp_ordered_predictive.png`
- Runtime seconds: MCMC 6.93, HMC 82.49, NPE train 335.99
- MCMC acceptance: 0.343
- HMC acceptance: 0.992
