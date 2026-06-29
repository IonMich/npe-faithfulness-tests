# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| two_exp_ordered | True | True | 0.2234 | False | 0.0367 | 0.1927 | 0.2234 |

## two_exp_ordered

- Summary JSON: `runs/06_two_exponential/01_npe_flow/05_npe_flow_stress_tests_two_exp_ordered_profile/results/two_exp_ordered_summary.json`
- Corner overlay: `runs/06_two_exponential/01_npe_flow/05_npe_flow_stress_tests_two_exp_ordered_profile/figures/two_exp_ordered_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/06_two_exponential/01_npe_flow/05_npe_flow_stress_tests_two_exp_ordered_profile/figures/two_exp_ordered_trace.png`
- Predictive plot: `runs/06_two_exponential/01_npe_flow/05_npe_flow_stress_tests_two_exp_ordered_profile/figures/two_exp_ordered_predictive.png`
- Runtime seconds: MCMC 12.76, HMC 71.41, NPE train 296.79
- MCMC acceptance: 0.342
- HMC acceptance: 0.992
