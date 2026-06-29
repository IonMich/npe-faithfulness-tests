# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| two_exp_ordered | True | True | 0.1699 | False | 0.0105 | 0.1675 | 0.1699 |

## two_exp_ordered

- Summary JSON: `runs/06_two_exponential/01_npe_flow/16_npe_flow_stress_tests_two_exp_ordered_ridgecoords/results/two_exp_ordered_summary.json`
- Corner overlay: `runs/06_two_exponential/01_npe_flow/16_npe_flow_stress_tests_two_exp_ordered_ridgecoords/figures/two_exp_ordered_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/06_two_exponential/01_npe_flow/16_npe_flow_stress_tests_two_exp_ordered_ridgecoords/figures/two_exp_ordered_trace.png`
- Predictive plot: `runs/06_two_exponential/01_npe_flow/16_npe_flow_stress_tests_two_exp_ordered_ridgecoords/figures/two_exp_ordered_predictive.png`
- Runtime seconds: MCMC 10.59, HMC 48.34, NPE train 165.79
- MCMC acceptance: 0.115
- HMC acceptance: 0.939
