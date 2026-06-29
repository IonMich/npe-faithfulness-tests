# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| two_exp_ordered | True | True | 0.1445 | False | 0.0193 | 0.1374 | 0.1445 |

## two_exp_ordered

- Summary JSON: `runs/06_two_exponential/01_npe_flow/04_npe_flow_stress_tests_two_exp_ordered_moderate_ridge_rawctx/results/two_exp_ordered_summary.json`
- Corner overlay: `runs/06_two_exponential/01_npe_flow/04_npe_flow_stress_tests_two_exp_ordered_moderate_ridge_rawctx/figures/two_exp_ordered_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/06_two_exponential/01_npe_flow/04_npe_flow_stress_tests_two_exp_ordered_moderate_ridge_rawctx/figures/two_exp_ordered_trace.png`
- Predictive plot: `runs/06_two_exponential/01_npe_flow/04_npe_flow_stress_tests_two_exp_ordered_moderate_ridge_rawctx/figures/two_exp_ordered_predictive.png`
- Runtime seconds: MCMC 16.50, HMC 55.04, NPE train 243.60
- MCMC acceptance: 0.342
- HMC acceptance: 0.992
