# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| two_exp_ordered | True | True | 0.3506 | False | 0.0105 | 0.3506 | 0.3503 |

## two_exp_ordered

- Summary JSON: `runs/06_two_exponential/01_npe_flow/17_npe_flow_stress_tests_two_exp_ordered_ridgecoords_proposal_all/results/two_exp_ordered_summary.json`
- Corner overlay: `runs/06_two_exponential/01_npe_flow/17_npe_flow_stress_tests_two_exp_ordered_ridgecoords_proposal_all/figures/two_exp_ordered_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/06_two_exponential/01_npe_flow/17_npe_flow_stress_tests_two_exp_ordered_ridgecoords_proposal_all/figures/two_exp_ordered_trace.png`
- Predictive plot: `runs/06_two_exponential/01_npe_flow/17_npe_flow_stress_tests_two_exp_ordered_ridgecoords_proposal_all/figures/two_exp_ordered_predictive.png`
- Runtime seconds: MCMC 10.72, HMC 50.22, NPE train 372.32
- MCMC acceptance: 0.115
- HMC acceptance: 0.939
