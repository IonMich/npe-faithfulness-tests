# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| two_exp_ordered | True | True | 0.4487 | False | 0.0193 | 0.4377 | 0.4487 |

## two_exp_ordered

- Summary JSON: `runs/06_two_exponential/01_npe_flow/11_npe_flow_stress_tests_two_exp_ordered_proposal_i11/results/two_exp_ordered_summary.json`
- Corner overlay: `runs/06_two_exponential/01_npe_flow/11_npe_flow_stress_tests_two_exp_ordered_proposal_i11/figures/two_exp_ordered_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/06_two_exponential/01_npe_flow/11_npe_flow_stress_tests_two_exp_ordered_proposal_i11/figures/two_exp_ordered_trace.png`
- Predictive plot: `runs/06_two_exponential/01_npe_flow/11_npe_flow_stress_tests_two_exp_ordered_proposal_i11/figures/two_exp_ordered_predictive.png`
- Runtime seconds: MCMC 17.68, HMC 68.82, NPE train 207.78
- MCMC acceptance: 0.342
- HMC acceptance: 0.992
