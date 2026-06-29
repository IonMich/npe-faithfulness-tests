# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| sign | True | True | 0.0414 | False | 0.0214 | 0.0365 | 0.0414 |

## sign

- Summary JSON: `runs/02_stress_sign/01_npe_flow/14_npe_flow_stress_tests_medium/results/sign_summary.json`
- Corner overlay: `runs/02_stress_sign/01_npe_flow/14_npe_flow_stress_tests_medium/figures/sign_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/02_stress_sign/01_npe_flow/14_npe_flow_stress_tests_medium/figures/sign_trace.png`
- Predictive plot: `runs/02_stress_sign/01_npe_flow/14_npe_flow_stress_tests_medium/figures/sign_predictive.png`
- Runtime seconds: MCMC 0.24, HMC 8.07, NPE train 12.50
- MCMC acceptance: 0.653
- HMC acceptance: 0.973

- Mode metrics: `{"mcmc": {"positive_theta1_fraction": 0.542, "mode_mass_error_vs_half": 0.04200000000000004}, "hmc": {"positive_theta1_fraction": 0.5040333333333333, "mode_mass_error_vs_half": 0.004033333333333333}, "npe": {"positive_theta1_fraction": 0.4853, "mode_mass_error_vs_half": 0.01469999999999999}}`
