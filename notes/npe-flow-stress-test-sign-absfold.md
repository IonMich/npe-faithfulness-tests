# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| sign | True | True | 0.0534 | False | 0.0209 | 0.0534 | 0.0342 |

## sign

- Summary JSON: `runs/02_stress_sign/01_npe_flow/19_npe_flow_stress_tests_sign_absfold/results/sign_summary.json`
- Corner overlay: `runs/02_stress_sign/01_npe_flow/19_npe_flow_stress_tests_sign_absfold/figures/sign_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/02_stress_sign/01_npe_flow/19_npe_flow_stress_tests_sign_absfold/figures/sign_trace.png`
- Predictive plot: `runs/02_stress_sign/01_npe_flow/19_npe_flow_stress_tests_sign_absfold/figures/sign_predictive.png`
- Runtime seconds: MCMC 0.32, HMC 7.48, NPE train 175.58
- MCMC acceptance: 0.651
- HMC acceptance: 0.973

- Mode metrics: `{"mcmc": {"positive_theta1_fraction": 0.5618, "mode_mass_error_vs_half": 0.061799999999999966}, "hmc": {"positive_theta1_fraction": 0.5038690476190476, "mode_mass_error_vs_half": 0.0038690476190476053}, "npe": {"positive_theta1_fraction": 0.5003, "mode_mass_error_vs_half": 0.00029999999999996696}}`
