# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| sign | True | True | 0.0269 | True | 0.0186 | 0.0202 | 0.0269 |

## sign

- Summary JSON: `runs/02_stress_sign/01_npe_flow/15_npe_flow_stress_tests_sign_nolinear/results/sign_summary.json`
- Corner overlay: `runs/02_stress_sign/01_npe_flow/15_npe_flow_stress_tests_sign_nolinear/figures/sign_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/02_stress_sign/01_npe_flow/15_npe_flow_stress_tests_sign_nolinear/figures/sign_trace.png`
- Predictive plot: `runs/02_stress_sign/01_npe_flow/15_npe_flow_stress_tests_sign_nolinear/figures/sign_predictive.png`
- Runtime seconds: MCMC 0.36, HMC 11.46, NPE train 78.06
- MCMC acceptance: 0.651
- HMC acceptance: 0.973

- Mode metrics: `{"mcmc": {"positive_theta1_fraction": 0.5603409090909091, "mode_mass_error_vs_half": 0.060340909090909056}, "hmc": {"positive_theta1_fraction": 0.5036875, "mode_mass_error_vs_half": 0.0036874999999999547}, "npe": {"positive_theta1_fraction": 0.5111, "mode_mass_error_vs_half": 0.011099999999999999}}`
