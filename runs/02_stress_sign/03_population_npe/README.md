# 03_population_npe

Full-prior sign-symmetry population NPE runs. These runs train on
prior-predictive pairs from \(p(\theta)p(x\mid\theta)\) and evaluate folded
target NLL for \((|\theta_1|,\theta_2)\) against the full-prior entropy floor.

| Status | Run | Metric | Target | Reason |
| --- | --- | ---: | ---: | --- |
| `near_floor` | [01_flow2_residual_full_prior_512k_ensemble4](01_flow2_residual_full_prior_512k_ensemble4) | ensemble NLL: `-1.42261 +/- 0.00117` | folded floor: `-1.42694 +/- 0.00115` | Gap is `0.00433`, about `2.64` combined standard errors, on 1M full-prior validation examples. |

