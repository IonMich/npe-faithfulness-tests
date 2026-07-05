# Linear6 / 03_population_npe

Full-prior Linear6 population NPE runs. These runs train on
`p(z)p(x|z)` and evaluate target NLL for
`z=(w1, ..., w6, log_sigma)` against the full-prior entropy floor.

| Status | Run | Metric | Target | Reason |
| --- | --- | --- | --- | --- |
| `near_floor` | [01_flow2_residual_full_prior_512k_ensemble4](01_flow2_residual_full_prior_512k_ensemble4) | ensemble NLL: `-10.77984 +/- 0.00353` | entropy floor: `-10.78631 +/- 0.00353` | Gap is `0.00647` in z units on 1M full-prior validation examples. |
