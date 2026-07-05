# Banana / 03_population_npe

Full-prior Banana population NPE runs. These runs train on `p(theta)p(x|theta)`
and evaluate raw-coordinate target NLL for `theta=(theta1, theta2)` against the
full-prior entropy floor.

| Status | Run | Metric | Target | Reason |
| --- | --- | --- | --- | --- |
| `floor_pass` | [01_flow2_residual_full_prior_512k_ensemble4](01_flow2_residual_full_prior_512k_ensemble4) | ensemble NLL: `-0.52753 +/- 0.00100` | entropy floor: `-0.52826 +/- 0.00100` | Gap is `0.00073`, or `0.52` combined SE, on 1M full-prior validation examples. |
