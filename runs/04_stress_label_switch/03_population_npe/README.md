# 03_population_npe

Population NPE runs for the Label Switching stress model. These runs use
full-prior simulations and report NLL in sorted target coordinates:

```text
z_sorted = (mu_low, mu_high, log_sigma)
mu_low = min(mu_1, mu_2)
mu_high = max(mu_1, mu_2)
```

| Status | Run | Metric | Target | Reason |
| --- | --- | --- | --- | --- |
| `near_floor` | [02_flow2_residual_full_prior_512k_ensemble4_e30](02_flow2_residual_full_prior_512k_ensemble4_e30) | full-prior sorted z-NLL: `-3.09250 +/- 0.00822` | entropy floor: `-3.10112 +/- 0.00821` | gap `0.00862`, combined z `0.74`; paired cache resolves `0.00862 +/- 0.00060` |

The floor is estimated in the same sorted coordinates. For each signal, raw
evidence is estimated by symmetric Gaussian-mixture importance sampling over
the two label permutations, and the sorted posterior includes the `log 2` fold
factor.
