# Exponential Decay / 06_oracle_target_checks

| Status | Run | Metric | Target | Reason |
| --- | --- | --- | --- | --- |
| `diagnostic_pass` | [01_faithfulness_target_check](01_faithfulness_target_check) | best mean normalized Wasserstein: 0.03162 | 0.034 | diagnostic/reference metric |
| `near` | [02_oracle_posterior_fit](02_oracle_posterior_fit) | best discovered target metric: 0.03481 | 0.034 | nested target flags were false |
