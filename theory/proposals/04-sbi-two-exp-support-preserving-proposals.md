# Proposal 04: Make `sbi_two_exp_ordered.py` Proposals Support-Preserving

## Claim

The `sbi` two-exponential script should avoid pure HMC-Gaussian proposal rounds
without a prior-mixture or prior first round. Sequential NPE needs proposal
support where the true posterior has mass.

## Literature Signal

- APT/SNPE-C handles arbitrary sequential proposals, but support still matters:
  https://arxiv.org/abs/1905.07488
- TSNPE is explicitly motivated by robust sequential proposals:
  https://arxiv.org/abs/2210.04815
- The `sbi` sequential guide starts from the prior and then adapts proposals:
  https://sbi.readthedocs.io/en/latest/how_to_guide/02_multiround_inference.html

## Current Code Touchpoints

In `scripts/sbi_two_exp_ordered.py`:

- `--initial-proposal hmc_gaussian` fits a single inflated Gaussian from HMC
  samples.
- Later rounds sample directly from `posterior`.
- There is no prior-mixture component in the HMC-Gaussian initial proposal.

## Implementation Sketch

Option A: require prior first round.

```text
--initial-proposal prior
--rounds N
```

Then allow a flag:

```text
--second-round-proposal hmc_gaussian
```

Option B: implement a mixture distribution:

```text
r(theta) = alpha * p(theta) + (1 - alpha) * N(theta; mean_hmc, inflated_cov_hmc)
```

Add CLI options:

```text
--proposal-prior-mixture 0.02
--proposal-min-inflation 1.5
```

For `sbi`, a custom `torch.distributions.Distribution` wrapper can implement
`sample()` and `log_prob()`. It must be passed to `append_simulations(...,
proposal=proposal)` when it is not the prior.

## Acceptance Criteria

- `hmc_gaussian` proposal mode preserves a prior-mixture support component.
- Summary records proposal kind, inflation, prior-mixture weight, mean, and
  covariance.
- Each round summary reports proposal type.
- Existing pure-prior behavior remains unchanged.
- If custom mixture `log_prob` is not robust enough for `sbi`, the script should
  refuse `hmc_gaussian` without a prior first round.

## First Target

Rerun the two-exponential raw NSF `sbi` experiment with either:

```text
round 1: prior
rounds 2-N: posterior proposal
```

or:

```text
round 1: prior/HMC Gaussian mixture
rounds 2-N: posterior proposal
```

Compare against the existing broad-prior and pure-HMC-Gaussian runs.
