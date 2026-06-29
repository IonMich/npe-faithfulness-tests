# Local NPE Scaling-Law Evidence

## Question

Does the single-decay local NPE show a concrete scaling law with accepted local
training simulations?

## Controlled Pilot

Command:

```bash
uv run scripts/decay_local_scaling_sweep.py \
  --preset pilot \
  --output-root runs/01_exponential_decay/12_local_scaling/01_controlled_pilot \
  --train-simulations 2500,5000,10000,20000 \
  --skip-existing
```

Design controls:

- one fixed observed signal;
- one fixed grid/MCMC/HMC reference;
- one fixed local region;
- one fixed architecture and optimizer;
- one nested accepted local-data pool per seed;
- one shared validation suffix per seed;
- three replicate seeds.

Outputs:

- `runs/01_exponential_decay/12_local_scaling/01_controlled_pilot/results/local_data_scaling_summary.json`
- `runs/01_exponential_decay/12_local_scaling/01_controlled_pilot/results/local_data_scaling_rows.csv`
- `runs/01_exponential_decay/12_local_scaling/01_controlled_pilot/figures/local_data_scaling.png`

## Controlled Result

Median mean-normalized Wasserstein decreased as accepted local training data
increased:

| Accepted train simulations | Median W | Mean W | q16 | q84 | Median target ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,500 | 0.0784 | 0.0877 | 0.0720 | 0.1037 | 1.89 |
| 5,000 | 0.0732 | 0.0826 | 0.0678 | 0.0977 | 1.77 |
| 10,000 | 0.0706 | 0.0679 | 0.0614 | 0.0744 | 1.70 |
| 20,000 | 0.0474 | 0.0511 | 0.0461 | 0.0562 | 1.14 |

Best validation NLL in target-z units also improved monotonically:

| Accepted train simulations | Median best validation NLL |
| ---: | ---: |
| 2,500 | -4.3778 |
| 5,000 | -4.4112 |
| 10,000 | -4.4207 |
| 20,000 | -4.4350 |

Each individual seed improved monotonically in Wasserstein and validation NLL.
The per-seed zero-floor log-log exponents were approximately:

| Seed | Wasserstein exponent | R2 |
| ---: | ---: | ---: |
| 20260701 | 0.334 | 0.940 |
| 20260702 | 0.223 | 0.774 |
| 20260703 | 0.200 | 0.929 |

The median curve fit

```text
W(S) = W_inf + A * S^(-alpha)
```

returned:

```text
W_inf ~= 0
A     ~= 0.373
alpha ~= 0.194
R2    ~= 0.766
```

The near-zero floor is not interpretable from this pilot. Four small scale
points cannot identify an asymptotic floor.

## Historical Extension Check

Historical non-enhanced hard-local `q=0.005`, linear-adjusted runs are less
controlled but extend the data scale:

| Accepted train simulations | Median W | Min W | Max W | Runs |
| ---: | ---: | ---: | ---: | ---: |
| 40,000 | 0.0387 | 0.0387 | 0.0387 | 1 |
| 100,000 | 0.0366 | 0.0347 | 0.0383 | 3 |
| 150,000 | 0.0331 | 0.0331 | 0.0331 | 1 |

The historical-only fit gives a shallow exponent:

```text
alpha ~= 0.105
R2    ~= 0.858
```

Combining the controlled pilot medians with these historical medians gives:

```text
alpha ~= 0.227
R2    ~= 0.922
```

This combined fit is suggestive only because the historical runs changed
architecture and were not nested.

## Full Controlled Sweep

Command:

```bash
uv run scripts/decay_local_scaling_sweep.py \
  --preset full \
  --output-root runs/01_exponential_decay/12_local_scaling/02_full_large_controlled \
  --train-simulations 40000,80000,150000,300000 \
  --skip-existing
```

Design controls:

- one fixed observed signal;
- one fixed grid/MCMC/HMC reference;
- one fixed local region per seed;
- one fixed architecture and optimizer;
- one nested accepted local-data pool per seed;
- one shared validation suffix per seed;
- five replicate seeds.

Outputs:

- `runs/01_exponential_decay/12_local_scaling/02_full_large_controlled/results/local_data_scaling_summary.json`
- `runs/01_exponential_decay/12_local_scaling/02_full_large_controlled/results/local_data_scaling_summary.csv`
- `runs/01_exponential_decay/12_local_scaling/02_full_large_controlled/results/local_data_scaling_rows.csv`
- `runs/01_exponential_decay/12_local_scaling/02_full_large_controlled/figures/local_data_scaling.png`

Aggregate posterior Wasserstein improved from 40k to 150k accepted local
training simulations, but did not improve at 300k:

| Accepted train simulations | Median W | Mean W | Min W | Max W | Median target ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40,000 | 0.0446 | 0.0442 | 0.0384 | 0.0479 | 1.33 |
| 80,000 | 0.0408 | 0.0427 | 0.0364 | 0.0560 | 1.22 |
| 150,000 | 0.0350 | 0.0376 | 0.0344 | 0.0477 | 1.05 |
| 300,000 | 0.0367 | 0.0377 | 0.0352 | 0.0404 | 1.10 |

The median validation NLL continued to improve over the same scale range:

| Accepted train simulations | Median best validation NLL |
| ---: | ---: |
| 40,000 | -4.4495 |
| 80,000 | -4.4530 |
| 150,000 | -4.4574 |
| 300,000 | -4.4596 |

The four-point median Wasserstein curve fit returned:

```text
W_inf ~= 0.03449
A     ~= 655.36
alpha ~= 1.04
R2    ~= 0.873
```

The exponent is not interpretable as a stable law. The 300k point rises above
the 150k median, and the fitted floor is close to the 150k/300k measurement
range. The meaningful result is the floor/plateau: with this architecture,
local region, training recipe, and reference metric, posterior Wasserstein
appears to saturate around `0.035-0.037` even while validation NLL keeps
improving.

## Conclusion

There is concrete evidence of a low/mid-data scaling relationship in this
repo's single-decay local NPE:

- more accepted local simulator pairs consistently reduce posterior
  Wasserstein in the controlled pilot;
- validation NLL improves at the same time;
- all three pilot seeds are monotone over 2.5k to 20k accepted local pairs;
- the full controlled sweep continues the median Wasserstein improvement from
  40k to 150k accepted local pairs.

It is not defensible to claim a clean global scaling law for posterior
Wasserstein under the current setup. The full controlled sweep shows a plateau
or noise floor after roughly 150k accepted local pairs: 300k improves
validation NLL but not Wasserstein.

Follow-up metric probing in `notes/npe-wasserstein-noise-floor-investigation.md`
shows that the absolute grid-90 Wasserstein floor is mostly a grid/reference
discretization artifact rather than finite NPE posterior sampling noise.

The current evidence supports this narrower statement:

```text
Single-decay local NPE shows power-law-like data scaling before saturation, but
posterior Wasserstein reaches an apparent floor around 0.035-0.037 normalized W
for the current architecture/training/reference setup.
```

## Next Checks

The next useful checks are no longer "add more data" alone. They should test
what causes the plateau:

- repeat 150k/300k with a larger model or longer training to separate data
  limits from capacity/optimization limits;
- evaluate more posterior samples and/or repeated posterior sampling to bound
  metric Monte Carlo noise;
- compare validation NLL ranking against Wasserstein ranking to see whether the
  training objective is misaligned with posterior faithfulness near the floor;
- test ensembles or best-of-seed selection if the intended deployment can use
  repeated training runs.
