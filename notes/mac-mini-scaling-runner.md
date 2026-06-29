# Mac Mini Scaling Runner

Date: 2026-06-29

## Machine

The home-infrastructure repo documents the Mac mini SSH aliases as `mini`,
`macmini`, and `macmini-ts`; the Tailnet IP is `100.126.112.7`.

Observed from this workspace:

- `ssh mini` works.
- `/Users/ioannism/repos/npe` was not present on the Mac mini.
- Git is installed.
- GitHub SSH access from the Mac mini failed with `Permission denied (publickey)`.
- GitHub HTTPS read access works for this repo.
- `uv` was not found in the remote PATH.

Because of that, the launcher defaults to cloning over HTTPS and has an
explicit `--install-uv` flag. No GitHub credential setup is needed unless the
repo access policy changes.

## Launch

After pushing the branch, start the default broad MDN scaling run on the mini:

```bash
uv run scripts/run_scaling_on_mac_mini.py \
  --branch codex/mac-mini-scaling-runner \
  --install-uv
```

The default remote run:

- clones or updates `/Users/ioannism/repos/npe`;
- runs `uv sync`;
- creates the 1M validation-NLL cache if missing;
- creates the 16-signal panel marginal cache if missing;
- launches `scripts/decay_broad_scaling_sweep.py` detached;
- uses `--jobs 2`, `--torch-threads 2`, and `--eval-batch-size 16384`;
- trains `D=64k,128k,256k,512k,1M` for seeds
  `20260901,20260902,20260903`;
- writes logs under `/Users/ioannism/repos/npe/logs/macmini_scaling/`.

Use `--dry-run` to inspect the exact remote shell script before running it:

```bash
uv run scripts/run_scaling_on_mac_mini.py \
  --branch codex/mac-mini-scaling-runner \
  --install-uv \
  --dry-run
```

For a smaller timing test:

```bash
uv run scripts/run_scaling_on_mac_mini.py \
  --branch codex/mac-mini-scaling-runner \
  --install-uv \
  --train-simulations 64000,128000 \
  --seeds 20260901,20260902
```

## Metric Scope

The Mac mini launcher passes `--skip-x0-reference`. This avoids requiring the
large local `300^3` x0 grid cache on a fresh checkout. The scaling plots still
report the two metrics that matter for the broad scaling law investigation:

- panel mean marginal Wasserstein over a fixed prior-predictive signal panel;
- cached broad prior-predictive NLL.

If an x0-specific W audit is needed on the mini, transfer or regenerate the x0
grid cache and remove `--skip-x0-reference` from the launched command.

## Parallelization Design

`scripts/decay_broad_scaling_sweep.py --jobs N` parallelizes by seed, not by
individual `(seed, D)` cell. Each worker keeps the existing serial progression
over `D` for its seed, so the per-seed training pool remains nested and the
data-axis comparison stays controlled. The parent process aggregates once after
all workers finish.
