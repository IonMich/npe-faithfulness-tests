# Remote Train Endpoint

Date: 2026-06-29

## Machine

The home-infrastructure repo documents the Mac mini SSH aliases as `mini`,
`macmini`, and `macmini-ts`; the Tailnet IP is `100.126.112.7`.

Observed from this workspace:

- `ssh mini` works.
- `/Users/ioannism/repos/npe` was not present on the Mac mini before setup.
- Git is installed.
- GitHub HTTPS read access works for this repo.
- `uv` is installed at `/Users/ioannism/.local/bin/uv`, but non-interactive SSH
  sessions did not include that directory in `PATH`.

## Server Model

The Mac mini should run a generic train endpoint, not a scaling-law endpoint.
The endpoint accepts a repo-local training command at `POST /train`:

```json
{
  "run_name": "broad_mdn_1m",
  "branch": "codex/mac-mini-scaling-runner",
  "sync": true,
  "setup_commands": [
    {
      "name": "validation_cache",
      "skip_if_exists": "runs/.../broad_prior_val_1m_float32.npz",
      "command": ["uv", "run", "scripts/cache_decay_broad_validation.py", "..."]
    }
  ],
  "command": ["uv", "run", "scripts/decay_broad_scaling_sweep.py", "..."],
  "output_root": "runs/..."
}
```

The scaling logic stays local in `scripts/submit_remote_broad_scaling.py`.
The server only updates the repo, runs optional setup commands, and launches the
requested train command.

## Start The Endpoint

Initial one-time setup on the Mac mini:

```bash
mkdir -p /Users/ioannism/repos
git clone https://github.com/IonMich/npe-faithfulness-tests.git /Users/ioannism/repos/npe
cd /Users/ioannism/repos/npe
git checkout codex/mac-mini-scaling-runner
```

Start the endpoint on the Mac mini:

```bash
cd /Users/ioannism/repos/npe
PATH="$HOME/.local/bin:$PATH" \
uv run scripts/train_remote_server.py serve \
  --host 127.0.0.1 \
  --port 8765 \
  --uv "$HOME/.local/bin/uv"
```

For long-running use, start it inside `tmux`, `screen`, or a launchd service.

## Submit A Broad Scaling Train Request

If the endpoint is bound to localhost on the Mac mini, open a tunnel from the
local machine:

```bash
ssh -N -L 8765:127.0.0.1:8765 mini
```

Then submit the broad scaling request from this repo:

```bash
uv run scripts/submit_remote_broad_scaling.py \
  --endpoint http://127.0.0.1:8765 \
  --branch codex/mac-mini-scaling-runner
```

The default request:

- prepares the 1M validation-NLL cache if missing;
- prepares the 16-signal panel marginal cache if missing;
- launches `scripts/decay_broad_scaling_sweep.py`;
- uses `--jobs 2`, `--torch-threads 2`, and `--eval-batch-size 16384`;
- trains `D=64k,128k,256k,512k,1M` for seeds
  `20260901,20260902,20260903`;
- passes `--skip-x0-reference`, so the large local `300^3` x0 grid cache is
  not required on the mini.

Dry-run the train request JSON without submitting:

```bash
uv run scripts/submit_remote_broad_scaling.py \
  --branch codex/mac-mini-scaling-runner \
  --dry-run
```

For a smaller timing test:

```bash
uv run scripts/submit_remote_broad_scaling.py \
  --endpoint http://127.0.0.1:8765 \
  --branch codex/mac-mini-scaling-runner \
  --train-simulations 64000,128000 \
  --seeds 20260901,20260902
```

## Inspect Runs

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/train
curl http://127.0.0.1:8765/train/<run_id>
curl 'http://127.0.0.1:8765/train/<run_id>/log?tail=120'
```

Remote logs live under:

```text
/Users/ioannism/repos/npe/logs/train_remote/<run_id>/
```

## Parallelization Design

`scripts/decay_broad_scaling_sweep.py --jobs N` parallelizes by seed, not by
individual `(seed, D)` cell. Each worker keeps the existing serial progression
over `D` for its seed, so the per-seed training pool remains nested and the
data-axis comparison stays controlled. The parent process aggregates once after
all workers finish.
