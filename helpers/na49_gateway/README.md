# NA49 Cluster Gateway

SSH-native helpers for dispatching NA49 workloads to remote worker nodes
without opening network ports or running a persistent service.

System-specific settings are intentionally kept out of the repository. The
gateway reads user configuration from:

- `$NA49_CLUSTER_GW_CONFIG`, if set
- otherwise `~/.config/na49_cluster_gw/config.json`

A template is provided in `helpers/na49_gateway/config.example.json`.
For this project, `helpers/na49_gateway/config.na49_project.json` is the
pinned baseline with sanitized example values (`max_tasks_per_worker=64`).
The real system-specific settings (login host, worker hostnames) stay out
of the repository in the per-user config file.

## Files

- `cluster_agent.py`: runs on the login node and launches jobs on workers
- `cluster_client.py`: local/client-side CLI and Python API
- `gateway_config.py`: shared user-config loader
- `config.example.json`: non-site-specific config template

## Configuration

Typical config keys:

- `host`: SSH login host used by the client
- `workers`: worker-node hostnames used by the agent
- `max_tasks_per_worker`: task-slot capacity per worker
- `worker_bin_dir`: optional bin directory prepended to worker `PATH`
- `jobs_root`: shared directory for job metadata and logs
- `remote_project_relpath`: path to this repo under the remote user home directory
- `agent_command`: optional explicit override for the remote agent command
- `ssh_options`: optional extra SSH arguments

`agent_command` and `remote_project_relpath` are alternatives. If `agent_command`
is omitted, the client uses `remote_project_relpath` when configured, otherwise it
falls back to a path derived from the local repo location relative to the local
home directory.

## Example setup

```bash
mkdir -p ~/.config/na49_cluster_gw
cp helpers/na49_gateway/config.na49_project.json ~/.config/na49_cluster_gw/config.json
```

Then edit `~/.config/na49_cluster_gw/config.json` with site-specific values.

## Task accounting

Each submitted job reserves `--tasks` slots on a worker. Placement only occurs
when:

```text
active_reserved_tasks + requested_tasks <= max_tasks_per_worker
```

Use `--tasks` to match the parallelism of the submitted command.

The worker wrapper exports `NA49_CLUSTER_GW_TASKS` and defaults common
math-library thread counts to `1` so many independent jobs do not silently
oversubscribe a node.

## Stale-job recovery (LOST state)

Job records live in a shared filesystem, so a worker that dies mid-run cannot
always write its own completion markers. Without recovery such records stay
`RUNNING` forever and permanently consume task capacity. The gateway closes
this gap with a worker PID record, a heartbeat, and a terminal `LOST` state:

- The worker wrapper writes its own PID to `worker_pid` and refreshes
  `last_seen` (POSIX epoch seconds) every
  `NA49_CLUSTER_GW_HEARTBEAT_SECONDS` seconds while it is alive.
- `_compute_state()` reports `RUNNING` only when nothing proves otherwise:
  exit markers decide terminal states first (`COMPLETED`, `FAILED`,
  `CANCELED`, `SUBMIT_FAILED`); without them, a confirmed-dead PID or an
  unfresh heartbeat yields the terminal state `LOST`.
- An unfresh heartbeat means either a stale `last_seen` value (older than
  `stale_heartbeat_seconds`, default 300) or no heartbeat at all beyond the
  launch grace window (`launch_grace_seconds`, default 900 after
  `launched_at`/`submitted_at`). The grace window covers wrapper startup
  latency and submits killed between reservation and launch; those records
  stop reserving slots once the window passes.
- PID liveness is verified over SSH by `reconcile`
  (`ssh <worker> kill -0 <pid>`, dialing the submit alias recorded in
  meta `worker`). The wrapper-written `hostname` marker is only consulted
  for local PID-namespace checks when it equals the current host. Plain
  `status`/`list`/`capacity` calls never open SSH connections; they rely
  on heartbeat files alone, which frees stale capacity automatically. An
  inconclusive SSH probe (transport failure) never marks a job `LOST` by
  itself.
- All terminal states, including `LOST`, are excluded from task-capacity
  accounting on both the agent and the client, and `wait()` returns once a
  job becomes `LOST`.

`meta.json` files are written atomically (temporary file plus rename), so
readers never observe partially written metadata.

Job records are owner-only. The jobs root and each job directory are
mode ``0700``; ``meta.json``, ``command.sh``, and the submit lock are
mode ``0600``. The worker wrapper already uses ``umask 077`` for logs
and heartbeat markers. ``meta.json`` and ``command.sh`` store the full
submitted command, so pass tokens, passwords, and database URLs through
protected files or the worker environment rather than putting them on
the command line.

### Reconcile

```bash
python3 -m helpers.na49_gateway.cluster_agent reconcile            # via client over SSH
python3 helpers/na49_gateway/cluster_agent.py --jobs-root <dir> reconcile --no-probe
```

The scan derives each job's state, marks unreachable jobs `LOST` in their
`meta.json` (fields `state`, `lost_at`, `lost_reason`), and prints a JSON
summary (`scanned`, `active`, `terminal`, `lost_materialized`,
`already_lost`, `malformed`). Flags:

- `--probe/--no-probe`: SSH PID verification before trusting staleness
  (default enabled; disable for file-only scans).
- `--prune` with `--prune-age-days` (default 14): additionally delete
  terminal job directories whose end time precedes the cutoff. Active jobs
  are never pruned.

`--stale-heartbeat-seconds` and `--launch-grace-seconds` are top-level
options on both the agent and the client CLIs (placed before the
subcommand). They apply to every subcommand, including submit-time
capacity checks, `status`, `list`, `logs --follow`, `reconcile`, and
pruning decisions. Related config keys: `stale_heartbeat_seconds`,
`launch_grace_seconds`, `prune_age_days`. All CLI flags override config
values.

### Upgrade note

Job records created by a pre-heartbeat gateway never produce `last_seen`
markers, so after deploying this version they are judged by the same liveness
rules as everything else. Run one `reconcile` pass soon after upgrading,
with generous thresholds if long-running jobs were launched by the previous
wrapper:

```bash
python3 -m helpers.na49_gateway.cluster_agent \
  --stale-heartbeat-seconds 3600 --launch-grace-seconds 14400 reconcile
```

## Typical usage

Check worker capacity:

```bash
python3 -m helpers.na49_gateway.cluster_client capacity
```

Submit a command:

```bash
python3 -m helpers.na49_gateway.cluster_client submit \
  --workdir ~/NA49/2026/01-NA49-res \
  --name urqmd_chunk_00 \
  --tasks 8 \
  --cmd "python3 dev/mixing2006/examples/run_urqmd_pp158_to_csv.py --events 5000 --seed 1000 --out-csv dev/00runs/<run_description>/urqmd_chunk_00.csv"
```

Inspect status:

```bash
python3 -m helpers.na49_gateway.cluster_client status <job_id>
python3 -m helpers.na49_gateway.cluster_client result <job_id>
python3 -m helpers.na49_gateway.cluster_client logs <job_id> --follow
```

All CLI flags override config-file values.

## Python API

```python
from helpers.na49_gateway.cluster_client import ClusterGatewayClient

client = ClusterGatewayClient(host="your-login-host")
job = client.submit(
    "python3 dev/mixing2006/examples/run_epos_crmc_to_csv.py --events 5000",
    workdir="~/NA49/2026/01-NA49-res",
    tasks=8,
)
final = client.wait(job["job_id"])
```

## Agent vs client config

The same config schema can be used on both sides, but relevant keys differ:

- client side: `host`, `remote_project_relpath` or `agent_command`, `jobs_root`, `ssh_options`
- login-node agent side: `workers`, `max_tasks_per_worker`, `worker_bin_dir`, `jobs_root`, `ssh_options`

If a key is not configured, pass the corresponding CLI flag explicitly.
