# Entrypoint Scripts

This directory contains the container entrypoint scripts used by `docker-compose.yml`.

## Files

| File | Used by | Notes |
|---|---|---|
| `entrypoint.sh` | Standard path (`dsv4-d568` and general) | CLUSTER_MODE-aware; handles single / dual-rdma / Ray / mp dispatch |

`entrypoint.unholy.sh` (the `unholy-fusion` path's entrypoint) was removed from the active tree
2026-08-11 along with its remaining config (`.env.unholy-fusion`,
`compose/docker-compose.unholy.yml`, removed earlier the same day) — no currently supported preset,
Compose path, or documented workflow references it anymore. It is recoverable from Git history at
commit `282e656` or earlier; see [`docs/unholy-fusion-benchmark.md`](../docs/unholy-fusion-benchmark.md)
for the retained historical configuration and benchmark record.

## How selection works

`docker-compose.yml` mounts the selected file into the container as `/entrypoint.sh`:

```yaml
- ${ENTRYPOINT_FILE:-./entrypoints/entrypoint.sh}:/entrypoint.sh:ro
```

The variable `ENTRYPOINT_FILE` controls which host file is used. The container path is always `/entrypoint.sh`.

**Standard path** (default, no override needed):

```env
ENTRYPOINT_FILE=./entrypoints/entrypoint.sh
```

This default is implicit in `docker-compose.yml` — no explicit value is required for the normal path.

**unholy-fusion path — historical, not runnable from `main`.** All three files it required
(`entrypoint.unholy.sh`, `.env.unholy-fusion`, `compose/docker-compose.unholy.yml`) have been
removed from the active tree (2026-08-11) as historical/experimental and not a recommended
production path. All three are recoverable from Git history at commit `282e656` or earlier — see
[`docs/unholy-fusion-benchmark.md`](../docs/unholy-fusion-benchmark.md) for the retained
configuration and benchmark record.

## Do not overwrite entrypoint files

Do not use destructive operations such as copying one entrypoint script over another.
The `ENTRYPOINT_FILE` variable handles switching without modifying any files.
