# Entrypoint Scripts

This directory contains the container entrypoint scripts used by `docker-compose.yml`.

## Files

| File | Used by | Notes |
|---|---|---|
| `entrypoint.sh` | Standard path (`dsv4-d568` and general) | CLUSTER_MODE-aware; handles single / dual-rdma / Ray / mp dispatch |
| `entrypoint.unholy.sh` | `unholy-fusion` path only | mp-only; includes GB10 UMA patches and B12X safe defaults |

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

**unholy-fusion path** (set in `.env.unholy-fusion`):

```env
ENTRYPOINT_FILE=./entrypoints/entrypoint.unholy.sh
```

Pass this via `--env-file .env.unholy-fusion` when running the unholy-fusion compose override:

```bash
docker compose \
  -f docker-compose.yml \
  -f compose/docker-compose.unholy.yml \
  --env-file .env.unholy-fusion \
  --profile head up -d
```

## Do not overwrite entrypoint files

Do not use `cp entrypoint.unholy.sh entrypoint.sh` or similar destructive operations.
The `ENTRYPOINT_FILE` variable handles switching without modifying any files.
