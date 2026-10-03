# Solar portable configuration reference

**Status: Read-only preflight/reference; no activation or auto-start change.**

The production and rollback presets directly pin local Docker image IDs. Those exact images
must already exist on both nodes; IDs are not registry pull references. Obtain weights outside
Git and resolve the per-node MODEL_PATH, RoCE/IP/interface settings and Ray prerequisites in
[the production runbook](solar-open2-production.md). Shell overrides preserve clean Git state.

The following renders production configuration only, using the current tracked overlays:

```bash
# Set these explicitly on each node; cache directories must be node-local.
export MODEL_PATH=/absolute/path/to/Solar-Open2-250B-Nota-NVFP4
export B12X_CACHE_DIR=/absolute/node-local/solar-cache
export HEAD_ROCE_IP=192.0.2.1 WORKER_ROCE_IP=192.0.2.2
export ROCE_IF_NAME=your_roce_interface IB_HCA_NAME=your_ib_hca
: "${MODEL_PATH:?set MODEL_PATH}" "${B12X_CACHE_DIR:?set B12X_CACHE_DIR}"
docker compose --env-file presets/solar-open2-250b-nota-nvfp4-v0251-r4-production-tp2.env \
  -f docker-compose.yml -f compose/solar-open2/docker-compose.r4-production.yml \
  -f compose/solar-open2/docker-compose.b12x-cache.yml \
  --profile head --profile worker config
```

The `192.0.2.x` values above are documentation-only placeholders. The cache root must contain `dotcache/`, `triton/`,
`nv/`, and `cutedsl/` directories with appropriate permissions. Cold empty caches pay JIT warmup;
the recorded warm cache cannot be reconstructed from Git. The cache overlay also binds the
existing local observability sink `/home/bjk110/docker-build/c2-obs`; this path remains a prerequisite.
The rollback render uses `solar-open2-250b-nota-nvfp4-v022-kv4g-di-matched-tp2.env` and only
`docker-compose.yml` plus `compose/solar-open2/docker-compose.r4-production.yml`.

For future explicitly authorized activation, follow the production runbook's node order,
reboot/recovery and health gates. This reference does not replace the local production launcher.
The available benchmark launcher uses a superseded active-test preset, detached memory guard
and kernel watcher, and a timestamped cache pointer. Static inspection cannot prove a portable
launcher/fastguard replacement preserves production orchestration. That remains a documented
gap; no launcher or guard executable is introduced, and no service is started by this reference.

`python scripts/diag/verify_preset_catalog.py` renders all tracked presets with safe placeholders
and declared overlays without contacting nodes, pulling images or starting containers.
