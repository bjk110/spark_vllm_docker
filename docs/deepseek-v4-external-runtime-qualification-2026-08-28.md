# DeepSeek-V4 external runtime qualification — 2026-08-28

Status: **closed — no external candidate qualified to replace or extend the active v0.27 production route**.

This report records a bounded comparison of external DeepSeek-V4 runtime lineages on the dual DGX
Spark cluster (GB10/SM121, TP=2) with the official local
`deepseek-ai/DeepSeek-V4-Flash-0731` weights. It does not change production artifacts or extend the
approved concurrency envelope.

## Production baseline and decision boundary

The active repository route remains:

- preset: `presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2.env`;
- overlay: `compose/deepseek-v4/docker-compose.v027-b43s-candidate.yml`;
- image: `ghcr.io/bjk110/vllm-spark@sha256:7a005243701c5df6f8945ea56d509f747f2c63ecff6091a0169d8109a736d09f`;
- local image ID: `sha256:a7f0f4b8a508c0b2510fc7e4dcb916491efa03c380c9c7b84dddd4c16ad6f38d`;
- envelope: TP=2, `MAX_MODEL_LEN=262144`, `MAX_NUM_SEQS=1`, native DSpark k=7.

Candidates had to pass sequential repeatability at server `MAX_NUM_SEQS=1` before server
`MAX_NUM_SEQS=8` batch-invariance. One performance benchmark was allowed only after all correctness
gates were clean. This comparison does not rerun or supersede B4.3S-B4.4C validation of v0.27/MS=1.

## Canonical lossless contract

Every MS1 request reused one persisted 669-byte payload:

- SHA256 `38412b05c32e728604d23b85ace84173bad87829ad1b74a394b00af74c62a241`;
- `/v1/completions`, model `deepseek-ai/DeepSeek-V4-Flash-0731`;
- identical prompt, `max_tokens=512`, `temperature=0`, `top_p=1`, `seed=0`, `stream=false`;
- no retries; full raw response, UTF-8 completion, headers, lengths, usage, finish reason, and hashes retained.

CLEAN required 8/8 HTTP 200, clean JSON/UTF-8/Content-Length integrity, identical request bytes, and
one unique completion-text SHA256.

## Results

| Candidate | Immutable identity | Deepest startup layer | MS1 result | Classification |
|---|---|---|---|---|
| eugr B12X | manifest `sha256:7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd`; image `sha256:f89e9baedf38ffe3165641d4a937b59b227bbbd58d0116a7518c53b97d601823` | TP=2, B12X, DSpark k=5, AOT/CUDA graph, health 200 | 8/8 HTTP/integrity clean; 8 unique completion hashes; 7/8 divergence | `EUGR_MS1_BLOCKED_GREEDY_DIVERGENCE` |
| MiaAI/Anemll | Mia `70a7cc4b49664e83b51e9b73c0ed41db18ac3190`; manifest `sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`; image `sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8` | TP=2, B12X/MXFP4, sparse MLA, graph/JIT, health 200 | 8/8 HTTP/integrity clean; 8 unique completion hashes; 7/8 divergence | `MIA_MS1_BLOCKED_GREEDY_DIVERGENCE` |
| Aiden/Tony MTP=2 | Tony `c6b121cdbbb6bc7bd8ee5f4480c28a7ea674d538`; manifest `sha256:f869281d869b2a1d418cade7dcbabe65216cb8f54891d5e2e50718c3dc8b630f`; image `sha256:b2eb4e6ee5cc8b69b941bd7a00c2be93a5c2c93e0c74dc7306a42919ccdf9f2d` | TP/NCCL and main model 48/48 shards | MTP drafter failed: `KeyError: model.layers.43.mtp_block.main_norm.weight` | `AIDEN_TONY_0731_MTP_STARTUP_BLOCKED_WEIGHT_SCHEMA` |
| Aiden/Tony MTP-off child | same image/source/model; MTP was the single removed feature | model load, AOT/TileLang, FlashInfer autotune, CUDA graph, health 200 | 8/8 HTTP/integrity clean; 8 unique completion hashes; 7/8 divergence | `AIDEN_NOMTP_MS1_BLOCKED_GREEDY_DIVERGENCE` |

The MTP-off run was a new child candidate, not a retry under the failed MTP identity.

## Performance and production conclusion

No candidate established a trustworthy performance improvement over v0.27: correctness failed before
comparative benchmarking, so server MS8 and `llama-benchy` were not run, retried, or tuned. External
throughput and context reports used non-equivalent conditions and are not equal-envelope evidence.

B12X, sparse MLA, FlashInfer autotuning, CUDA graph, and MTP remain experimental inputs. Mixing any
provider image, patch, or launcher into production creates a new candidate requiring the full gate.
The conservative production recommendation remains v0.27 with `MAX_NUM_SEQS=1`; client concurrency
must not be treated as server MS8 qualification.

## Operational outcome and evidence

- Production artifacts were neither modified nor started during qualification.
- Experimental containers, images, caches, and node-local launch copies were removed.
- Official 0731 weights, v0.27 production image, node-exporter, and Portainer were preserved.
- Final free storage: spark01 about 436 GiB; spark02 about 569 GiB.
- After final reboots both nodes had about 123 GiB `MemAvailable`; port 8000 was closed.

Host-local raw evidence (not committed):

- `/home/bjk110/docker-build/dsv4-external-recipes-20260827` — eugr and MiaAI/Anemll;
- `/home/bjk110/docker-build/dsv4-aiden-tony-20260828` — Aiden/Tony.

These directories contain raw responses, runtime logs, classifications, and checksum manifests. This
repository stores the stable decisions and immutable identities only.
