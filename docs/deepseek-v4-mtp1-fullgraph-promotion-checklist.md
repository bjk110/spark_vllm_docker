# DeepSeek-V4-Flash MTP+FULL-graph — Validated Preset Promotion Acceptance Checklist

For the LATER, separate promotion-approval step. This checklist gates the transition
`VALIDATED_PROMOTION_CANDIDATE` → `VALIDATED_PRESET`. It does **not** authorize a
`PRODUCTION_BASELINE` cutover. Every item must be explicitly confirmed before staging or
committing the promotion patch. No runtime execution is part of this checklist.

## Identity and presets
- [ ] Validated preset path = `presets/deepseek-v4-v023-stack-pr41834-mtp1-fullgraph-validated-tp2.env`
- [ ] Validated preset SHA-256 = `d116f132ca0087a6773ba9769134c46ace3e707fbe313456cca6ab34b46969b1`
- [ ] Validated preset runtime is byte-identical to tested candidate `f0bb73814dd600c1…` (empty normalized diff)
- [ ] Immutable image config ID = `sha256:4c41950c47ecb771eead3e32147a750240d6605d15ee99c54b43e0a85ea26105`
- [ ] Image manifest digest = `sha256:2f4a96283fc5b491d5e28cee607525e32e914615bb469978beb2336cf8e62c44`
- [ ] Official model identity = `deepseek-ai/DeepSeek-V4-Flash` (46 shards, ~148.7 GiB, `DeepseekV4ForCausalLM`)
- [ ] Pinned source = `72261a7af149fa5d3fe2ed2b9956e92590731012`

## Provenance and rollback
- [ ] Provenance doc present = `docs/deepseek-v4-mtp1-fullgraph-validated-preset.md`
- [ ] Rollback L1 (graph-only) = `presets/deepseek-v4-v023-stack-pr41834-fullgraph-validated-rollback-tp2.env`, SHA `87ad3920b3e681ef…` (tested source: `...-fullgraph-safety-f2-tp2.env` `61468aeff7faf8f4…`)
- [ ] Rollback L2 (eager U0-RDMA) = `presets/deepseek-v4-v023-stack-pr41834-eager-u0-rollback-tp2.env`, SHA `bffea15863867f5e…`
- [ ] Bounded rollback procedure reviewed (stop → preserve → shutdown → reboot-if-UVM → cache clear → hash verify → start → correctness → transport → graph/eager proof → record reason)

## Operational envelope
- [ ] Clean-start procedure reviewed (17-step startup gate; automatic reboot NOT in preset)
- [ ] Dedicated-cache-clear procedure reviewed (clear ONLY `./.cache/vllm`, both nodes)
- [ ] Memory envelope confirmed (clean ~117–118 GiB; steady min ~30–32 GiB; post-stop UVM ~37–40 GiB; reboot before next load)
- [ ] Swap behaviour confirmed (~2–3 GiB init, flat after; no sustained paging)
- [ ] Throughput range confirmed (d0 ~38.9–40.2 t/s, d4096 ~37.1–39.8 t/s; NO fixed 39 t/s floor)

## Evidence
- [ ] Long-soak evidence present (`MTP1_FULLGRAPH_LONG_SOAK_PASS`, 4.004 h, 2,040/2,040)
- [ ] Cold-start evidence present (`MTP1_FULLGRAPH_COLD_REPRO_PASS`, R0–R4, 60.1 min, 507/507)

## Limitations (must be acknowledged, not waived)
- [ ] Concurrency validated only at 1
- [ ] MTP n=1 / capture size `[2]` locked
- [ ] Parser / tool-parser / B12X / Ray off and unvalidated
- [ ] Single-node TP=1 impossible for the official checkpoint
- [ ] Historical 34.59 t/s equivalence remains OPEN

## Production boundary
- [ ] The current production/promoted DSV4 path (`dsv4-d568`) remains UNTOUCHED until a separate explicit production-promotion authorization
- [ ] This promotion does NOT declare `PRODUCTION_BASELINE`
