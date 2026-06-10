#!/usr/bin/env python3
"""Fix NVFP4 ModelOpt MoE input_scale loading for Step-3.7-Flash-NVFP4.

Root cause of all-NaN logits on dual-GB10 TP=2:

The checkpoint's "old packed 3D format" stores per-expert NVFP4 input
scales as `.moe.{gate,up,down}_proj.input_scale` (shape [num_experts]).
`Step3p5Model.load_weights()`'s `expert_params_mapping` only mapped the
`.weight` tensors to `w13_weight`/`w2_weight` -- there was no entry for
`.input_scale`, so these tensors were never matched and the FusedMoE
`w13_input_scale`/`w2_input_scale` parameters were left as
`torch.empty(...)` (uninitialized garbage), corrupting the NVFP4 MoE GEMM
and producing NaN for every token.

Fix part 1 (step3p5.py): add the missing `.input_scale` entries to
`expert_params_mapping` so these tensors are loaded.

Fix part 2 (fused_moe/layer.py): `w13_input_scale` is shape
[num_experts, 2] (one slot each for w1/w3), but the generic
`_load_single_value()` path does `param.data[expert_id] = loaded_weight`,
which broadcasts a scalar across both slots -- so loading w3 after w1
overwrites w1's value. Add a ModelOpt-aware dual-shard write
(`param.data[expert_id][shard_idx] = loaded_weight`) before that path,
mirroring the existing `_load_per_tensor_weight_scale()` handling already
used for `weight_scale_2`.

Reference: eugr/spark-vllm-docker mods/step-3.7-flash/step-3.7-support.patch
"""

STEP3P5_PATH = (
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/step3p5.py"
)
LAYER_PATH = (
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py"
)

# --- Part 1: step3p5.py expert_params_mapping ---
step3p5_old = """        # Old packed 3D format: .moe.gate_proj.weight [num_experts, out, in]
        expert_params_mapping = [
            (f".moe.experts.{base_layer}w13_weight", ".moe.gate_proj.weight", "w1"),
            (f".moe.experts.{base_layer}w13_weight", ".moe.up_proj.weight", "w3"),
            (f".moe.experts.{base_layer}w2_weight", ".moe.down_proj.weight", "w2"),
        ]"""

step3p5_new = """        # Old packed 3D format: .moe.gate_proj.weight [num_experts, out, in]
        expert_params_mapping = [
            (f".moe.experts.{base_layer}w13_weight", ".moe.gate_proj.weight", "w1"),
            (f".moe.experts.{base_layer}w13_weight", ".moe.up_proj.weight", "w3"),
            (f".moe.experts.{base_layer}w2_weight", ".moe.down_proj.weight", "w2"),
            # NVFP4 ModelOpt per-expert input scales (old packed 3D format).
            # Without these, w13_input_scale/w2_input_scale stay
            # uninitialized -> NaN MoE output (see module docstring).
            (
                f".moe.experts.{base_layer}w13_input_scale",
                ".moe.gate_proj.input_scale",
                "w1",
            ),
            (
                f".moe.experts.{base_layer}w13_input_scale",
                ".moe.up_proj.input_scale",
                "w3",
            ),
            (
                f".moe.experts.{base_layer}w2_input_scale",
                ".moe.down_proj.input_scale",
                "w2",
            ),
        ]"""

# --- Part 2: fused_moe/layer.py dual-shard input_scale write ---
layer_old = """        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                "compressed" in quant_method_name.lower()"""

layer_new = """        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            # ModelOpt NVFP4 stores w13 input scales as two logical shards
            # (w1, w3) in a [num_experts, 2] tensor. The generic
            # _load_single_value() below assigns a scalar to
            # param.data[expert_id], which broadcasts across both shards
            # and lets the second shard overwrite the first.
            if (
                "ModelOpt" in quant_method_name
                and param.data.ndim == 2
                and shard_id in ("w1", "w3")
            ):
                scale_expert_id = global_expert_id if use_global_sf else expert_id
                scale_shard_id = 0 if shard_id == "w1" else 1
                param.data[scale_expert_id][scale_shard_id] = loaded_weight.reshape(())
                return True if return_success else None

            if (
                "compressed" in quant_method_name.lower()"""


def apply(path: str, old: str, new: str) -> None:
    with open(path) as f:
        content = f.read()
    if new in content:
        print(f"{path}: already patched -- no change needed.")
        return
    assert old in content, f"{path}: anchor not found:\n{old}"
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"{path}: patched.")


apply(STEP3P5_PATH, step3p5_old, step3p5_new)
apply(LAYER_PATH, layer_old, layer_new)
