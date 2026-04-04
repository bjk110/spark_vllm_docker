#!/usr/bin/env python3
"""TurboQuant WPH v2 regression test.

Tests gather-free paged decode: Triton vs CUDA WPH accuracy.
Covers BLOCK_D 128/256, outlier on/off, multi-head, multi-entry.

Run:  python3 test_wph_v2.py          (inside container with GPU)
Exit: 0=pass, 1=fail
"""
import math
import sys
import torch


def _run_case(name, head_size, outlier_fraction, num_tokens, num_kv_heads):
    device = torch.device("cuda")

    from vllm.model_executor.layers.quantization.turboquant import (
        TurboQuantConfig, TurboQuantState,
    )
    from vllm.v1.attention.ops.triton_fused_turboquant import (
        fused_hadamard_encode_and_store, fused_paged_decode,
    )

    cfg = TurboQuantConfig(bit_width=4, outlier_fraction=outlier_fraction)
    state = TurboQuantState(config=cfg, head_size=head_size, layer_idx=0, device=device)

    normal_size = state.normal_size
    n_outliers = head_size - normal_size
    packed_bytes = math.ceil(normal_size * 4 / 8)
    block_d = state.sign_flips.shape[0]
    slot_bytes = 128 if block_d <= 128 else 256

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  head={head_size} normal={normal_size} outliers={n_outliers} "
          f"block_d={block_d} kv_heads={num_kv_heads} tokens={num_tokens}")
    print(f"{'='*60}")

    # Test data
    torch.manual_seed(42)
    x = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.bfloat16, device=device)

    # Encode
    block_size = 2  # small block_size for multi-block test
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache = torch.zeros(num_blocks, block_size, num_kv_heads, slot_bytes,
                        dtype=torch.uint8, device=device)

    block_indices = torch.arange(num_tokens, device=device) // block_size
    block_offsets = torch.arange(num_tokens, device=device) % block_size

    has_outliers = state.outlier_idx is not None and n_outliers > 0
    if has_outliers:
        normal_x = x[..., state.normal_idx].contiguous()
        outlier_x = x[..., state.outlier_idx]
    else:
        normal_x = x
        outlier_x = None

    fused_hadamard_encode_and_store(
        normal_x=normal_x, outlier_x=outlier_x,
        sign_flips=state.sign_flips, boundaries=state.boundaries,
        cache=cache, block_indices=block_indices, block_offsets=block_offsets,
        bit_width=4,
    )

    # Decode setup
    flat_bt = torch.arange(num_blocks, device=device, dtype=torch.long)
    num_entries = num_blocks
    max_seq_len = num_tokens

    # Triton decode
    triton_out = fused_paged_decode(
        cache=cache, flat_bt=flat_bt,
        sign_flips=state.sign_flips, codebook=state.codebook,
        normal_idx=state.normal_idx, outlier_idx=state.outlier_idx,
        head_size=head_size, normal_size=normal_size,
        n_outliers=n_outliers, packed_bytes=packed_bytes,
        max_seq_len=max_seq_len,
    )

    x_flat = x.reshape(-1, head_size).float()
    # Triton output: (entries, block_size, kv_heads, head_size)
    # Match to x ordering: entry i, slot j → token i*block_size+j
    triton_flat = triton_out.reshape(-1, head_size)[:num_tokens * num_kv_heads].float()
    triton_mse = ((triton_flat - x_flat.repeat_interleave(1, dim=0)[:triton_flat.shape[0]]) ** 2).mean().item()
    print(f"  Triton round-trip MSE: {triton_mse:.6f}")

    # WPH decode
    try:
        from vllm.v1.attention.ops.cuda_turboquant_decode import cuda_wph_paged_decode

        wph_out = cuda_wph_paged_decode(
            cache=cache, flat_bt=flat_bt,
            sign_flips=state.sign_flips, codebook=state.codebook,
            normal_idx=state.normal_idx, outlier_idx=state.outlier_idx,
            head_size=head_size, normal_size=normal_size,
            n_outliers=n_outliers, packed_bytes=packed_bytes,
            max_seq_len=max_seq_len,
        )

        wph_flat = wph_out.reshape(-1, head_size)[:triton_flat.shape[0]].float()
        diff_mse = ((wph_flat - triton_flat) ** 2).mean().item()
        diff_max = (wph_flat - triton_flat).abs().max().item()
        print(f"  WPH-Triton diff: MSE={diff_mse:.8f} max={diff_max:.6f}")

        ok = diff_mse < 1e-5
        print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
        return ok

    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    results = []

    # BLOCK_D=256 + outliers (Qwen3.5 actual config)
    results.append(_run_case(
        "BLOCK_D=256 + outliers (Qwen3.5)", head_size=256,
        outlier_fraction=0.15, num_tokens=8, num_kv_heads=2))

    # BLOCK_D=256 + outliers, more tokens & heads
    results.append(_run_case(
        "BLOCK_D=256 + outliers, 16tok 4heads", head_size=256,
        outlier_fraction=0.15, num_tokens=16, num_kv_heads=4))

    # BLOCK_D=256, no outliers
    results.append(_run_case(
        "BLOCK_D=256 no outliers", head_size=256,
        outlier_fraction=0.0, num_tokens=8, num_kv_heads=2))

    # BLOCK_D=128 + outliers
    results.append(_run_case(
        "BLOCK_D=128 + outliers", head_size=128,
        outlier_fraction=0.15, num_tokens=8, num_kv_heads=2))

    # BLOCK_D=128, no outliers
    results.append(_run_case(
        "BLOCK_D=128 no outliers", head_size=128,
        outlier_fraction=0.0, num_tokens=8, num_kv_heads=2))

    # Stress: many tokens
    results.append(_run_case(
        "Stress: 64 tokens, 4 heads, BLOCK_D=256", head_size=256,
        outlier_fraction=0.15, num_tokens=64, num_kv_heads=4))

    print(f"\n{'='*60}")
    p = sum(results)
    t = len(results)
    print(f"  {p}/{t} PASSED" + (" ✅" if p == t else f", {t-p} FAILED ❌"))
    sys.exit(0 if p == t else 1)


if __name__ == "__main__":
    main()
