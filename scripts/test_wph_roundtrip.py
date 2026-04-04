#!/usr/bin/env python3
"""TurboQuant decode regression test: Triton vs CUDA WPH.

Tests encode→decode round-trip accuracy and WPH/Triton equivalence.
Covers: outlier on/off, odd/even norm_offset, butterfly correctness.

Run inside container:
  python3 /opt/scripts/test_wph_roundtrip.py

Success criteria:
  - Triton-WPH MSE < 1e-6 (should be exact 0.0)
  - Round-trip MSE < 0.15 (4-bit quantization noise)
"""
import math
import sys

import torch


def _run_test(
    name: str,
    head_size: int,
    outlier_fraction: float,
    num_tokens: int = 8,
    num_kv_heads: int = 2,
):
    device = torch.device("cuda")

    from vllm.model_executor.layers.quantization.turboquant import (
        TurboQuantConfig,
        TurboQuantState,
    )
    from vllm.v1.attention.ops.triton_fused_turboquant import (
        _safe_view_fp16,
        fused_hadamard_decode_from_slots,
        fused_hadamard_encode_and_store,
    )

    cfg = TurboQuantConfig(bit_width=4, outlier_fraction=outlier_fraction)
    state = TurboQuantState(config=cfg, head_size=head_size, layer_idx=0, device=device)

    normal_size = state.normal_size
    n_outliers = head_size - normal_size
    packed_bytes = math.ceil(normal_size * 4 / 8)
    outlier_u8_count = n_outliers * 2
    norm_offset = outlier_u8_count + packed_bytes
    slot_bytes = 128  # padded

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  head={head_size} normal={normal_size} outliers={n_outliers} "
          f"norm_offset={norm_offset} ({'ODD' if norm_offset % 2 else 'EVEN'})")
    print(f"{'='*60}")

    # Test data
    torch.manual_seed(42)
    x = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.bfloat16, device=device)

    # Encode
    cache = torch.zeros(num_tokens, 1, num_kv_heads, slot_bytes, dtype=torch.uint8, device=device)
    block_indices = torch.arange(num_tokens, device=device)
    block_offsets = torch.zeros(num_tokens, dtype=torch.long, device=device)

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

    N = num_tokens * num_kv_heads
    flat = cache.reshape(N, slot_bytes)

    # Triton decode
    triton_out = fused_hadamard_decode_from_slots(
        flat_slots=flat, sign_flips=state.sign_flips, codebook=state.codebook,
        normal_idx=state.normal_idx, outlier_idx=state.outlier_idx,
        head_size=head_size, normal_size=normal_size,
        n_outliers=n_outliers, packed_bytes=packed_bytes,
    )

    x_flat = x.reshape(N, head_size).float()
    triton_mse = ((triton_out.float() - x_flat) ** 2).mean().item()
    print(f"  Triton round-trip MSE: {triton_mse:.6f}")

    # WPH decode
    try:
        from vllm.v1.attention.ops.cuda_turboquant_decode import cuda_wph_decode_from_slots

        wph_out = cuda_wph_decode_from_slots(
            flat_slots=flat, sign_flips=state.sign_flips, codebook=state.codebook,
            normal_idx=state.normal_idx, outlier_idx=state.outlier_idx,
            head_size=head_size, normal_size=normal_size,
            n_outliers=n_outliers, packed_bytes=packed_bytes,
        )

        wph_mse = ((wph_out.float() - x_flat) ** 2).mean().item()
        diff_mse = ((wph_out.float() - triton_out.float()) ** 2).mean().item()
        print(f"  WPH round-trip MSE:    {wph_mse:.6f}")
        print(f"  Triton-WPH diff MSE:   {diff_mse:.8f}")

        # Assertions
        assert diff_mse < 1e-6, f"FAIL: Triton-WPH mismatch MSE={diff_mse}"
        assert triton_mse < 0.15, f"FAIL: round-trip MSE too high: {triton_mse}"
        print(f"  ✅ PASS")
        return True

    except ImportError:
        print(f"  ⚠️  WPH extension not available — Triton-only test")
        assert triton_mse < 0.15, f"FAIL: round-trip MSE too high: {triton_mse}"
        print(f"  ✅ PASS (Triton only)")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def _test_safe_view_fp16():
    """Test _safe_view_fp16 utility with odd and even offsets."""
    from vllm.v1.attention.ops.triton_fused_turboquant import _safe_view_fp16

    print(f"\n{'='*60}")
    print(f"  _safe_view_fp16 alignment test")
    print(f"{'='*60}")

    device = torch.device("cuda")
    raw = torch.tensor([0x00, 0x3C, 0x00, 0x40], dtype=torch.uint8, device=device)
    # 0x3C00 = 1.0 in fp16, 0x4000 = 2.0 in fp16

    # Even offset
    even_slice = raw[0:2]
    result = _safe_view_fp16(even_slice.unsqueeze(0))
    assert abs(result.item() - 1.0) < 0.01, f"Even offset failed: {result.item()}"

    # Odd offset — simulated via a larger tensor
    big = torch.zeros(5, dtype=torch.uint8, device=device)
    big[1] = 0x00
    big[2] = 0x3C  # fp16 1.0 at odd offset 1
    odd_slice = big[1:3]
    assert odd_slice.storage_offset() == 1  # Confirm odd
    result2 = _safe_view_fp16(odd_slice.unsqueeze(0))
    assert abs(result2.item() - 1.0) < 0.01, f"Odd offset failed: {result2.item()}"

    print(f"  ✅ PASS (even={1.0}, odd={result2.item():.1f})")
    return True


def main():
    results = []

    # Test 0: _safe_view_fp16 utility
    results.append(_test_safe_view_fp16())

    # Test 1: With outliers (norm_offset=93, ODD)
    results.append(_run_test(
        "4-bit + outliers (norm_offset=93, ODD)",
        head_size=128, outlier_fraction=0.15,
    ))

    # Test 2: No outliers (norm_offset=64, EVEN)
    results.append(_run_test(
        "4-bit no outliers (norm_offset=64, EVEN)",
        head_size=128, outlier_fraction=0.0,
    ))

    # Test 3: Different head size (norm_offset may vary)
    results.append(_run_test(
        "4-bit + outliers, head_size=64",
        head_size=64, outlier_fraction=0.15,
    ))

    print(f"\n{'='*60}")
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"  ALL {total} TESTS PASSED ✅")
    else:
        print(f"  {passed}/{total} PASSED, {total-passed} FAILED ❌")
        sys.exit(1)


if __name__ == "__main__":
    main()
