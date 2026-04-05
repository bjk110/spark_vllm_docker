# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA warp-per-head decode kernel for TurboQuant (Plan A).

Gather-free direct-read from paged KV cache.
Parametric BLOCK_D: supports 128 (4 elems/thread) and 256 (8 elems/thread).
Uses warp shuffle butterfly — no shared memory, no barriers.

AOT-compiled extension: turboquant_wph_ext (built in Dockerfile).
"""

import math
from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import next_power_of_2

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def _load_cuda_module():
    """Load AOT-compiled CUDA WPH extension."""
    try:
        import turboquant_wph_ext
        logger.info("[TQ-WPH] Using AOT-compiled CUDA extension")
        return turboquant_wph_ext
    except ImportError:
        logger.warning("[TQ-WPH] AOT extension not found")
        raise


def cuda_wph_paged_decode(
    cache: torch.Tensor,         # [num_blocks, block_size, num_kv_heads, slot_bytes] uint8
    flat_bt: torch.Tensor,       # [num_entries] int64
    sign_flips: torch.Tensor,    # [BLOCK_D] float32
    codebook: torch.Tensor,      # [num_centroids] float32
    normal_idx: torch.Tensor | None,
    outlier_idx: torch.Tensor | None,
    head_size: int,
    normal_size: int,
    n_outliers: int,
    packed_bytes: int,
    max_seq_len: int = 0,
    warps_per_cta: int = 4,
) -> torch.Tensor:
    """Gather-free WPH decode from paged cache.

    Reads directly from paged KV cache using block_table + strides.
    Norms are read inside the CUDA kernel (no Python extraction).
    Supports BLOCK_D=128 and BLOCK_D=256.

    Returns: [num_entries, block_size, num_kv_heads, head_size] bf16
    """
    mod = _load_cuda_module()

    num_entries = flat_bt.shape[0]
    _, block_size, num_kv_heads, slot_bytes = cache.shape
    block_d = sign_flips.shape[0]

    outlier_u8_count = n_outliers * 2
    norm_offset = outlier_u8_count + packed_bytes
    effective_max = max_seq_len if max_seq_len > 0 else block_size * num_entries

    has_outliers = normal_idx is not None and n_outliers > 0
    if normal_idx is None:
        normal_idx = torch.empty(0, dtype=torch.int64, device=cache.device)
    if outlier_idx is None:
        outlier_idx = torch.empty(0, dtype=torch.int64, device=cache.device)

    # Profiling (TQ_PROFILE=1)
    import os as _os
    _prof = _os.environ.get("TQ_PROFILE", "0") in ("1", "true")
    if _prof and not torch.cuda.is_current_stream_capturing():
        if not hasattr(cuda_wph_paged_decode, "_call_cnt"):
            cuda_wph_paged_decode._call_cnt = 0
            cuda_wph_paged_decode._total_us = 0.0
        cuda_wph_paged_decode._call_cnt += 1
        N_total = num_entries * block_size * num_kv_heads
        effective_rows = min(N_total, effective_max * num_kv_heads) if max_seq_len > 0 else N_total
        skip_pct = (1.0 - effective_rows / max(N_total, 1)) * 100
        if cuda_wph_paged_decode._call_cnt <= 5:
            logger.info(
                "[TQ-WPH-PROF] N=%d effective=%d skip=%.0f%% block_d=%d",
                N_total, effective_rows, skip_pct, block_d,
            )

    # Ensure flat_bt is int64 (some models use int32 block tables)
    if flat_bt.dtype != torch.int64:
        flat_bt = flat_bt.to(torch.int64)

    out_flat = mod.turboquant_wph_paged_decode(
        cache,
        flat_bt,
        sign_flips,
        codebook,
        normal_idx,
        outlier_idx,
        block_size,
        num_kv_heads,
        normal_size,
        head_size,
        n_outliers,
        outlier_u8_count,
        packed_bytes,
        norm_offset,
        effective_max,
        block_d,
        warps_per_cta,
        has_outliers,
    )

    return out_flat.reshape(num_entries, block_size, num_kv_heads, head_size)
