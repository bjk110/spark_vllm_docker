#!/usr/bin/env python3
"""
Fix cuMemcpyBatchAsync API change in CUDA 13.0+.

CUDA 13.0 removed the `failIdx` parameter from cuMemcpyBatchAsync_v2.
vLLM code targets CUDA 12.8 API (9 args) but CUDA 13.x has 8 args.
This patch updates the call to be compatible with both versions.
"""
import re
from pathlib import Path

target = Path("/workspace/vllm-src/csrc/cache_kernels.cu")
if not target.exists():
    print("[patch] cache_kernels.cu not found, skipping")
    exit(0)

content = target.read_text()

# Replace the CUDA 12.8+ block to handle CUDA 13.0+ API change
old = """#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  CUmemcpyAttributes attr = {};
  attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
  size_t attrs_idx = 0;
  size_t fail_idx = 0;
  CUresult result = cuMemcpyBatchAsync(
      reinterpret_cast<CUdeviceptr*>(const_cast<int64_t*>(dst_data)),
      reinterpret_cast<CUdeviceptr*>(const_cast<int64_t*>(src_data)),
      reinterpret_cast<size_t*>(const_cast<int64_t*>(size_data)),
      static_cast<size_t>(n), &attr, &attrs_idx, 1, &fail_idx,
      static_cast<CUstream>(stream));
  TORCH_CHECK(result == CUDA_SUCCESS, "cuMemcpyBatchAsync failed at index ",
              fail_idx, " with error ", result);"""

new = """#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13000
  // CUDA 13.0+ removed failIdx parameter from cuMemcpyBatchAsync_v2
  CUmemcpyAttributes attr = {};
  attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
  size_t attrs_idx = 0;
  CUresult result = cuMemcpyBatchAsync(
      reinterpret_cast<CUdeviceptr*>(const_cast<int64_t*>(dst_data)),
      reinterpret_cast<CUdeviceptr*>(const_cast<int64_t*>(src_data)),
      reinterpret_cast<size_t*>(const_cast<int64_t*>(size_data)),
      static_cast<size_t>(n), &attr, &attrs_idx, 1,
      static_cast<CUstream>(stream));
  TORCH_CHECK(result == CUDA_SUCCESS, "cuMemcpyBatchAsync failed with error ", result);
#elif !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  // CUDA 12.8-12.x: original API with failIdx
  CUmemcpyAttributes attr = {};
  attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
  size_t attrs_idx = 0;
  size_t fail_idx = 0;
  CUresult result = cuMemcpyBatchAsync(
      reinterpret_cast<CUdeviceptr*>(const_cast<int64_t*>(dst_data)),
      reinterpret_cast<CUdeviceptr*>(const_cast<int64_t*>(src_data)),
      reinterpret_cast<size_t*>(const_cast<int64_t*>(size_data)),
      static_cast<size_t>(n), &attr, &attrs_idx, 1, &fail_idx,
      static_cast<CUstream>(stream));
  TORCH_CHECK(result == CUDA_SUCCESS, "cuMemcpyBatchAsync failed at index ",
              fail_idx, " with error ", result);"""

if old in content:
    content = content.replace(old, new)
    target.write_text(content)
    print("[patch] cache_kernels.cu: fixed cuMemcpyBatchAsync for CUDA 13.0+")
else:
    print("[patch] cache_kernels.cu: pattern not found, may already be patched")
