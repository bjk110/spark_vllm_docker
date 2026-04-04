#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// TurboQuant warp-per-head decode kernel (Plan A v2).
//
// Key optimizations over v1:
//   - Multi-warp CTA: WARPS_PER_CTA warps per block (default 4)
//     → 4x fewer launches, better occupancy, shared L1 for metadata
//   - Parametric BLOCK_D=128/256 via template
//   - Gather-free direct paged-cache read
//   - Norm read inside kernel (no Python extraction)
//   - Early exit for padding slots

constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_shuffle_xor(float val, int mask) {
    return __shfl_xor_sync(0xFFFFFFFF, val, mask);
}

// Intra-thread butterfly: XOR-based swap for EPT elements
template <int EPT>
__device__ __forceinline__ void butterfly_intra_thread(float* vals) {
    #pragma unroll
    for (int h = 1; h < EPT; h <<= 1) {
        #pragma unroll
        for (int e = 0; e < EPT; e++) {
            int partner_e = e ^ h;
            if (partner_e > e) {
                float a = vals[e], b = vals[partner_e];
                vals[e]         = a + b;
                vals[partner_e] = a - b;
            }
        }
    }
}

template <int BLOCK_D, int WPC>
__global__ void turboquant_wph_paged_decode_kernel(
    const uint8_t* __restrict__ cache,
    const int64_t* __restrict__ flat_bt,
    const float*   __restrict__ sign_flips,
    const float*   __restrict__ codebook,
    const int64_t* __restrict__ normal_idx,
    const int64_t* __restrict__ outlier_idx,
    __nv_bfloat16* __restrict__ output,
    int64_t cache_stride_block,
    int64_t cache_stride_bs,
    int64_t cache_stride_head,
    int block_size,
    int num_kv_heads,
    int normal_size,
    int head_size,
    int n_outliers,
    int outlier_u8_count,
    int packed_bytes,
    int norm_offset,
    int max_seq_len,
    int total_rows,
    bool has_outliers
) {
    constexpr int EPT = BLOCK_D / WARP_SIZE;

    // Multi-warp CTA: each warp handles its own row
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WPC + warp_id;

    if (row >= total_rows) return;

    // Decompose row → (entry, slot_in_block, head)
    // Pre-compute reciprocal would help but nvcc optimizes this for us
    const int slots_per_entry = block_size * num_kv_heads;
    const int entry_idx = row / slots_per_entry;
    const int rem = row % slots_per_entry;
    const int bs_idx = rem / num_kv_heads;
    const int head_idx = rem % num_kv_heads;

    // Early exit for padding
    if (entry_idx * block_size + bs_idx >= max_seq_len) return;

    // Direct paged-cache slot address
    const int64_t block_id = flat_bt[entry_idx];
    const uint8_t* slot = cache
        + block_id * cache_stride_block
        + bs_idx * cache_stride_bs
        + head_idx * cache_stride_head;

    const int base_idx = lane_id * EPT;
    __nv_bfloat16* out_row = output + (int64_t)row * head_size;

    // ── Unpack 4-bit + codebook lookup ──
    float vals[EPT];
    const uint8_t* packed = slot + outlier_u8_count;

    #pragma unroll
    for (int e = 0; e < EPT; e++) {
        int dim = base_idx + e;
        if (dim < normal_size) {
            int byte_idx = dim / 2;
            int is_high = dim % 2;
            uint8_t packed_byte = packed[byte_idx];
            int idx = is_high ? ((packed_byte >> 4) & 0xF) : (packed_byte & 0xF);
            vals[e] = codebook[idx];
        } else {
            vals[e] = 0.0f;
        }
    }

    // ── Inverse Hadamard butterfly ──
    // Phase 1: intra-thread (h = 1 .. EPT/2)
    butterfly_intra_thread<EPT>(vals);

    // Phase 2: cross-thread warp shuffle (h = EPT .. BLOCK_D/2)
    #pragma unroll
    for (int h = EPT; h < BLOCK_D; h *= 2) {
        int shuffle_mask = h / EPT;
        bool is_lower = ((lane_id * EPT) & h) == 0;
        #pragma unroll
        for (int e = 0; e < EPT; e++) {
            float partner = warp_shuffle_xor(vals[e], shuffle_mask);
            vals[e] = is_lower ? vals[e] + partner : partner - vals[e];
        }
    }

    // ── Scale + sign flip + norm ──
    const float had_scale = 1.0f / sqrtf((float)BLOCK_D);

    // Norm: read 2 bytes from slot, reinterpret as fp16
    uint16_t norm_bits = slot[norm_offset] | (uint16_t(slot[norm_offset + 1]) << 8);
    half norm_h;
    memcpy(&norm_h, &norm_bits, sizeof(norm_h));
    const float norm_val = __half2float(norm_h) * had_scale;

    #pragma unroll
    for (int e = 0; e < EPT; e++) {
        int dim = base_idx + e;
        if (dim < BLOCK_D) {
            vals[e] *= sign_flips[dim] * norm_val;
        }
    }

    // ── Write output ──
    if (has_outliers) {
        #pragma unroll
        for (int e = 0; e < EPT; e++) {
            int dim = base_idx + e;
            if (dim < normal_size) {
                out_row[normal_idx[dim]] = __float2bfloat16(vals[e]);
            }
        }
        const uint8_t* ob = slot;
        for (int o = lane_id; o < n_outliers; o += WARP_SIZE) {
            uint16_t bf16_bits = ob[o * 2] | (uint16_t(ob[o * 2 + 1]) << 8);
            __nv_bfloat16 val;
            memcpy(&val, &bf16_bits, sizeof(val));
            out_row[outlier_idx[o]] = val;
        }
    } else {
        #pragma unroll
        for (int e = 0; e < EPT; e++) {
            int dim = base_idx + e;
            if (dim < normal_size) {
                out_row[dim] = __float2bfloat16(vals[e]);
            }
        }
    }
}

// ── C++ dispatch ──
torch::Tensor turboquant_wph_paged_decode(
    torch::Tensor cache,
    torch::Tensor flat_bt,
    torch::Tensor sign_flips,
    torch::Tensor codebook,
    torch::Tensor normal_idx,
    torch::Tensor outlier_idx,
    int block_size,
    int num_kv_heads,
    int normal_size,
    int head_size,
    int n_outliers,
    int outlier_u8_count,
    int packed_bytes,
    int norm_offset,
    int max_seq_len,
    int block_d,
    int warps_per_cta,
    bool has_outliers
) {
    int num_entries = flat_bt.size(0);
    int N = num_entries * block_size * num_kv_heads;

    auto output = torch::zeros({N, head_size},
        torch::TensorOptions().dtype(torch::kBFloat16).device(cache.device()));

    if (N == 0) return output;

    dim3 grid((N + warps_per_cta - 1) / warps_per_cta);
    dim3 blk(WARP_SIZE * warps_per_cta);
    auto stream = c10::cuda::getCurrentCUDAStream();

    int64_t stride_block = cache.stride(0);
    int64_t stride_bs    = cache.stride(1);
    int64_t stride_head  = cache.stride(2);

    #define ARGS \
            cache.data_ptr<uint8_t>(), \
            flat_bt.data_ptr<int64_t>(), \
            sign_flips.data_ptr<float>(), \
            codebook.data_ptr<float>(), \
            has_outliers ? normal_idx.data_ptr<int64_t>() : nullptr, \
            has_outliers ? outlier_idx.data_ptr<int64_t>() : nullptr, \
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()), \
            stride_block, stride_bs, stride_head, \
            block_size, num_kv_heads, normal_size, head_size, \
            n_outliers, outlier_u8_count, packed_bytes, norm_offset, \
            max_seq_len, N, has_outliers

    #define LAUNCH(BD, WPC) \
        turboquant_wph_paged_decode_kernel<BD, WPC><<<grid, blk, 0, stream>>>(ARGS)

    // Dispatch: BLOCK_D × WARPS_PER_CTA
    if (block_d == 256) {
        if      (warps_per_cta == 2) { LAUNCH(256, 2); }
        else if (warps_per_cta == 4) { LAUNCH(256, 4); }
        else if (warps_per_cta == 8) { LAUNCH(256, 8); }
        else { TORCH_CHECK(false, "warps_per_cta must be 2/4/8, got ", warps_per_cta); }
    } else if (block_d == 128) {
        if      (warps_per_cta == 2) { LAUNCH(128, 2); }
        else if (warps_per_cta == 4) { LAUNCH(128, 4); }
        else if (warps_per_cta == 8) { LAUNCH(128, 8); }
        else { TORCH_CHECK(false, "warps_per_cta must be 2/4/8, got ", warps_per_cta); }
    } else {
        TORCH_CHECK(false, "Unsupported BLOCK_D=", block_d);
    }
    #undef LAUNCH
    #undef ARGS

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("turboquant_wph_paged_decode", &turboquant_wph_paged_decode,
          "TurboQuant gather-free warp-per-head paged decode (CUDA)");
}
