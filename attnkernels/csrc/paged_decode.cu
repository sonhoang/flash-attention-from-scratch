// Decode attention over a paged KV cache (vLLM-style), with split-K
// ("flash-decoding") parallelism over the context.
//
// Decode: each sequence has one new query token per head and attends to its
// whole context. The cache is a pool of fixed-size blocks,
//     k_cache, v_cache: [num_blocks, num_kv_heads, block_size, head_dim]
// and sequence b finds token t in physical block block_tables[b][t / block_size]
// at offset t % block_size. Context lengths can differ per sequence.
//
// The work is tiny per sequence (one row of Q), so with a small batch there
// are not enough (sequence, head) pairs to fill the GPU. Split-K fixes that:
// the context is cut into partitions, one thread block per
// (partition, kv head, sequence). Each block writes an unnormalised partial
// result plus its softmax max and sum, and a second small kernel merges the
// partials:
//     M = max_s m_s,  w_s = l_s * 2^(m_s - M),  O = sum_s w_s O_s / sum_s w_s
//
// Grouped-query attention: a block handles all G = num_heads / num_kv_heads
// query heads that share one kv head, so each K/V byte is read once.
//
// Inside a block the partition is processed in chunks of 256 tokens with an
// online softmax across chunks. Per chunk:
//   1. scores: 8 lanes per token, each lane holds 8 (D=64) or 16 (D=128)
//      elements of K as 16-byte loads, 3 shuffles reduce the dot product;
//      a warp covers 4 tokens per step, the block 16.
//   2. softmax statistics per head (one warp per head).
//   3. P V: D/8 threads cover one V row with 16-byte loads; the other
//      threads take other tokens; partial sums are reduced at the end.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "common.cuh"

namespace attn {

constexpr int kDecThreads = 128;
constexpr int kChunk = 256;

template <int D, int G>
__global__ void __launch_bounds__(kDecThreads)
paged_decode_kernel(const __half* __restrict__ q, const __half* __restrict__ kc,
                    const __half* __restrict__ vc, const int* __restrict__ block_tables,
                    const int* __restrict__ ctx_lens, __half* __restrict__ out,
                    float* __restrict__ o_part, float* __restrict__ m_part,
                    float* __restrict__ l_part, int hkv, int bs, int max_blocks, int partition,
                    int num_splits, float scale_log2) {
    constexpr int E = D / 8;               // K elements per lane (scores)
    constexpr int DV = D / 8;              // threads per V row (P V)
    constexpr int NTG = kDecThreads / DV;  // token groups in P V
    constexpr int RED = (kDecThreads / 32) * G * D;      // floats for the final reduction
    constexpr int BUF = RED > G * kChunk ? RED : G * kChunk;

    __shared__ float q_s[G][D];
    __shared__ __align__(16) float buf[BUF];  // logits [G][kChunk], later reduction [4][G][D]
    __shared__ float m_s[G], l_s[G], a_s[G];
    float(*logit)[kChunk] = reinterpret_cast<float(*)[kChunk]>(buf);

    const int split = blockIdx.x, kvh = blockIdx.y, b = blockIdx.z;
    const int hq = hkv * G;
    const int ctx = ctx_lens[b];
    const int start = split * partition;
    if (start >= ctx) return;
    const int end = min(ctx, start + partition);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    for (int idx = tid; idx < G * D; idx += kDecThreads) {
        const int h = idx / D, d = idx % D;
        q_s[h][d] = __half2float(q[((size_t)b * hq + kvh * G + h) * D + d]) * scale_log2;
    }
    if (tid < G) {
        m_s[tid] = -INFINITY;
        l_s[tid] = 0.f;
    }
    const int tg = tid / DV, dv = tid % DV;
    float acc[G][8];
#pragma unroll
    for (int h = 0; h < G; ++h)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[h][j] = 0.f;
    const int* bt = block_tables + (size_t)b * max_blocks;
    const size_t head_off = (size_t)kvh * bs * D;
    const size_t block_stride = (size_t)hkv * bs * D;
    __syncthreads();

    const int sub = lane >> 3, l8 = lane & 7;
    for (int c0 = start; c0 < end; c0 += kChunk) {
        const int nt = min(kChunk, end - c0);
        // 1. scores
        for (int base = 0; base < nt; base += 16) {
            const int t = base + warp * 4 + sub;
            const bool valid = t < nt;
            float kf[E];
            if (valid) {
                const int pos = c0 + t;
                const __half* kp = kc + (size_t)bt[pos / bs] * block_stride + head_off +
                                   (size_t)(pos % bs) * D;
#pragma unroll
                for (int e8 = 0; e8 < E / 8; ++e8) load8(kp + e8 * 64 + l8 * 8, kf + e8 * 8);
            } else {
#pragma unroll
                for (int e = 0; e < E; ++e) kf[e] = 0.f;
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
                float dot = 0.f;
#pragma unroll
                for (int e = 0; e < E; ++e) dot += q_s[h][(e >> 3) * 64 + l8 * 8 + (e & 7)] * kf[e];
                dot += __shfl_xor_sync(0xffffffffu, dot, 4);
                dot += __shfl_xor_sync(0xffffffffu, dot, 2);
                dot += __shfl_xor_sync(0xffffffffu, dot, 1);
                if (valid && l8 == 0) logit[h][t] = dot;
            }
        }
        __syncthreads();
        // 2. softmax statistics, one warp per head
        for (int h = warp; h < G; h += kDecThreads / 32) {
            float mx = -INFINITY;
            for (int t = lane; t < nt; t += 32) mx = fmaxf(mx, logit[h][t]);
            mx = warp_max(mx);
            const float m_old = m_s[h];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.f;
            for (int t = lane; t < nt; t += 32) {
                const float p = exp2f(logit[h][t] - m_new);
                logit[h][t] = p;
                sum += p;
            }
            sum = warp_sum(sum);
            if (lane == 0) {
                const float alpha = exp2f(m_old - m_new);
                a_s[h] = alpha;
                l_s[h] = l_s[h] * alpha + sum;
                m_s[h] = m_new;
            }
        }
        __syncthreads();
        // 3. acc = acc * alpha + P V
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float a = a_s[h];
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[h][j] *= a;
        }
        for (int t = tg; t < nt; t += NTG) {
            const int pos = c0 + t;
            const __half* vp = vc + (size_t)bt[pos / bs] * block_stride + head_off +
                               (size_t)(pos % bs) * D + dv * 8;
            float vf[8];
            load8(vp, vf);
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float p = logit[h][t];
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[h][j] += p * vf[j];
            }
        }
        __syncthreads();  // logits and a_s are rewritten by the next chunk
    }

    // Reduce acc over token groups: inside the warp by shuffles, then across
    // the 4 warps through shared memory (reusing the logit buffer).
#pragma unroll
    for (int h = 0; h < G; ++h)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int off = DV; off < 32; off <<= 1)
                acc[h][j] += __shfl_xor_sync(0xffffffffu, acc[h][j], off);
    float(*red)[G][D] = reinterpret_cast<float(*)[G][D]>(buf);
    if (lane < DV) {
#pragma unroll
        for (int h = 0; h < G; ++h)
#pragma unroll
            for (int j = 0; j < 8; ++j) red[warp][h][dv * 8 + j] = acc[h][j];
    }
    __syncthreads();
    for (int idx = tid; idx < G * D; idx += kDecThreads) {
        const int h = idx / D, d = idx % D;
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < kDecThreads / 32; ++w) s += red[w][h][d];
        s /= l_s[h];
        const size_t row = (size_t)b * hq + kvh * G + h;
        if (num_splits == 1) {
            out[row * D + d] = __float2half(s);
        } else {
            o_part[(row * num_splits + split) * D + d] = s;
            if (d == 0) {
                m_part[row * num_splits + split] = m_s[h];
                l_part[row * num_splits + split] = l_s[h];
            }
        }
    }
}

// One block per (sequence, query head), one thread per head dim.
__global__ void paged_reduce_kernel(const float* __restrict__ o_part,
                                    const float* __restrict__ m_part,
                                    const float* __restrict__ l_part,
                                    const int* __restrict__ ctx_lens, __half* __restrict__ out,
                                    int hq, int d, int partition, int num_splits) {
    const size_t row = blockIdx.x;
    const int b = (int)(row / hq);
    const int ns = (ctx_lens[b] + partition - 1) / partition;
    const float* mp = m_part + row * num_splits;
    const float* lp = l_part + row * num_splits;
    float M = -INFINITY;
    for (int s = 0; s < ns; ++s) M = fmaxf(M, mp[s]);
    float num = 0.f, den = 0.f;
    for (int s = 0; s < ns; ++s) {
        const float w = lp[s] * exp2f(mp[s] - M);
        num += w * o_part[(row * num_splits + s) * d + threadIdx.x];
        den += w;
    }
    out[row * d + threadIdx.x] = __float2half(num / den);
}

template <int D, int G>
void launch_decode(const torch::Tensor& q, const torch::Tensor& kc, const torch::Tensor& vc,
                   const torch::Tensor& bt, const torch::Tensor& ctx, torch::Tensor& out,
                   int partition, int num_splits, float scale_log2) {
    const int B = (int)q.size(0), hq = (int)q.size(1);
    const int hkv = (int)kc.size(1), bs = (int)kc.size(2);
    auto stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor o_part, m_part, l_part;
    float *op = nullptr, *mp = nullptr, *lp = nullptr;
    if (num_splits > 1) {
        auto f32 = q.options().dtype(torch::kFloat32);
        o_part = torch::empty({B, hq, num_splits, D}, f32);
        m_part = torch::empty({B, hq, num_splits}, f32);
        l_part = torch::empty({B, hq, num_splits}, f32);
        op = o_part.data_ptr<float>();
        mp = m_part.data_ptr<float>();
        lp = l_part.data_ptr<float>();
    }
    dim3 grid(num_splits, hkv, B);
    paged_decode_kernel<D, G><<<grid, kDecThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(kc.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(vc.data_ptr<at::Half>()), bt.data_ptr<int>(),
        ctx.data_ptr<int>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), op, mp, lp, hkv,
        bs, (int)bt.size(1), partition, num_splits, scale_log2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (num_splits > 1) {
        paged_reduce_kernel<<<B * hq, D, 0, stream>>>(
            op, mp, lp, ctx.data_ptr<int>(), reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            hq, D, partition, num_splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

}  // namespace attn

// partition: tokens per split, a multiple of 256, chosen by the caller (the
// Python wrapper picks it from the batch size and SM count).
torch::Tensor paged_decode(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor bt,
                           torch::Tensor ctx, int64_t max_ctx, double scale, int64_t partition) {
    const int D = (int)q.size(2);
    const int hq = (int)q.size(1), hkv = (int)kc.size(1);
    TORCH_CHECK(hq % hkv == 0, "num_heads must be a multiple of num_kv_heads");
    TORCH_CHECK(partition > 0 && partition % attn::kChunk == 0,
                "partition size must be a positive multiple of 256");
    const int G = hq / hkv;
    const int num_splits = (int)((max_ctx + partition - 1) / partition);
    auto out = torch::empty_like(q);
    const float sl2 = (float)scale * attn::kLog2e;
#define DISPATCH(DD, GG)                                                                     \
    if (D == DD && G == GG) {                                                                \
        attn::launch_decode<DD, GG>(q, kc, vc, bt, ctx, out, (int)partition, num_splits, sl2); \
        return out;                                                                          \
    }
    DISPATCH(64, 1) DISPATCH(64, 2) DISPATCH(64, 4) DISPATCH(64, 8)
    DISPATCH(128, 1) DISPATCH(128, 2) DISPATCH(128, 4) DISPATCH(128, 8)
#undef DISPATCH
    TORCH_CHECK(false, "paged_decode: head_dim must be 64/128 and num_heads/num_kv_heads in {1,2,4,8}");
    return out;
}
