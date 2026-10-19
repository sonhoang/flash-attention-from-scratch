// FlashAttention-style forward pass on CUDA cores (no tensor cores).
//
// One thread block owns BR = 64 query rows of one (batch, head). It walks over
// the keys in tiles of BC rows. For each tile it loads K and V into shared
// memory, computes the BR x BC block of scores, and folds them into a running
// softmax (online softmax), so the full N x N score matrix never exists.
//
//   m_new = max(m, rowmax(S))          running max
//   P     = exp(S - m_new)
//   l     = l * exp(m - m_new) + rowsum(P)
//   O     = O * exp(m - m_new) + P V
//
// and at the end O / l. Everything is fp32 inside; Q, K, V and O are fp16.
//
// Threads form a 16 x 16 grid. Thread (ty, tx) owns rows 4*ty .. 4*ty+3 of
// the tile. For the scores it owns columns tx, tx+16, ...; for the output it
// owns head dims tx, tx+16, ... . The 16 threads that share a row are 16
// consecutive lanes of one warp, so row max and row sum are 4 shuffles.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "common.cuh"

namespace attn {

template <int D, int BC>
struct SimtCfg {
    static constexpr int BR = 64;
    static constexpr int THREADS = 256;
    static constexpr int RM = BR / 16;  // query rows per thread
    static constexpr int CN = BC / 16;  // score columns per thread
    static constexpr int DN = D / 16;   // output dims per thread
    static constexpr int LDQ = D + 2;   // padded rows: no bank conflicts between the 2 row groups of a warp
    static constexpr int LDKT = BC + 2;
    static constexpr int LDS = BC + 1;
    static constexpr int Q_BYTES = BR * LDQ * 2;
    static constexpr int KT_BYTES = D * LDKT * 2;
    static constexpr int V_BYTES = BC * D * 2;
    static constexpr int S_BYTES = BR * LDS * 4;
    static constexpr int KT_OFF = round16(Q_BYTES);
    static constexpr int V_OFF = KT_OFF + round16(KT_BYTES);
    static constexpr int S_OFF = V_OFF + round16(V_BYTES);
    static constexpr int SMEM = S_OFF + S_BYTES;
};

template <int D, int BC>
__global__ void __launch_bounds__(256)
flash_simt_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                  const __half* __restrict__ V, __half* __restrict__ O, int n, float scale_log2,
                  bool causal) {
    using C = SimtCfg<D, BC>;
    extern __shared__ __align__(16) unsigned char smem[];
    __half* Qs = reinterpret_cast<__half*>(smem);
    __half* Kt = reinterpret_cast<__half*>(smem + C::KT_OFF);  // K tile, transposed: [D][BC]
    __half* Vs = reinterpret_cast<__half*>(smem + C::V_OFF);   // V tile: [BC][D]
    float* Ss = reinterpret_cast<float*>(smem + C::S_OFF);      // P tile: [BR][BC]

    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int q0 = blockIdx.x * C::BR;
    const size_t base = (size_t)blockIdx.y * n * D;
    const __half* qb = Q + base;
    const __half* kb = K + base;
    const __half* vb = V + base;
    constexpr int V8 = D / 8;  // 16-byte vectors per row

    // Q tile -> shared memory (zero rows past the end).
    for (int idx = tid; idx < C::BR * V8; idx += C::THREADS) {
        const int r = idx / V8, c8 = idx % V8;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (q0 + r < n) val = *reinterpret_cast<const uint4*>(qb + (size_t)(q0 + r) * D + c8 * 8);
        const __half2* h = reinterpret_cast<const __half2*>(&val);
        __half2* dst = reinterpret_cast<__half2*>(Qs + r * C::LDQ + c8 * 8);
#pragma unroll
        for (int e = 0; e < 4; ++e) dst[e] = h[e];
    }

    float o[C::RM][C::DN];
    float m[C::RM], l[C::RM];
#pragma unroll
    for (int i = 0; i < C::RM; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.f;
#pragma unroll
        for (int j = 0; j < C::DN; ++j) o[i][j] = 0.f;
    }

    const int kend = causal ? min(n, q0 + C::BR) : n;
    const int ntiles = (kend + BC - 1) / BC;
    for (int t = 0; t < ntiles; ++t) {
        const int k0 = t * BC;
        __syncthreads();  // previous tile fully consumed
        for (int idx = tid; idx < BC * V8; idx += C::THREADS) {
            const int c = idx / V8, c8 = idx % V8;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (k0 + c < n) {
                kv = *reinterpret_cast<const uint4*>(kb + (size_t)(k0 + c) * D + c8 * 8);
                vv = *reinterpret_cast<const uint4*>(vb + (size_t)(k0 + c) * D + c8 * 8);
            }
            const __half* kh = reinterpret_cast<const __half*>(&kv);
#pragma unroll
            for (int e = 0; e < 8; ++e) Kt[(c8 * 8 + e) * C::LDKT + c] = kh[e];
            *reinterpret_cast<uint4*>(Vs + c * D + c8 * 8) = vv;
        }
        __syncthreads();

        // S = Q K^T for this thread's 4 x CN block.
        float s[C::RM][C::CN];
#pragma unroll
        for (int i = 0; i < C::RM; ++i)
#pragma unroll
            for (int j = 0; j < C::CN; ++j) s[i][j] = 0.f;
#pragma unroll 4
        for (int kk = 0; kk < D; kk += 2) {
            float2 qv[C::RM];
#pragma unroll
            for (int i = 0; i < C::RM; ++i)
                qv[i] = __half22float2(
                    *reinterpret_cast<const __half2*>(Qs + (ty * C::RM + i) * C::LDQ + kk));
#pragma unroll
            for (int j = 0; j < C::CN; ++j) {
                const float k_a = __half2float(Kt[kk * C::LDKT + tx + 16 * j]);
                const float k_b = __half2float(Kt[(kk + 1) * C::LDKT + tx + 16 * j]);
#pragma unroll
                for (int i = 0; i < C::RM; ++i) s[i][j] += qv[i].x * k_a + qv[i].y * k_b;
            }
        }

        // Online softmax update, row by row.
#pragma unroll
        for (int i = 0; i < C::RM; ++i) {
            const int qr = q0 + ty * C::RM + i;
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < C::CN; ++j) {
                const int kc = k0 + tx + 16 * j;
                float x = s[i][j] * scale_log2;
                if (kc >= n || (causal && kc > qr)) x = -INFINITY;
                s[i][j] = x;
                mx = fmaxf(mx, x);
            }
#pragma unroll
            for (int off = 8; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float m_new = fmaxf(m[i], mx);
            const float m_use = (m_new == -INFINITY) ? 0.f : m_new;
            const float alpha = exp2f(m[i] - m_use);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < C::CN; ++j) {
                const float p = exp2f(s[i][j] - m_use);
                sum += p;
                Ss[(ty * C::RM + i) * C::LDS + tx + 16 * j] = p;
            }
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
            l[i] = l[i] * alpha + sum;
            m[i] = m_new;
#pragma unroll
            for (int j = 0; j < C::DN; ++j) o[i][j] *= alpha;
        }
        __syncthreads();

        // O += P V
#pragma unroll 4
        for (int c = 0; c < BC; ++c) {
            float p[C::RM];
#pragma unroll
            for (int i = 0; i < C::RM; ++i) p[i] = Ss[(ty * C::RM + i) * C::LDS + c];
#pragma unroll
            for (int j = 0; j < C::DN; ++j) {
                const float v = __half2float(Vs[c * D + tx + 16 * j]);
#pragma unroll
                for (int i = 0; i < C::RM; ++i) o[i][j] += p[i] * v;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < C::RM; ++i) {
        const int qr = q0 + ty * C::RM + i;
        if (qr >= n) continue;
        const float inv = l[i] > 0.f ? 1.f / l[i] : 0.f;
#pragma unroll
        for (int j = 0; j < C::DN; ++j)
            O[base + (size_t)qr * D + tx + 16 * j] = __float2half(o[i][j] * inv);
    }
}

template <int D, int BC>
void launch_simt(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                 torch::Tensor& o, float scale_log2, bool causal) {
    using C = SimtCfg<D, BC>;
    const int n = (int)q.size(2);
    const int bh = (int)(q.size(0) * q.size(1));
    auto kern = flash_simt_kernel<D, BC>;
    if (C::SMEM > 48 * 1024)
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    dim3 grid((n + C::BR - 1) / C::BR, bh);
    kern<<<grid, C::THREADS, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(o.data_ptr<at::Half>()), n, scale_log2, causal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace attn

torch::Tensor flash_simt(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
                         double scale) {
    auto o = torch::empty_like(q);
    const float sl2 = (float)scale * attn::kLog2e;
    const int64_t D = q.size(3);
    if (D == 64)
        attn::launch_simt<64, 64>(q, k, v, o, sl2, causal);
    else if (D == 128)
        attn::launch_simt<128, 32>(q, k, v, o, sl2, causal);
    else
        TORCH_CHECK(false, "flash_simt: head_dim must be 64 or 128");
    return o;
}
