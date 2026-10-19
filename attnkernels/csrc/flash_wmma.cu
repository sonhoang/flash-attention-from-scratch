// FlashAttention-style forward pass on tensor cores (WMMA, sm_70+).
//
// Same algorithm as flash_simt.cu, but the two matrix products per tile
// (S = Q K^T and O += P V) run on tensor cores through the WMMA API with
// 16 x 16 x 16 fp16 tiles and fp32 accumulators.
//
// A block has 4 warps and owns BR = 64 query rows; each warp owns 16 of them,
// so warps never share a softmax row. Per warp:
//   * its 16 x D slice of Q lives in registers as WMMA A-fragments for the
//     whole kernel (loaded once, reused for every key tile);
//   * its 16 x D slice of O lives in registers as fp32 accumulator fragments;
//   * K and V tiles (BC x D) are shared by the 4 warps through shared memory.
//
// The softmax is not done on fragments: S goes to a per-warp scratch buffer
// in shared memory, two lanes handle each row (max, exp, sum via one shuffle),
// and P is written back as fp16 in place, ready to be loaded as an A-fragment.
//
// The one awkward part is rescaling O by exp(m_old - m_new) per row, because
// the mapping from accumulator-fragment elements to matrix rows is not
// specified by CUDA. We discover it at run time: each warp loads a 16 x 16
// matrix whose entry (r, c) is r into an accumulator fragment, and element i
// of that fragment then tells us the row of element i of every accumulator
// fragment of the same type. No architecture-specific layout is assumed.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>

#include "common.cuh"

namespace attn {

using namespace nvcuda;

template <int D, int BC>
struct WmmaCfg {
    static constexpr int BR = 64;
    static constexpr int NW = 4;
    static constexpr int THREADS = NW * 32;
    static constexpr int KD = D / 16;    // fragments along the head dim
    static constexpr int NC = BC / 16;   // fragments along a key tile
    static constexpr int LDK = D + 8;    // halves; WMMA needs a multiple of 8
    static constexpr int LDS = BC + 4;   // floats; WMMA needs a multiple of 4
    static constexpr int LDP = 2 * LDS;  // P (fp16) reuses S's rows
    static constexpr int K_OFF = 0;
    static constexpr int V_OFF = K_OFF + BC * LDK * 2;
    static constexpr int S_OFF = V_OFF + BC * LDK * 2;
    static constexpr int STAT_OFF = S_OFF + NW * 16 * LDS * 4;
    static constexpr int SMEM = STAT_OFF + NW * 16 * 4;
    static_assert(2 * BC >= BR, "Q is staged through the K and V buffers");
    static_assert(KD % NC == 0, "output is written NC fragments at a time");
};

template <int D, int BC>
__global__ void __launch_bounds__(128)
flash_wmma_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                  const __half* __restrict__ V, __half* __restrict__ O, int n, float scale_log2,
                  bool causal) {
    using C = WmmaCfg<D, BC>;
    using AFrag = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
    using BColFrag = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;
    using BRowFrag = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
    using AccFrag = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    constexpr int NE = AccFrag::num_elements;

    extern __shared__ __align__(128) unsigned char smem[];
    __half* Ks = reinterpret_cast<__half*>(smem + C::K_OFF);
    __half* Vs = reinterpret_cast<__half*>(smem + C::V_OFF);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float* Sw = reinterpret_cast<float*>(smem + C::S_OFF) + warp * 16 * C::LDS;
    __half* Pw = reinterpret_cast<__half*>(Sw);
    float* stat = reinterpret_cast<float*>(smem + C::STAT_OFF) + warp * 16;

    const int q0 = blockIdx.x * C::BR;
    const int wrow0 = q0 + warp * 16;  // first query row of this warp
    const size_t base = (size_t)blockIdx.y * n * D;
    const __half* qb = Q + base;
    const __half* kb = K + base;
    const __half* vb = V + base;
    constexpr int V8 = D / 8;

    // Stage the Q tile through the K/V buffers, then keep it in registers.
    for (int idx = tid; idx < C::BR * V8; idx += C::THREADS) {
        const int r = idx / V8, c8 = idx % V8;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (q0 + r < n) val = *reinterpret_cast<const uint4*>(qb + (size_t)(q0 + r) * D + c8 * 8);
        *reinterpret_cast<uint4*>(Ks + r * C::LDK + c8 * 8) = val;
    }
    // Row-index matrix for discovering the accumulator layout.
    for (int idx = lane; idx < 256; idx += 32) Sw[(idx >> 4) * C::LDS + (idx & 15)] = (float)(idx >> 4);
    __syncthreads();
    AFrag qa[C::KD];
#pragma unroll
    for (int kk = 0; kk < C::KD; ++kk)
        wmma::load_matrix_sync(qa[kk], Ks + warp * 16 * C::LDK + kk * 16, C::LDK);
    int rows[NE];
    {
        AccFrag rf;
        wmma::load_matrix_sync(rf, Sw, C::LDS, wmma::mem_row_major);
#pragma unroll
        for (int i = 0; i < NE; ++i) rows[i] = (int)rf.x[i];
    }

    AccFrag o[C::KD];
#pragma unroll
    for (int f = 0; f < C::KD; ++f) wmma::fill_fragment(o[f], 0.f);

    // Softmax mapping: lane handles row r, columns hf, hf+2, hf+4, ...
    const int r = lane >> 1, hf = lane & 1;
    const int qr = wrow0 + r;
    float m_i = -INFINITY, l_i = 0.f;

    const int kend = causal ? min(n, q0 + C::BR) : n;
    const int ntiles = (kend + BC - 1) / BC;
    for (int t = 0; t < ntiles; ++t) {
        const int k0 = t * BC;
        __syncthreads();  // all warps done with the previous K/V tile (and with Q staging)
        for (int idx = tid; idx < BC * V8; idx += C::THREADS) {
            const int c = idx / V8, c8 = idx % V8;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (k0 + c < n) {
                kv = *reinterpret_cast<const uint4*>(kb + (size_t)(k0 + c) * D + c8 * 8);
                vv = *reinterpret_cast<const uint4*>(vb + (size_t)(k0 + c) * D + c8 * 8);
            }
            *reinterpret_cast<uint4*>(Ks + c * C::LDK + c8 * 8) = kv;
            *reinterpret_cast<uint4*>(Vs + c * C::LDK + c8 * 8) = vv;
        }
        __syncthreads();
        // Causal: a tile whose first key is after this warp's last row is all masked.
        if (causal && k0 > wrow0 + 15) continue;

        // S = Q K^T (16 x BC per warp) -> scratch.
#pragma unroll
        for (int j = 0; j < C::NC; ++j) {
            AccFrag s;
            wmma::fill_fragment(s, 0.f);
#pragma unroll
            for (int kk = 0; kk < C::KD; ++kk) {
                BColFrag kf;
                wmma::load_matrix_sync(kf, Ks + j * 16 * C::LDK + kk * 16, C::LDK);
                wmma::mma_sync(s, qa[kk], kf, s);
            }
            wmma::store_matrix_sync(Sw + j * 16, s, C::LDS, wmma::mem_row_major);
        }
        __syncwarp();

        // Online softmax on the scratch rows.
        float sv[BC / 2];
        float mx = -INFINITY;
#pragma unroll
        for (int jj = 0; jj < BC / 2; ++jj) {
            const int c = hf + 2 * jj, kc = k0 + c;
            float x = Sw[r * C::LDS + c] * scale_log2;
            if (kc >= n || (causal && kc > qr)) x = -INFINITY;
            sv[jj] = x;
            mx = fmaxf(mx, x);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        const float m_new = fmaxf(m_i, mx);
        const float m_use = (m_new == -INFINITY) ? 0.f : m_new;
        const float alpha = exp2f(m_i - m_use);
        float sum = 0.f;
#pragma unroll
        for (int jj = 0; jj < BC / 2; ++jj) {
            sv[jj] = exp2f(sv[jj] - m_use);
            sum += sv[jj];
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        l_i = l_i * alpha + sum;
        m_i = m_new;
        __syncwarp();  // every lane has read S before P overwrites it
#pragma unroll
        for (int jj = 0; jj < BC / 2; ++jj) Pw[r * C::LDP + hf + 2 * jj] = __float2half(sv[jj]);
        if (hf == 0) stat[r] = alpha;
        __syncwarp();

        // O = O * alpha + P V
#pragma unroll
        for (int f = 0; f < C::KD; ++f)
#pragma unroll
            for (int i = 0; i < NE; ++i) o[f].x[i] *= stat[rows[i]];
        AFrag pa[C::NC];
#pragma unroll
        for (int kk = 0; kk < C::NC; ++kk) wmma::load_matrix_sync(pa[kk], Pw + kk * 16, C::LDP);
#pragma unroll
        for (int f = 0; f < C::KD; ++f) {
#pragma unroll
            for (int kk = 0; kk < C::NC; ++kk) {
                BRowFrag vf;
                wmma::load_matrix_sync(vf, Vs + kk * 16 * C::LDK + f * 16, C::LDK);
                wmma::mma_sync(o[f], pa[kk], vf, o[f]);
            }
        }
        __syncwarp();
    }

    // O / l, then out through the scratch buffer, NC fragments at a time.
    if (hf == 0) stat[r] = l_i > 0.f ? 1.f / l_i : 0.f;
    __syncwarp();
#pragma unroll
    for (int f = 0; f < C::KD; ++f)
#pragma unroll
        for (int i = 0; i < NE; ++i) o[f].x[i] *= stat[rows[i]];
    constexpr int COLS = C::NC * 16;
#pragma unroll
    for (int f0 = 0; f0 < C::KD; f0 += C::NC) {
#pragma unroll
        for (int f = 0; f < C::NC; ++f)
            wmma::store_matrix_sync(Sw + f * 16, o[f0 + f], C::LDS, wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < 16 * COLS / 2; idx += 32) {
            const int rr = idx / (COLS / 2), cc = (idx % (COLS / 2)) * 2;
            const int row = wrow0 + rr;
            if (row < n)
                *reinterpret_cast<__half2*>(O + base + (size_t)row * D + f0 * 16 + cc) =
                    __floats2half2_rn(Sw[rr * C::LDS + cc], Sw[rr * C::LDS + cc + 1]);
        }
        __syncwarp();
    }
}

template <int D, int BC>
void launch_wmma(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                 torch::Tensor& o, float scale_log2, bool causal) {
    using C = WmmaCfg<D, BC>;
    const int n = (int)q.size(2);
    const int bh = (int)(q.size(0) * q.size(1));
    auto kern = flash_wmma_kernel<D, BC>;
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

torch::Tensor flash_wmma(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
                         double scale, int64_t block_n) {
    auto o = torch::empty_like(q);
    const float sl2 = (float)scale * attn::kLog2e;
    const int64_t D = q.size(3);
    if (D == 64 && block_n == 32)
        attn::launch_wmma<64, 32>(q, k, v, o, sl2, causal);
    else if (D == 64 && block_n == 64)
        attn::launch_wmma<64, 64>(q, k, v, o, sl2, causal);
    else if (D == 128 && block_n == 32)
        attn::launch_wmma<128, 32>(q, k, v, o, sl2, causal);
    else if (D == 128 && block_n == 64)
        attn::launch_wmma<128, 64>(q, k, v, o, sl2, causal);
    else
        TORCH_CHECK(false, "flash_wmma: head_dim must be 64 or 128 and block_n 32 or 64");
    return o;
}
