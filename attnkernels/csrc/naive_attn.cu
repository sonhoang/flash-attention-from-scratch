// Naive attention: the textbook three steps, each its own kernel.
//
//   S = scale * Q K^T     (fp32, the full N x N matrix is written to DRAM)
//   P = softmax(S)        (row by row, in place)
//   O = P V
//
// This is the version FlashAttention is measured against. It does the same
// arithmetic, but the N x N scores make two round trips through DRAM and use
// O(N^2) memory, which is what limits it at long sequence lengths.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "common.cuh"

namespace attn {

// One thread per score. Q, K: [BH, N, D] fp16. S: [BH, N, N] fp32.
__global__ void scores_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                              float* __restrict__ s, int n, int d, float scale, bool causal) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;  // key
    const int i = blockIdx.y * blockDim.y + threadIdx.y;  // query
    const int bh = blockIdx.z;
    if (i >= n || j >= n) return;
    float acc = 0.f;
    if (!causal || j <= i) {
        const __half* qi = q + ((size_t)bh * n + i) * d;
        const __half* kj = k + ((size_t)bh * n + j) * d;
        for (int t = 0; t < d; ++t) acc += __half2float(qi[t]) * __half2float(kj[t]);
        acc *= scale;
    } else {
        acc = -INFINITY;
    }
    s[((size_t)bh * n + i) * n + j] = acc;
}

// One block per row of S. Three passes over the row: max, sum of exp, write.
__global__ void softmax_rows_kernel(float* __restrict__ s, int n) {
    float* row = s + (size_t)blockIdx.x * n;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, row[j]);
    m = block_reduce(m, true);
    float sum = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) sum += __expf(row[j] - m);
    sum = block_reduce(sum, false);
    const float inv = 1.f / sum;
    for (int j = threadIdx.x; j < n; j += blockDim.x) row[j] = __expf(row[j] - m) * inv;
}

// One thread per output element. P: [BH, N, N] fp32, V: [BH, N, D] fp16.
__global__ void pv_kernel(const float* __restrict__ p, const __half* __restrict__ v,
                          __half* __restrict__ o, int n, int d, bool causal) {
    const int col = threadIdx.x;
    const int i = blockIdx.x;
    const int bh = blockIdx.y;
    const float* prow = p + ((size_t)bh * n + i) * n;
    const __half* vb = v + (size_t)bh * n * d;
    const int jend = causal ? i + 1 : n;
    float acc = 0.f;
    for (int j = 0; j < jend; ++j) acc += prow[j] * __half2float(vb[(size_t)j * d + col]);
    o[((size_t)bh * n + i) * d + col] = __float2half(acc);
}

}  // namespace attn

torch::Tensor naive_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal,
                              double scale) {
    const int64_t B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
    const int BH = (int)(B * H);
    auto s = torch::empty({B, H, N, N}, q.options().dtype(torch::kFloat32));
    auto o = torch::empty_like(q);
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 blk(32, 8);
    dim3 grd((unsigned)((N + 31) / 32), (unsigned)((N + 7) / 8), (unsigned)BH);
    attn::scores_kernel<<<grd, blk, 0, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()), s.data_ptr<float>(), (int)N,
        (int)D, (float)scale, causal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attn::softmax_rows_kernel<<<(unsigned)(BH * N), 256, 0, stream>>>(s.data_ptr<float>(), (int)N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attn::pv_kernel<<<dim3((unsigned)N, (unsigned)BH), (unsigned)D, 0, stream>>>(
        s.data_ptr<float>(), reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(o.data_ptr<at::Half>()), (int)N, (int)D, causal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}
