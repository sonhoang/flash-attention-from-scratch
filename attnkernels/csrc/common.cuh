// Helpers shared by the attention kernels.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <c10/cuda/CUDAException.h>

namespace attn {

// log2(e). The kernels work in base 2: exp(x) = exp2(x * log2(e)), and the
// softmax scale is folded into the same multiply.
constexpr float kLog2e = 1.4426950408889634f;

__host__ __device__ constexpr int round16(int x) { return (x + 15) / 16 * 16; }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block-wide reductions (every thread gets the result). Safe to call
// repeatedly in one kernel.
__device__ __forceinline__ float block_reduce(float v, bool is_max) {
    __shared__ float partial[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = is_max ? warp_max(v) : warp_sum(v);
    if (lane == 0) partial[wid] = v;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    const float neutral = is_max ? -INFINITY : 0.f;
    v = (threadIdx.x < nwarps) ? partial[threadIdx.x] : neutral;
    if (wid == 0) v = is_max ? warp_max(v) : warp_sum(v);
    if (threadIdx.x == 0) partial[0] = v;
    __syncthreads();
    v = partial[0];
    __syncthreads();
    return v;
}

// 8 halves (16 bytes) to 8 floats.
__device__ __forceinline__ void load8(const __half* p, float* out) {
    uint4 v = *reinterpret_cast<const uint4*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __half22float2(h[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

}  // namespace attn
