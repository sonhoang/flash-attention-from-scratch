# flash-attention-from-scratch

Attention kernels for LLM inference, written by hand in CUDA C++ and measured
on an NVIDIA T4:

- **prefill**: a naive three-kernel reference, a FlashAttention-style tiled
  kernel with online softmax on CUDA cores, and the same algorithm on tensor
  cores through WMMA;
- **decode**: attention over a vLLM-style paged KV cache, with split-K
  ("flash-decoding") parallelism over the context and a reduce kernel.

Everything is exposed to PyTorch as an extension, checked against an fp32
reference with pytest, and benchmarked against
`torch.nn.functional.scaled_dot_product_attention` (SDPA). This README is
written as a short report: why the kernels look the way they do, what they
measure, and where PyTorch is still faster.

## Contents

1. [Why attention needs its own kernels](#why-attention-needs-its-own-kernels)
2. [Online softmax in a few lines](#online-softmax-in-a-few-lines)
3. [The kernels](#the-kernels)
4. [Results on a T4](#results-on-a-t4)
5. [Where PyTorch wins, and why](#where-pytorch-wins-and-why)
6. [Limitations](#limitations)
7. [What would change on Ampere and Hopper](#what-would-change-on-ampere-and-hopper)
8. [Reproducing](#reproducing)

## Why attention needs its own kernels

For one head, attention is `O = softmax(Q K^T / sqrt(d)) V` with Q, K, V of
shape N x d. Written as three PyTorch ops it is two matrix multiplies and a
softmax, and that is exactly what the naive kernel here does. The problem is
the middle tensor. `S = Q K^T` is N x N: at N = 8192 it is 64 M entries per
head, 256 MB in fp32, for a result O that is only 1 MB (fp16, d = 64).

Count the DRAM traffic. The matrix multiplies do 4 N^2 d FLOPs in total. The
naive version writes S, reads it back for the softmax, writes P, and reads P
again for P V: about 4 N^2 values moved for those FLOPs. With d = 64 that is
roughly 64 FLOPs per fp32 value moved, 16 FLOPs per byte. A T4 does about
65 TFLOP/s on fp16 tensor cores but only moves about 320 GB/s, so to keep the
tensor cores busy a kernel needs about 200 FLOPs per byte. The naive
version is memory-bound by more than 10x, and it also needs O(N^2) memory.

FlashAttention (Dao et al., 2022) removes the N x N tensor. It tiles Q into
blocks of rows and streams K and V through on-chip memory in blocks of
columns, so S only ever exists one tile at a time in shared memory and
registers. DRAM traffic drops to reading Q, K, V once per query block and
writing O once. The catch is the softmax: each output row needs the max and
the sum over the *whole* row of S, which is not known until the last tile.
Online softmax solves that.

Decode is a different regime. One new token per sequence means Q is a single
row, so there are only about 2 FLOPs per K/V element read. No amount of
tiling makes that compute-bound: the best a decode kernel can do is stream
the KV cache at close to DRAM bandwidth, which is why the decode results
below are in GB/s, not TFLOP/s.

## Online softmax in a few lines

For one query row with scores s_1 ... s_N, the output is

    o = sum_j exp(s_j - m) v_j / l,   m = max_j s_j,   l = sum_j exp(s_j - m)

(subtracting m changes nothing mathematically and keeps exp from
overflowing). Process the keys in blocks. Keep a running max m, a running
sum l and an unnormalised accumulator a. When block B arrives with scores
s_B:

    m' = max(m, max(s_B))
    l' = l * exp(m - m') + sum_{j in B} exp(s_j - m')
    a' = a * exp(m - m') + sum_{j in B} exp(s_j - m') v_j

Every earlier term in l and a was computed relative to m; multiplying by
exp(m - m') moves it to the new reference m', so after the last block
`a / l` is exactly the softmax-weighted sum. The cost is one rescale of the
accumulator per block. Two partial results (m1, l1, a1) and (m2, l2, a2) merge
the same way, which is what the split-K decode reduce kernel does.

All kernels here work in base 2: `exp(x) = exp2(x * log2 e)`, with
`1/sqrt(d) * log2 e` folded into one multiply, because `exp2` is a single
hardware instruction (MUFU.EX2).

## The kernels

All kernels take fp16 Q, K, V and accumulate in fp32. Layout is
`[batch, heads, seq, head_dim]`, head_dim 64 or 128, causal or not. Sources are
in [attnkernels/csrc](attnkernels/csrc).

### Naive ([naive_attn.cu](attnkernels/csrc/naive_attn.cu))

Three kernels: one thread per score for `S = scale * Q K^T` (fp32, written to
DRAM), one block per row for an in-place softmax, one thread per output
element for `O = P V`. It is the honest "what the math says" baseline, and
it is the version that runs out of memory first.

### Tiled, CUDA cores ([flash_simt.cu](attnkernels/csrc/flash_simt.cu))

One block of 256 threads owns 64 query rows of one (batch, head). The Q tile
sits in shared memory for the whole kernel. For each key tile (64 keys for
d = 64, 32 for d = 128, to stay under 48 KB of shared memory) the block loads
K transposed and V into shared memory, then:

1. computes its 64 x BC block of scores; threads form a 16 x 16 grid and
   each holds a 4 x (BC/16) register tile;
2. updates the running max and sum; the 16 threads that share a row are 16
   consecutive lanes, so a row reduction is 4 `__shfl_xor_sync` steps;
3. rescales its slice of O (4 rows x d/16 dims, in registers) and adds P V,
   with P going through shared memory.

Causal masking skips key tiles entirely above the diagonal and masks inside
the diagonal tile. Rows and keys past the end of an odd-length sequence are
zero-filled and masked.

### Tiled, tensor cores ([flash_wmma.cu](attnkernels/csrc/flash_wmma.cu))

Same algorithm with the two matrix products on tensor cores through the
WMMA API (16 x 16 x 16 fp16 fragments, fp32 accumulators), which is the
tensor-core interface sm_75 supports. A block has 4 warps and 64 query rows,
16 per warp, so no softmax row is shared between warps:

- each warp's 16 x d slice of Q is loaded once into WMMA A-fragments and stays
  in registers;
- its 16 x d slice of O lives in fp32 accumulator fragments;
- the K and V tiles (BC x d) are shared by the 4 warps in shared memory.

Per key tile a warp computes `S = Q K^T` (16 x BC) on tensor cores, stores it
to a per-warp scratch buffer, does the online-softmax update there (two lanes
per row, one shuffle per reduction), writes P back as fp16 *in place*, loads
P as A-fragments and accumulates `O += P V` on tensor cores.

One detail took some thought. Rescaling O by `exp(m - m')` is per row, but
CUDA does not specify which matrix element each thread holds in an
accumulator fragment. Instead of hard-coding an architecture's layout, every
warp loads a 16 x 16 matrix with entry (r, c) = r into an accumulator
fragment once at the start; element i of that fragment then *is* the row of
element i in every accumulator fragment of the same type. Rescaling becomes
`o.x[i] *= alpha[row[i]]`, portable across GPUs.

The key-tile width BC is a template parameter (32 or 64); both are measured.
BC = 64 halves the number of block-wide syncs but uses more shared memory,
which on Turing (64 KB per SM) means fewer resident blocks.

### Paged decode ([paged_decode.cu](attnkernels/csrc/paged_decode.cu))

The KV cache is a pool of fixed-size blocks,
`k_cache, v_cache: [num_blocks, kv_heads, block_size, head_dim]`, and each
sequence has a block table mapping its logical block i to a physical block,
like vLLM's PagedAttention. Context lengths differ per sequence.

With one query token per sequence, a batch of 4 sequences with 8 kv heads is
only 32 (sequence, head) pairs: not enough thread blocks for 40 SMs, and each
block would walk a 32k-token context alone. Split-K (Dao et al.'s
flash-decoding) cuts the context into partitions and runs one block per
(partition, kv head, sequence). Each block writes its partial output
(normalised by its own sum) together with its max and sum; a second kernel
merges the partials with weights `l_s * exp(m_s - max_s m_s)`. The
partition size is chosen so the grid has about 4 blocks per SM, never less
than one 256-token chunk per split.

Inside a block:

- grouped-query attention is handled by giving one block all G query heads
  that share a kv head, so each K/V byte is read once (G = 4 for the
  Llama-3-8B shape benchmarked: 32 query heads, 8 kv heads);
- scores: 8 lanes per token, 16-byte loads (a 128-dim fp16 key is two
  128-byte coalesced rows per 8 lanes), 3 shuffles per dot product;
- the partition is processed in 256-token chunks with online softmax
  between chunks, so any partition size works with a fixed 20 KB of shared
  memory;
- `P V`: d/8 threads cover one V row with 16-byte loads, the remaining
  threads take other tokens, and partial sums are reduced with shuffles and
  shared memory at the end.

## Results on a T4

These numbers come from one **short run** on a Kaggle T4: prefill at sequence
length 512 and 1024, decode at context 1k and 4k, plus a split-size sweep at
8k and a ragged batch with up to 6.9k tokens. The full sweep (prefill up to
8192, decode up to 32k) is written in `bench/bench.py` and is queued on Kaggle;
this section will be replaced with it when it has run. Everything below was
measured; nothing is extrapolated.

Every table is generated by [bench/report.py](bench/report.py) from
[results/t4.json](results/t4.json). T4 timings on Kaggle move by up to about
30% between sessions, so compare implementations within one run.

### Summary

- **Correctness.** 192 pytest cases pass against an fp32 reference, including
  causal masks, odd lengths and ragged paged batches
  ([results/pytest.txt](results/pytest.txt)). The worst prefill error is
  9.85e-4, the same as PyTorch's own SDPA.
- **Prefill, head_dim 128: the WMMA kernel beats PyTorch.** 16.2 to 16.5
  TFLOP/s non-causal against 13.6 to 13.8 for SDPA (about 1.2x), and 14.0 to
  15.1 against 11.8 to 12.3 with a causal mask (about 1.2x).
- **Prefill, head_dim 64: PyTorch is level or ahead.** Non-causal the WMMA
  kernel matches SDPA at 512 tokens (13.4 vs 13.4) and trails at 1024 (13.6 vs
  15.4); causal, SDPA leads by 13 to 25%.
- **On a T4, PyTorch has no FlashAttention.** SDPA reports FLASH_ATTENTION and
  CUDNN_ATTENTION as unavailable on sm_75 and runs its memory-efficient kernel,
  so that is the real comparison here.
- **Memory.** The tiled kernels need only the output (8 to 16 MB extra), like
  SDPA. The naive version needs 136 to 272 MB, and SDPA's math backend 352 to
  704 MB, because they write the N x N score matrix.
- **Tensor cores are the whole story.** The same online-softmax algorithm on CUDA cores
  (flash_simt) reaches about 3 TFLOP/s. On tensor cores it reaches 13 to 16.
  The naive three-kernel version manages 0.11 TFLOP/s.
- **Decode, one sequence: split-K matters most.** At batch 1 and 4k context the split
  kernel reads the KV cache at 183 GB/s (0.092 ms), 2.6x faster than SDPA on a
  contiguous cache and 6.2x faster than gathering the pages and calling SDPA.
  Without split-K the same kernel gets 29 GB/s: 8 thread blocks cannot fill 40
  SMs.
- **Decode, batch 16: PyTorch on a contiguous cache is still faster** (242 to
  257 GB/s against 217 to 223 for the paged kernel without split). The paged
  kernel pays for the block-table indirection and smaller loads. Gathering
  pages first and then calling SDPA is 4.5 to 5x slower than either.
- **Ragged batches** (8 sequences, mean 4.9k, max 6.9k tokens): 216 GB/s of
  useful bytes with split-K, 6.6x the gather + SDPA route.
- **Split size.** At batch 1 and 8k context, 512-token partitions are best
  (201 GB/s); with one partition the kernel drops to 29 GB/s.

![Prefill throughput](results/figures/prefill_tflops.png)

In the prefill chart the two tensor-core variants share a colour: dashed is
the 32-key tile (BC = 32), solid the 64-key tile.

![Paged decode bandwidth](results/figures/decode_gbps.png)

![Split-K partition sweep](results/figures/decode_split_ablation.png)

![Prefill memory](results/figures/prefill_memory.png)

GPU: Tesla T4 (sm_75, 40 SMs, 14.6 GB), driver 580.178.04, PyTorch 2.11.0+cu128, CUDA 12.8, measured 2026-10-19 10:54 UTC.

Measured ceilings: device-to-device copy 243.1 GB/s, fp16 GEMM 8192^3 (cuBLAS) 24.96 TFLOP/s.

SDPA backends on this GPU: FLASH_ATTENTION: unavailable, EFFICIENT_ATTENTION: ok, CUDNN_ATTENTION: unavailable, MATH: ok.

### Prefill TFLOP/s, head_dim 64, non-causal, 16 heads

| seq | batch | sdpa | sdpa_efficient | sdpa_math | naive | flash_simt | flash_wmma_bn32 | flash_wmma_bn64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 13.36 | 13.54 | 1.26 | 0.11 | 2.96 | 13.41 | 9.53 |
| 1024 | 4 | 15.44 | 13.71 | 1.32 | 0.11 | 2.98 | 13.61 | 10.92 |

### Prefill TFLOP/s, head_dim 64, causal, 16 heads

| seq | batch | sdpa | sdpa_efficient | sdpa_math | naive | flash_simt | flash_wmma_bn32 | flash_wmma_bn64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 13.86 | 13.04 | 0.54 | 0.11 | 2.71 | 11.11 | 7.99 |
| 1024 | 4 | 14.31 | 12.27 | 0.53 | 0.11 | 2.74 | 12.70 | 9.38 |

### Prefill TFLOP/s, head_dim 128, non-causal, 16 heads

| seq | batch | sdpa | sdpa_efficient | sdpa_math | naive | flash_simt | flash_wmma_bn32 | flash_wmma_bn64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 13.79 | 12.90 | 1.91 | 0.11 | 2.88 | 16.21 | 13.03 |
| 1024 | 4 | 13.56 | 13.56 | 2.09 | 0.11 | 2.91 | 16.46 | 14.57 |

### Prefill TFLOP/s, head_dim 128, causal, 16 heads

| seq | batch | sdpa | sdpa_efficient | sdpa_math | naive | flash_simt | flash_wmma_bn32 | flash_wmma_bn64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 11.78 | 10.15 | 0.87 | 0.11 | 2.49 | 13.99 | 10.66 |
| 1024 | 4 | 12.32 | 11.52 | 0.87 | 0.11 | 2.66 | 15.09 | 12.74 |

### Prefill peak extra memory (MB), head_dim 64, non-causal (includes the output)

| seq | batch | sdpa | sdpa_efficient | sdpa_math | naive | flash_simt | flash_wmma_bn32 | flash_wmma_bn64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 8 | 8 | 352 | 136 | 8 | 8 | 8 |
| 1024 | 4 | 8 | 8 | 640 | 264 | 8 | 8 | 8 |

### Prefill peak extra memory (MB), head_dim 128, non-causal (includes the output)

| seq | batch | sdpa | sdpa_efficient | sdpa_math | naive | flash_simt | flash_wmma_bn32 | flash_wmma_bn64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 16 | 16 | 416 | 144 | 16 | 16 | 16 |
| 1024 | 4 | 16 | 16 | 704 | 272 | 16 | 16 | 16 |

### Prefill max abs error vs fp32 (worst case over all shapes)

| impl | max abs error |
|---|---:|
| sdpa | 9.85e-04 |
| sdpa_efficient | 9.85e-04 |
| sdpa_math | 9.68e-04 |
| naive | 9.68e-04 |
| flash_simt | 9.68e-04 |
| flash_wmma_bn32 | 9.85e-04 |
| flash_wmma_bn64 | 9.85e-04 |

### Decode GB/s (K+V bytes / time), 32 query heads, 8 kv heads, head_dim 128

| batch | ctx | paged_split | paged_nosplit | torch_gather_sdpa | torch_sdpa_contiguous |
|---|---|---:|---:|---:|---:|
| 1 | 1024 | 77.4 | 26.7 | 27.1 | 61.4 |
| 1 | 4096 | 183.3 | 28.8 | 29.7 | 70.2 |
| 16 | 1024 | 195.4 | 222.6 | 43.8 | 241.8 |
| 16 | 4096 | 202.9 | 216.6 | 44.8 | 257.1 |

### Decode time (ms)

| batch | ctx | paged_split | paged_nosplit | torch_gather_sdpa | torch_sdpa_contiguous |
|---|---|---:|---:|---:|---:|
| 1 | 1024 | 0.0542 | 0.1568 | 0.1549 | 0.0683 |
| 1 | 4096 | 0.0916 | 0.5818 | 0.5651 | 0.2390 |
| 16 | 1024 | 0.3434 | 0.3015 | 1.5339 | 0.2775 |
| 16 | 4096 | 1.3228 | 1.2392 | 5.9947 | 1.0440 |

### Split-K partition size

| batch | ctx | partition | splits | blocks | ms | GB/s |
|---|---|---:|---:|---:|---:|---:|
| 1 | 8192 | 256 | 32 | 256 | 0.2124 | 158.0 |
| 1 | 8192 | 512 | 16 | 128 | 0.1673 | 200.6 |
| 1 | 8192 | 1024 | 8 | 64 | 0.2079 | 161.4 |
| 1 | 8192 | 2048 | 4 | 32 | 0.3173 | 105.7 |
| 1 | 8192 | 4096 | 2 | 16 | 0.5889 | 57.0 |
| 1 | 8192 | 8192 | 1 | 8 | 1.1422 | 29.4 |

### Ragged batch: 8 sequences, mean context 4904, max 6860

| impl | ms | GB/s (useful bytes) |
|---|---:|---:|
| paged_split | 0.7446 | 215.8 |
| paged_nosplit | 1.2467 | 128.9 |
| torch_gather_sdpa | 4.8942 | 32.8 |

## Where PyTorch wins, and why

- **head_dim 64 prefill.** With d = 64 each K/V tile carries half the math of
  d = 128 for the same number of block-wide syncs and softmax updates, so the
  fixed per-tile overhead weighs more. PyTorch's memory-efficient kernel
  (from xformers/CUTLASS) pipelines its loads and uses tuned tile shapes per
  head size; mine use one tile shape for both. BC = 64 should help here and
  does not: on Turing the extra shared memory costs resident blocks.
- **Causal prefill at d = 64.** The tiled kernel stops at the diagonal and
  lets a warp skip a tile that is fully masked for its rows, but diagonal
  tiles still run the full WMMA multiply and apply the mask afterwards. Warps
  that skip a tile still wait at the block-wide syncs, and blocks for early
  query rows finish long before blocks for late rows. I have not profiled
  which of these costs most; that is the next thing to check.
- **Decode with many sequences on a contiguous cache.** With 16 sequences x 8
  kv heads there is already enough parallelism, so split-K only adds a reduce
  pass (paged_split is slower than paged_nosplit there), and SDPA on a flat
  cache gets long, perfectly coalesced reads with no indirection. The paged
  layout exists to stop memory fragmentation under real serving load, and
  this benchmark does not measure that benefit.

## Limitations

- **Short run only, so far.** Prefill at 512 and 1024 tokens, decode at 1k and 4k
  (plus the 8k sweep and the ragged batch). Longer sequences, where the N x N
  matrix of the naive path stops fitting at all, are in the queued full run.
- **Forward pass only.** No backward pass, so this is for inference, not training.
- **fp16 only**, head_dim 64 and 128, no dropout, no ALiBi or sliding window.
- **One GPU generation.** Tile sizes are tuned only for the T4 (sm_75, 64 KB shared memory per SM).
- **One run per configuration**, no error bars; Kaggle T4s vary by up to 30%
  between sessions.
- **The decode baselines are not paged.** `torch_sdpa_contiguous` reads an
  ordinary contiguous cache, which a paged serving system does not have; it is
  there as a bandwidth reference, not as a like-for-like competitor.

## What would change on Ampere and Hopper

- **Ampere (sm_80+).** `cp.async` lets the next K/V tile load into shared
  memory while the current one is multiplied (double buffering without
  register staging). `ldmatrix` and `mma.sync` (m16n8k16) replace WMMA and
  give direct control of fragment layout. bf16 becomes available, and up to
  164 KB of shared memory per SM allows bigger tiles. This is roughly the
  step from this repo to FlashAttention-2.
- **Hopper (sm_90).** TMA moves whole tiles asynchronously; `wgmma` runs
  warp-group matrix multiplies; warp specialisation splits producer warps
  (loads) from consumer warps (math); FP8 roughly doubles throughput again.
  That is the FlashAttention-3 design.
- **Decode** changes less: it is bandwidth bound on every GPU, so the same
  split-K and grouped-query layout carries over, with wider loads and
  more SMs to fill.

## Reproducing

The Mac this was written on has no CUDA GPU, so GPU runs went through a
private Kaggle script kernel on the T4 accelerator.
[kaggle/make_kernel.py](kaggle/make_kernel.py) packs `attnkernels/`, `tests/`
and `bench/` into one `run.py`.

```
pip install kaggle
python kaggle/make_kernel.py --user <your-kaggle-username> [--quick]
kaggle kernels push -p kaggle/build
kaggle kernels output <your-kaggle-username>/flash-attention-from-scratch-bench -p kaggle/out
```

On a machine with a CUDA GPU, nvcc and ninja:

```
pip install torch pytest numpy matplotlib ninja
pytest -q tests
python bench/bench.py --out results/t4.json          # add --quick for the short run
python bench/report.py results/t4.json > results/t4.md
python bench/plots.py results/t4.json results/figures
```

## License

MIT, see [LICENSE](LICENSE).
