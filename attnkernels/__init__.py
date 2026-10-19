"""Attention kernels for LLM inference, written by hand in CUDA C++.

Prefill (many query tokens per sequence):
    naive_attention   three kernels, materialises the N x N scores
    flash_attention   tiled, online softmax; variant "simt" (CUDA cores)
                      or "wmma" (tensor cores)

Decode (one query token per sequence) over a paged KV cache:
    paged_decode      split-K / flash-decoding, plus a reduce kernel

The extension is compiled on first use with torch.utils.cpp_extension.load
(needs nvcc and ninja). Set TORCH_CUDA_ARCH_LIST, for example "7.5" on a T4,
to compile for one GPU only.
"""
from __future__ import annotations

import functools
import math
import os
from pathlib import Path

import torch

__all__ = ["naive_attention", "flash_attention", "paged_decode", "decode_partition",
           "load_extension"]

_CSRC = Path(__file__).resolve().parent / "csrc"
_SOURCES = ["ext.cpp", "naive_attn.cu", "flash_simt.cu", "flash_wmma.cu", "paged_decode.cu"]
CHUNK = 256  # tokens per chunk inside a decode block; partitions are multiples of it

# Key-tile width of the WMMA kernel per head dim (picked from the tile-size
# ablation in the README).
WMMA_BLOCK_N = {64: 64, 128: 64}


@functools.lru_cache(maxsize=None)
def load_extension():
    from torch.utils.cpp_extension import load

    cuda_flags = ["-O3", "-lineinfo", "--use_fast_math"]
    if os.environ.get("ATTN_PTXAS_VERBOSE") == "1":
        cuda_flags.append("-Xptxas=-v")
    return load(
        name="attnkernels_ext",
        sources=[str(_CSRC / s) for s in _SOURCES],
        extra_include_paths=[str(_CSRC)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_flags,
        verbose=os.environ.get("ATTN_VERBOSE") == "1",
    )


def _check_prefill(q, k, v):
    for t in (q, k, v):
        if not (t.is_cuda and t.dtype == torch.float16 and t.dim() == 4):
            raise ValueError("q, k, v must be fp16 CUDA tensors of shape [B, H, N, D]")
    if not (q.shape == k.shape == v.shape):
        raise ValueError("q, k, v must have the same shape (self-attention prefill)")
    if q.shape[-1] not in (64, 128):
        raise ValueError("head_dim must be 64 or 128")
    return q.contiguous(), k.contiguous(), v.contiguous()


def naive_attention(q, k, v, causal=False, scale=None):
    """softmax(scale * Q K^T [+ causal mask]) V with the scores in DRAM."""
    q, k, v = _check_prefill(q, k, v)
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    return load_extension().naive_attention(q, k, v, causal, scale)


def flash_attention(q, k, v, causal=False, scale=None, variant="wmma", block_n=None):
    """Tiled attention with online softmax. q, k, v: fp16 [B, H, N, D], D in {64, 128}.

    variant: "simt" (CUDA cores) or "wmma" (tensor cores).
    block_n: key-tile width for "wmma" (32 or 64).
    """
    q, k, v = _check_prefill(q, k, v)
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    ext = load_extension()
    if variant == "simt":
        return ext.flash_simt(q, k, v, causal, scale)
    if variant == "wmma":
        bn = WMMA_BLOCK_N[q.shape[-1]] if block_n is None else block_n
        return ext.flash_wmma(q, k, v, causal, scale, bn)
    raise ValueError(f"unknown variant {variant!r}")


@functools.lru_cache(maxsize=None)
def _sm_count(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def decode_partition(batch, num_kv_heads, max_context_len, device=None):
    """Tokens per split for paged_decode: enough splits for about 4 blocks per
    SM in total, at least one 256-token chunk per split."""
    dev = torch.cuda.current_device() if device is None else device
    blocks = batch * num_kv_heads
    want = -(-4 * _sm_count(dev) // blocks)
    max_chunks = -(-max_context_len // CHUNK)
    splits = max(1, min(want, max_chunks))
    return -(-max_chunks // splits) * CHUNK


def paged_decode(q, k_cache, v_cache, block_tables, context_lens, max_context_len,
                 scale=None, partition_size=None):
    """Decode attention over a paged KV cache.

    q:            [B, H, D] fp16, one query token per sequence
    k_cache:      [num_blocks, H_kv, block_size, D] fp16 (v_cache the same)
    block_tables: [B, max_blocks_per_seq] int32, physical block ids
    context_lens: [B] int32, tokens of context per sequence (>= 1)
    max_context_len: Python int, max(context_lens) (passed in to avoid a sync)
    partition_size: tokens per split (multiple of 256). None picks one from the
                  batch size; a value >= max_context_len disables split-K.
    """
    if not (q.dtype == k_cache.dtype == v_cache.dtype == torch.float16):
        raise ValueError("q and caches must be fp16")
    if q.shape[-1] not in (64, 128) or k_cache.shape[-1] != q.shape[-1]:
        raise ValueError("head_dim must be 64 or 128 and match the cache")
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    if partition_size is None:
        partition_size = decode_partition(q.shape[0], k_cache.shape[1], max_context_len,
                                          q.device.index)
    partition_size = -(-int(partition_size) // CHUNK) * CHUNK
    return load_extension().paged_decode(
        q.contiguous(), k_cache.contiguous(), v_cache.contiguous(),
        block_tables.to(torch.int32).contiguous(), context_lens.to(torch.int32).contiguous(),
        int(max_context_len), float(scale), partition_size)
