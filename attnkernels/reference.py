"""Plain PyTorch references used by the tests and the benchmark."""
from __future__ import annotations

import math

import torch


def attention_fp32(q, k, v, causal=False, scale=None):
    """Exact attention in fp32. q, k, v: [B, H, N, D] (any float dtype)."""
    q, k, v = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    s = (q @ k.transpose(-1, -2)) * scale
    if causal:
        n = q.shape[-2]
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def make_paged_cache(context_lens, num_kv_heads, head_dim, block_size=16, dtype=torch.float16,
                     device="cuda", extra_blocks=0, generator=None):
    """Random K/V caches and block tables for the given context lengths.

    Physical blocks are handed out in a random order, so a sequence's blocks are
    scattered through the pool like they are in a long-running server.
    Returns k_cache, v_cache [num_blocks, H_kv, block_size, D], block_tables
    [B, max_blocks] int32 (unused slots point at block 0, never read), and
    context_lens as an int32 tensor.
    """
    nblk = [-(-c // block_size) for c in context_lens]
    total = sum(nblk) + extra_blocks
    perm = torch.randperm(total, generator=generator).to(torch.int32)
    max_blocks = max(nblk)
    tables = torch.zeros(len(context_lens), max_blocks, dtype=torch.int32)
    pos = 0
    for b, nb in enumerate(nblk):
        tables[b, :nb] = perm[pos:pos + nb]
        pos += nb
    shape = (total, num_kv_heads, block_size, head_dim)
    k_cache = torch.randn(shape, device=device, dtype=dtype)
    v_cache = torch.randn(shape, device=device, dtype=dtype)
    ctx = torch.tensor(context_lens, dtype=torch.int32, device=device)
    return k_cache, v_cache, tables.to(device), ctx


def gather_kv(cache, block_tables, n_blocks):
    """[num_blocks, H_kv, bs, D] -> contiguous [B, H_kv, n_blocks * bs, D]."""
    g = cache[block_tables[:, :n_blocks].long()]  # [B, n_blocks, H_kv, bs, D]
    b, nb, h, bs, d = g.shape
    return g.permute(0, 2, 1, 3, 4).reshape(b, h, nb * bs, d)


def paged_decode_fp32(q, k_cache, v_cache, block_tables, context_lens, scale=None):
    """Per-sequence fp32 reference for paged decode. q: [B, H, D]."""
    B, H, D = q.shape
    hkv, bs = k_cache.shape[1], k_cache.shape[2]
    g = H // hkv
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    out = torch.empty(B, H, D, dtype=torch.float32, device=q.device)
    for b in range(B):
        c = int(context_lens[b])
        nb = -(-c // bs)
        k = gather_kv(k_cache, block_tables[b:b + 1], nb)[0, :, :c].float()  # [H_kv, c, D]
        v = gather_kv(v_cache, block_tables[b:b + 1], nb)[0, :, :c].float()
        k = k.repeat_interleave(g, dim=0)
        v = v.repeat_interleave(g, dim=0)
        s = torch.einsum("hd,hcd->hc", q[b].float(), k) * scale
        out[b] = torch.einsum("hc,hcd->hd", torch.softmax(s, -1), v)
    return out


def paged_decode_torch(q, k_cache, v_cache, block_tables, n_blocks, context_lens=None):
    """What you would do without a paged kernel: gather the blocks into a
    contiguous (padded) cache, then call SDPA. Grouped-query attention is
    expressed by folding the G query heads of one kv head into G query rows, so
    no K/V copy per query head is needed. With context_lens, tokens past each
    sequence's length are masked out."""
    B, H, D = q.shape
    hkv = k_cache.shape[1]
    k = gather_kv(k_cache, block_tables, n_blocks)
    v = gather_kv(v_cache, block_tables, n_blocks)
    qg = q.view(B, hkv, H // hkv, D)
    mask = None
    if context_lens is not None:
        idx = torch.arange(k.shape[2], device=q.device)
        mask = (idx[None, :] < context_lens[:, None].long())[:, None, None, :]
    o = torch.nn.functional.scaled_dot_product_attention(qg, k, v, attn_mask=mask)
    return o.reshape(B, H, D)
