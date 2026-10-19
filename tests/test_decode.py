"""Paged decode attention vs a per-sequence fp32 reference."""
import pytest
import torch

import attnkernels as ak
from attnkernels.reference import make_paged_cache, paged_decode_fp32, paged_decode_torch
from conftest import max_err

ATOL = 3e-3


def run(ctx, hq, hkv, d, bs=16, partition=None):
    k_cache, v_cache, tables, lens = make_paged_cache(ctx, hkv, d, block_size=bs, extra_blocks=5)
    q = torch.randn(len(ctx), hq, d, device="cuda").half()
    out = ak.paged_decode(q, k_cache, v_cache, tables, lens, max(ctx), partition_size=partition)
    ref = paged_decode_fp32(q, k_cache, v_cache, tables, lens)
    assert out.shape == q.shape and out.dtype == torch.float16
    assert torch.isfinite(out).all()
    return max_err(out, ref)


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("heads", [(8, 8), (8, 4), (32, 8), (16, 2)])
@pytest.mark.parametrize("ctx", [[1], [100], [256], [1000, 1000], [5000]])
def test_uniform(d, heads, ctx):
    assert run(ctx, heads[0], heads[1], d) < ATOL


RAGGED = [1, 17, 255, 256, 257, 999, 4097, 33]


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("partition", [None, 256, 1024, 1 << 20])
def test_ragged_batch(d, partition):
    """Different context lengths in one batch; partition 1<<20 means no split-K."""
    assert run(RAGGED, 32, 8, d, partition=partition) < ATOL


@pytest.mark.parametrize("bs", [8, 16, 32])
def test_block_sizes(bs):
    assert run([300, 2000, 17], 8, 2, 128, bs=bs) < ATOL


def test_long_context_split():
    assert run([32768], 32, 8, 128) < ATOL


def test_split_choice_does_not_change_result():
    k_cache, v_cache, tables, lens = make_paged_cache([6000, 3000], 4, 128)
    q = torch.randn(2, 16, 128, device="cuda").half()
    outs = [ak.paged_decode(q, k_cache, v_cache, tables, lens, 6000, partition_size=p)
            for p in (256, 768, 2048, 6144)]
    for o in outs[1:]:
        assert max_err(o, outs[0]) < 2e-3


def test_torch_baseline_matches():
    """The gather + SDPA baseline used in the benchmark is itself correct."""
    ctx = [512, 512, 300]
    k_cache, v_cache, tables, lens = make_paged_cache(ctx, 8, 128)
    q = torch.randn(3, 32, 128, device="cuda").half()
    out = paged_decode_torch(q, k_cache, v_cache, tables, 32, context_lens=lens)
    assert max_err(out, paged_decode_fp32(q, k_cache, v_cache, tables, lens)) < ATOL
