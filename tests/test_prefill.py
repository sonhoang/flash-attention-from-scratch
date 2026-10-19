"""Prefill attention vs an fp32 PyTorch reference.

Inputs are fp16, the reference is computed in fp32 from the same fp16 values,
so the error is only the kernel's own rounding (fp16 output, and fp16 P in the
tensor-core kernel). The outputs are averages of N(0, 1) values, |o| < ~3.
"""
import pytest
import torch

import attnkernels as ak
from attnkernels.reference import attention_fp32
from conftest import max_err

ATOL = 5e-3

VARIANTS = {
    "naive": lambda q, k, v, c: ak.naive_attention(q, k, v, causal=c),
    "simt": lambda q, k, v, c: ak.flash_attention(q, k, v, causal=c, variant="simt"),
    "wmma32": lambda q, k, v, c: ak.flash_attention(q, k, v, causal=c, variant="wmma", block_n=32),
    "wmma64": lambda q, k, v, c: ak.flash_attention(q, k, v, causal=c, variant="wmma", block_n=64),
}


def rand_qkv(b, h, n, d, q_scale=1.0):
    q = (torch.randn(b, h, n, d, device="cuda") * q_scale).half()
    k = torch.randn(b, h, n, d, device="cuda").half()
    v = torch.randn(b, h, n, d, device="cuda").half()
    return q, k, v


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("n", [1, 17, 64, 100, 129, 256, 333])
def test_matches_fp32(variant, causal, d, n):
    q, k, v = rand_qkv(2, 3, n, d)
    out = VARIANTS[variant](q, k, v, causal)
    ref = attention_fp32(q, k, v, causal)
    assert out.shape == q.shape and out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert max_err(out, ref) < ATOL


@pytest.mark.parametrize("variant", ["simt", "wmma32", "wmma64"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
def test_long_sequence(variant, causal, d):
    q, k, v = rand_qkv(1, 2, 2048 + 7, d)
    out = VARIANTS[variant](q, k, v, causal)
    assert max_err(out, attention_fp32(q, k, v, causal)) < ATOL


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("d", [64, 128])
def test_large_logits(variant, d):
    """Scores with a std of about 6: exp would overflow without the running max."""
    q, k, v = rand_qkv(1, 2, 300, d, q_scale=6.0)
    out = VARIANTS[variant](q, k, v, True)
    assert torch.isfinite(out).all()
    assert max_err(out, attention_fp32(q, k, v, True)) < 2 * ATOL


def test_causal_first_row_is_v0():
    q, k, v = rand_qkv(1, 1, 128, 64)
    for name, fn in VARIANTS.items():
        out = fn(q, k, v, True)
        assert max_err(out[0, 0, 0], v[0, 0, 0]) < 1e-3, name


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_matches_sdpa(variant):
    q, k, v = rand_qkv(2, 4, 512, 64)
    out = VARIANTS[variant](q, k, v, True)
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert max_err(out, sdpa) < ATOL


def test_rejects_bad_input():
    q, k, v = rand_qkv(1, 1, 16, 32)
    with pytest.raises(ValueError):
        ak.flash_attention(q, k, v)
