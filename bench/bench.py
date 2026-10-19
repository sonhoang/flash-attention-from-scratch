"""Benchmark the attention kernels against PyTorch and write a JSON file.

Timing: CUDA events around each call, after warm-up calls, median over N
calls (N chosen so a case runs for about 0.3 s, between 3 and 50 calls). The
L2 cache (4 MB on a T4) is flushed before every timed call, outside the
timed region, so small decode cases are not served from cache.

    python bench/bench.py --out results/t4.json
    python bench/bench.py --quick          # small sizes, smoke test
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import attnkernels as ak  # noqa: E402
from attnkernels import reference as ref  # noqa: E402

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # older torch
    SDPBackend = sdpa_kernel = None

_FLUSH = None


def flush_l2():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(32 * 2**20, dtype=torch.uint8, device="cuda")
    _FLUSH.zero_()


def time_ms(fn, warmup=1, target_ms=300.0, min_reps=3, max_reps=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    reps = max_reps
    while len(times) < reps:
        flush_l2()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        if len(times) == 1:
            reps = int(min(max_reps, max(min_reps, target_ms / max(times[0], 1e-3))))
    return statistics.median(times)


def peak_extra_mb(fn):
    """Peak memory allocated during one call, above what was allocated before
    (includes the output tensor)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    return round(peak / 2**20, 1)


def kernel_names(fn):
    """Names of the CUDA kernels one call launches (to see which SDPA backend ran)."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for e in prof.events():
        dev = getattr(e, "device_type", None)
        if dev is not None and "CUDA" in str(dev) and e.name not in names:
            names.append(e.name)
    return names


def sdpa_with(backend):
    def run(q, k, v, causal):
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return run


def sdpa_backends():
    """Which SDPA backends run on this GPU for fp16 [B, H, N, 64/128]."""
    out = {}
    if sdpa_kernel is None:
        return out
    q = torch.randn(1, 2, 256, 128, device="cuda", dtype=torch.float16)
    for name in ["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"]:
        be = getattr(SDPBackend, name, None)
        if be is None:
            continue
        try:
            sdpa_with(be)(q, q, q, True)
            torch.cuda.synchronize()
            out[name] = "ok"
        except Exception as e:  # noqa: BLE001
            out[name] = "unavailable: " + str(e).splitlines()[0][:200]
    return out


def roofs():
    """Practical ceilings on this GPU: device copy bandwidth and fp16 GEMM."""
    x = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    ms = time_ms(lambda: y.copy_(x))
    copy_gbps = 2 * x.numel() / (ms * 1e-3) / 1e9
    del x, y
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    ms = time_ms(lambda: a @ a)
    mm_tflops = 2 * 8192**3 / (ms * 1e-3) / 1e12
    del a
    torch.cuda.empty_cache()
    return dict(copy_gbps=round(copy_gbps, 1), fp16_gemm_tflops=round(mm_tflops, 2))


def max_err_slice(fn_out, q, k, v, causal):
    """Max abs error against fp32 on batch 0, heads 0-1 (keeps the reference small)."""
    r = ref.attention_fp32(q[:1, :2], k[:1, :2], v[:1, :2], causal)
    return (fn_out[:1, :2].float() - r).abs().max().item()


# ---------------------------------------------------------------- prefill --

def bench_prefill(results, quick, backends):
    heads = 16
    tokens = 4096 if quick else 16384  # batch * seq kept constant
    seqs = [512, 1024] if quick else [512, 1024, 2048, 4096, 8192]
    for d in [64, 128]:
        for causal in [False, True]:
            for n in seqs:
                b = max(1, tokens // n)
                q = torch.randn(b, heads, n, d, device="cuda", dtype=torch.float16)
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                flops = 4.0 * b * heads * n * n * d * (0.5 if causal else 1.0)
                impls = {"sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)}
                for name in ["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION"]:
                    if backends.get(name) == "ok":
                        impls["sdpa_" + name.split("_")[0].lower()] = (
                            lambda be=getattr(SDPBackend, name): sdpa_with(be)(q, k, v, causal))
                if n <= 4096 and backends.get("MATH") == "ok":
                    impls["sdpa_math"] = lambda: sdpa_with(SDPBackend.MATH)(q, k, v, causal)
                if n <= 4096:
                    impls["naive"] = lambda: ak.naive_attention(q, k, v, causal=causal)
                impls["flash_simt"] = lambda: ak.flash_attention(q, k, v, causal, variant="simt")
                for bn in (32, 64):
                    impls[f"flash_wmma_bn{bn}"] = (
                        lambda bn=bn: ak.flash_attention(q, k, v, causal, variant="wmma", block_n=bn))
                rows = []
                for name, fn in impls.items():
                    try:
                        out = fn()
                        err = max_err_slice(out, q, k, v, causal)
                        del out
                        ms = time_ms(fn)
                        mem = peak_extra_mb(fn)
                    except Exception as e:  # noqa: BLE001  e.g. out of memory
                        print(f"prefill {name} d={d} N={n} failed: {e!r}"[:300], flush=True)
                        torch.cuda.empty_cache()
                        continue
                    rows.append(dict(section="prefill", impl=name, batch=b, heads=heads, seq=n,
                                     head_dim=d, causal=causal, ms=round(ms, 4),
                                     tflops=round(flops / (ms * 1e-3) / 1e12, 3),
                                     peak_mem_mb=mem, max_abs_err=float(f"{err:.3g}")))
                base = next((r["ms"] for r in rows if r["impl"] == "sdpa"), None)
                for r in rows:
                    r["speedup_vs_sdpa"] = round(base / r["ms"], 3) if base else None
                    results.append(r)
                    print(f"prefill d={d} causal={int(causal)} B={b} N={n:5d} {r['impl']:18s} "
                          f"{r['ms']:9.3f} ms {r['tflops']:7.3f} TFLOP/s {r['peak_mem_mb']:8.1f} MB "
                          f"err {r['max_abs_err']:.2e} x{r['speedup_vs_sdpa']}", flush=True)
                del q, k, v
                torch.cuda.empty_cache()


# ----------------------------------------------------------------- decode --

HQ, HKV, D, BS = 32, 8, 128, 16  # Llama-3-8B-like attention layer


def kv_bytes(ctx_lens):
    return sum(ctx_lens) * HKV * D * 2 * 2  # K and V, fp16


def decode_case(results, section, ctx_lens, impls_wanted, partition=None, extra=None):
    B = len(ctx_lens)
    kc, vc, tables, lens = ref.make_paged_cache(ctx_lens, HKV, D, block_size=BS)
    q = torch.randn(B, HQ, D, device="cuda", dtype=torch.float16)
    max_ctx = max(ctx_lens)
    nblk = -(-max_ctx // BS)
    uniform = len(set(ctx_lens)) == 1
    impls = {}
    if "paged_split" in impls_wanted:
        impls["paged_split"] = lambda: ak.paged_decode(q, kc, vc, tables, lens, max_ctx,
                                                        partition_size=partition)
    if "paged_nosplit" in impls_wanted:
        impls["paged_nosplit"] = lambda: ak.paged_decode(q, kc, vc, tables, lens, max_ctx,
                                                          partition_size=1 << 30)
    if "torch_gather_sdpa" in impls_wanted:
        mask_lens = None if uniform else lens
        impls["torch_gather_sdpa"] = lambda: ref.paged_decode_torch(q, kc, vc, tables, nblk,
                                                                     context_lens=mask_lens)
    kg = vg = None
    if "torch_sdpa_contiguous" in impls_wanted and uniform:
        kg = ref.gather_kv(kc, tables, nblk)
        vg = ref.gather_kv(vc, tables, nblk)
        qg = q.view(B, HKV, HQ // HKV, D)
        impls["torch_sdpa_contiguous"] = lambda: F.scaled_dot_product_attention(qg, kg, vg)
    expect = ref.paged_decode_fp32(q[:1], kc, vc, tables[:1], lens[:1])
    nbytes = kv_bytes(ctx_lens)
    part = partition if partition is not None else ak.decode_partition(B, HKV, max_ctx)
    rows = []
    for name, fn in impls.items():
        try:
            out = fn()
            err = (out[:1].float().reshape(expect.shape) - expect).abs().max().item()
            del out
            ms = time_ms(fn)
        except Exception as e:  # noqa: BLE001  e.g. out of memory
            print(f"{section} {name} B={B} ctx={max_ctx} failed: {e!r}"[:300], flush=True)
            torch.cuda.empty_cache()
            continue
        row = dict(section=section, impl=name, batch=B, ctx=max_ctx, ctx_mean=sum(ctx_lens) / B,
                   heads=HQ, kv_heads=HKV, head_dim=D, block_size=BS, ms=round(ms, 4),
                   gbps=round(nbytes / (ms * 1e-3) / 1e9, 1), max_abs_err=float(f"{err:.3g}"))
        if name == "paged_split":
            row["partition"] = part
            row["splits"] = -(-max_ctx // part)
        if extra:
            row.update(extra)
        rows.append(row)
    base_name = "torch_gather_sdpa" if any(r["impl"] == "torch_gather_sdpa" for r in rows) else None
    for r in rows:
        if base_name:
            r["speedup_vs_torch_gather"] = round(
                next(x["ms"] for x in rows if x["impl"] == base_name) / r["ms"], 3)
        results.append(r)
        print(f"{section} B={B:3d} ctx={max_ctx:6d} {r['impl']:22s} {r['ms']:8.4f} ms "
              f"{r['gbps']:7.1f} GB/s err {r['max_abs_err']:.1e}"
              f"{'  part=' + str(r.get('partition')) if 'partition' in r else ''}", flush=True)
    del kc, vc, kg, vg, q
    torch.cuda.empty_cache()


def bench_decode(results, quick):
    batches = [1, 16] if quick else [1, 4, 16, 64]
    ctxs = [1024, 4096] if quick else [1024, 2048, 4096, 8192, 16384, 32768]
    all_impls = ["paged_split", "paged_nosplit", "torch_gather_sdpa", "torch_sdpa_contiguous"]
    for B in batches:
        for c in ctxs:
            if kv_bytes([c] * B) > 2.2e9:
                continue
            decode_case(results, "decode", [c] * B, all_impls)


def bench_split_ablation(results, quick):
    cases = [(1, 8192)] if quick else [(1, 32768), (4, 8192), (16, 4096)]
    for B, c in cases:
        p = 256
        while p <= c:
            decode_case(results, "split_ablation", [c] * B, ["paged_split"], partition=p)
            p *= 2


def bench_ragged(results, quick):
    g = torch.Generator().manual_seed(0)
    B = 8 if quick else 16
    hi = 8192 if quick else 32768
    lens = torch.randint(512, hi + 1, (B,), generator=g).tolist()
    decode_case(results, "ragged", lens, ["paged_split", "paged_nosplit", "torch_gather_sdpa"],
                extra=dict(ctx_lens=lens))


# ------------------------------------------------------------------- main --

def meta():
    p = torch.cuda.get_device_properties(0)
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,"
                              "clocks.max.mem", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        smi = ""
    return dict(gpu=p.name, sm=f"{p.major}.{p.minor}", sm_count=p.multi_processor_count,
                memory_gb=round(p.total_memory / 2**30, 1), torch=torch.__version__,
                cuda=torch.version.cuda, nvidia_smi=smi, python=platform.python_version(),
                date=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bench.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=["prefill", "decode", "split_ablation", "ragged"])
    args = ap.parse_args()
    torch.manual_seed(0)
    ak.load_extension()
    info = meta()
    info["sdpa_backends"] = sdpa_backends()
    try:
        for d in (64, 128):
            q = torch.randn(4, 16, 2048, d, device="cuda", dtype=torch.float16)
            info[f"sdpa_default_kernels_prefill_d{d}"] = kernel_names(
                lambda: F.scaled_dot_product_attention(q, q, q, is_causal=True))
        qd = torch.randn(4, 8, 4, 128, device="cuda", dtype=torch.float16)
        kd = torch.randn(4, 8, 4096, 128, device="cuda", dtype=torch.float16)
        info["sdpa_default_kernels_decode"] = kernel_names(
            lambda: F.scaled_dot_product_attention(qd, kd, kd))
        del q, qd, kd
    except Exception as e:  # noqa: BLE001
        info["sdpa_default_kernels_error"] = repr(e)[:300]
    info["roofs"] = roofs()
    print(json.dumps(info, indent=1), flush=True)
    results = []
    for name in args.only:
        try:
            if name == "prefill":
                bench_prefill(results, args.quick, info["sdpa_backends"])
            else:
                globals()["bench_" + name](results, args.quick)
        except Exception as e:  # noqa: BLE001  keep what we have
            print(f"section {name} failed: {e!r}", flush=True)
            info.setdefault("failed_sections", {})[name] = repr(e)[:500]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(meta=info, results=results), indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
