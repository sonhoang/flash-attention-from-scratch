"""Turn a bench JSON file into markdown tables.

    python bench/report.py results/t4.json > results/t4.md
"""
import json
import sys
from collections import OrderedDict

PREFILL_ORDER = ["sdpa", "sdpa_flash", "sdpa_efficient", "sdpa_cudnn", "sdpa_math", "naive",
                 "flash_simt", "flash_wmma_bn32", "flash_wmma_bn64"]
DECODE_ORDER = ["paged_split", "paged_nosplit", "torch_gather_sdpa", "torch_sdpa_contiguous"]


def pivot(rows, key_fn, col, val, order):
    table = OrderedDict()
    cols = [c for c in order if any(r[col] == c for r in rows)]
    for r in rows:
        table.setdefault(key_fn(r), {})[r[col]] = r[val]
    return table, cols


def fmt(x, nd):
    return "" if x is None else f"{x:.{nd}f}"


def print_pivot(title, keys_hdr, table, cols, nd):
    print(f"#### {title}\n")
    print("| " + " | ".join(keys_hdr + cols) + " |")
    print("|" + "---|" * len(keys_hdr) + "---:|" * len(cols))
    for k, vals in table.items():
        k = k if isinstance(k, tuple) else (k,)
        print("| " + " | ".join([str(x) for x in k] + [fmt(vals.get(c), nd) for c in cols]) + " |")
    print()


def main(path):
    data = json.load(open(path))
    m = data["meta"]
    smi = m.get("nvidia_smi", "")
    driver = smi.split(",")[1].strip() if smi.count(",") >= 1 else "?"
    print(f"GPU: {m['gpu']} (sm_{m['sm'].replace('.', '')}, {m['sm_count']} SMs, "
          f"{m['memory_gb']} GB), driver {driver}, PyTorch {m['torch']}, CUDA {m['cuda']}, "
          f"measured {m['date']}.\n")
    if "roofs" in m:
        print(f"Measured ceilings: device-to-device copy {m['roofs']['copy_gbps']} GB/s, "
              f"fp16 GEMM 8192^3 (cuBLAS) {m['roofs']['fp16_gemm_tflops']} TFLOP/s.\n")
    if "sdpa_backends" in m:
        print("SDPA backends on this GPU: " + ", ".join(
            f"{k}: {'ok' if v == 'ok' else 'unavailable'}" for k, v in m["sdpa_backends"].items()) + ".\n")
    res = data["results"]

    pre = [r for r in res if r["section"] == "prefill"]
    for d in sorted({r["head_dim"] for r in pre}):
        for causal in (False, True):
            rows = [r for r in pre if r["head_dim"] == d and r["causal"] == causal]
            if not rows:
                continue
            t, cols = pivot(rows, lambda r: (r["seq"], r["batch"]), "impl", "tflops", PREFILL_ORDER)
            print_pivot(f"Prefill TFLOP/s, head_dim {d}, {'causal' if causal else 'non-causal'}, "
                        f"16 heads", ["seq", "batch"], t, cols, 2)
    for d in sorted({r["head_dim"] for r in pre}):
        rows = [r for r in pre if r["head_dim"] == d and not r["causal"]]
        t, cols = pivot(rows, lambda r: (r["seq"], r["batch"]), "impl", "peak_mem_mb", PREFILL_ORDER)
        print_pivot(f"Prefill peak extra memory (MB), head_dim {d}, non-causal "
                    f"(includes the output)", ["seq", "batch"], t, cols, 0)
    if pre:
        print("#### Prefill max abs error vs fp32 (worst case over all shapes)\n")
        print("| impl | max abs error |\n|---|---:|")
        for c in PREFILL_ORDER:
            errs = [r["max_abs_err"] for r in pre if r["impl"] == c]
            if errs:
                print(f"| {c} | {max(errs):.2e} |")
        print()

    dec = [r for r in res if r["section"] == "decode"]
    if dec:
        t, cols = pivot(dec, lambda r: (r["batch"], r["ctx"]), "impl", "gbps", DECODE_ORDER)
        print_pivot("Decode GB/s (K+V bytes / time), 32 query heads, 8 kv heads, head_dim 128",
                    ["batch", "ctx"], t, cols, 1)
        t, cols = pivot(dec, lambda r: (r["batch"], r["ctx"]), "impl", "ms", DECODE_ORDER)
        print_pivot("Decode time (ms)", ["batch", "ctx"], t, cols, 4)
    abl = [r for r in res if r["section"] == "split_ablation"]
    if abl:
        print("#### Split-K partition size\n")
        print("| batch | ctx | partition | splits | blocks | ms | GB/s |\n|---|---|---:|---:|---:|---:|---:|")
        for r in abl:
            print(f"| {r['batch']} | {r['ctx']} | {r['partition']} | {r['splits']} | "
                  f"{r['splits'] * r['batch'] * r['kv_heads']} | {r['ms']:.4f} | {r['gbps']:.1f} |")
        print()
    rag = [r for r in res if r["section"] == "ragged"]
    if rag:
        lens = rag[0].get("ctx_lens", [])
        print(f"#### Ragged batch: {len(lens)} sequences, mean context "
              f"{rag[0]['ctx_mean']:.0f}, max {rag[0]['ctx']}\n")
        print("| impl | ms | GB/s (useful bytes) |\n|---|---:|---:|")
        for r in rag:
            print(f"| {r['impl']} | {r['ms']:.4f} | {r['gbps']:.1f} |")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
