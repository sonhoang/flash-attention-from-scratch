"""Charts for the README, from a bench JSON file.

    python bench/plots.py results/t4.json results/figures
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "sdpa": ("#444444", "o", "-"),
    "sdpa_efficient": ("#888888", "s", "--"),
    "sdpa_flash": ("#888888", "^", "--"),
    "sdpa_math": ("#bbbbbb", "v", ":"),
    "naive": ("#d62728", "x", ":"),
    "flash_simt": ("#ff7f0e", "D", "-"),
    "flash_wmma_bn32": ("#1f77b4", "o", "--"),
    "flash_wmma_bn64": ("#1f77b4", "o", "-"),
    "paged_split": ("#1f77b4", "o", "-"),
    "paged_nosplit": ("#ff7f0e", "D", "-"),
    "torch_gather_sdpa": ("#444444", "s", "-"),
    "torch_sdpa_contiguous": ("#888888", "^", "--"),
}


def line(ax, rows, x, y, impl, label=None):
    rows = sorted((r for r in rows if r["impl"] == impl), key=lambda r: r[x])
    if not rows:
        return
    c, mk, ls = STYLE.get(impl, ("k", ".", "-"))
    ax.plot([r[x] for r in rows], [r[y] for r in rows], color=c, marker=mk, ls=ls,
            label=label or impl, lw=1.6, ms=4)


def prefill(res, out):
    pre = [r for r in res if r["section"] == "prefill"]
    if not pre:
        return
    impls = [i for i in STYLE if any(r["impl"] == i for r in pre)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for i, d in enumerate([64, 128]):
        for j, causal in enumerate([False, True]):
            ax = axes[i][j]
            rows = [r for r in pre if r["head_dim"] == d and r["causal"] == causal]
            for impl in impls:
                line(ax, rows, "seq", "tflops", impl)
            ax.set_xscale("log", base=2)
            ax.set_title(f"head_dim {d}, {'causal' if causal else 'non-causal'}", fontsize=10)
            ax.set_ylabel("TFLOP/s")
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("sequence length (batch x seq = 16k tokens, 16 heads)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Prefill attention forward on a T4")
    fig.tight_layout()
    fig.savefig(out / "prefill_tflops.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    rows = [r for r in pre if r["head_dim"] == 64 and not r["causal"]]
    for impl in impls:
        line(ax, rows, "seq", "peak_mem_mb", impl)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length (batch x seq = 16k tokens)")
    ax.set_ylabel("peak extra memory per call (MB)")
    ax.set_title("Memory: materialised scores vs tiling (head_dim 64)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "prefill_memory.png", dpi=130)
    plt.close(fig)


def decode(res, meta, out):
    dec = [r for r in res if r["section"] == "decode"]
    if not dec:
        return
    batches = sorted({r["batch"] for r in dec})
    fig, axes = plt.subplots(1, len(batches), figsize=(3.2 * len(batches), 3.6), sharey=True)
    axes = axes if len(batches) > 1 else [axes]
    copy = meta.get("roofs", {}).get("copy_gbps")
    for ax, B in zip(axes, batches):
        rows = [r for r in dec if r["batch"] == B]
        for impl in ["paged_split", "paged_nosplit", "torch_gather_sdpa", "torch_sdpa_contiguous"]:
            line(ax, rows, "ctx", "gbps", impl)
        ax.axhline(320, color="k", lw=0.8, ls=":")
        if copy:
            ax.axhline(copy, color="g", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_title(f"batch {B}", fontsize=10)
        ax.set_xlabel("context length")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("GB/s of K+V read")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Paged decode attention on a T4 (dotted: 320 GB/s spec; green: measured copy)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "decode_gbps.png", dpi=130)
    plt.close(fig)

    abl = [r for r in res if r["section"] == "split_ablation"]
    if abl:
        fig, ax = plt.subplots(figsize=(6, 4))
        for (B, c) in sorted({(r["batch"], r["ctx"]) for r in abl}):
            rows = sorted((r for r in abl if r["batch"] == B and r["ctx"] == c),
                          key=lambda r: r["partition"])
            ax.plot([r["splits"] for r in rows], [r["gbps"] for r in rows], marker="o",
                    label=f"batch {B}, ctx {c}")
        ax.set_xscale("log", base=2)
        ax.axhline(320, color="k", lw=0.8, ls=":")
        ax.set_xlabel("splits per sequence (context / partition size)")
        ax.set_ylabel("GB/s")
        ax.set_title("Split-K: how many pieces to cut the context into", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "decode_split_ablation.png", dpi=130)
        plt.close(fig)


def main(path, out):
    data = json.load(open(path))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prefill(data["results"], out)
    decode(data["results"], data["meta"], out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
