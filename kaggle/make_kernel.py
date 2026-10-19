"""Bundle this repo into a single Kaggle script kernel.

Kaggle script kernels are one file, so run.py carries the package, tests and
benchmark as an embedded tar.gz. On Kaggle it unpacks them, builds the
extension, runs pytest and the benchmark, and leaves build.txt, pytest.txt,
t4.json, t4.md and figures/ in /kaggle/working (the kernel output).

    python kaggle/make_kernel.py --user "$KAGGLE_USERNAME" [--quick]
    kaggle kernels push -p kaggle/build
"""
import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["attnkernels", "tests", "bench"]

RUNNER = r'''
import base64, io, os, subprocess, sys, tarfile, time
BUNDLE = "{bundle}"
QUICK = {quick}
SKIP_TESTS = {skip_tests}
SRC = "/kaggle/temp/fa-scratch" if os.path.isdir("/kaggle") else "/tmp/fa-scratch"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
os.makedirs(SRC, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall(SRC)

def sh(cmd, log=None, **kw):
    print("$", " ".join(cmd), flush=True)
    t = time.time()
    p = subprocess.run(cmd, cwd=SRC, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
    print(p.stdout, flush=True)
    if log:
        open(os.path.join(OUT, log), "w").write(p.stdout)
    print(f"[exit {{p.returncode}} after {{time.time() - t:.0f}} s]", flush=True)
    return p.returncode

sh(["nvidia-smi"], log="nvidia-smi.txt")
sh([sys.executable, "-m", "pip", "install", "-q", "ninja", "pytest", "matplotlib"])
import torch
cap = torch.cuda.get_device_capability(0)
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{{cap[0]}}.{{cap[1]}}"
os.environ["MAX_JOBS"] = "4"
env = dict(os.environ, ATTN_VERBOSE="1", ATTN_PTXAS_VERBOSE="1")
rc = sh([sys.executable, "-c", "import attnkernels; attnkernels.load_extension(); print('build ok')"],
        log="build.txt", env=env)
if rc == 0:
    if not SKIP_TESTS:
        sh([sys.executable, "-m", "pytest", "-q", "-rfE", "tests"], log="pytest.txt")
    args = [sys.executable, "bench/bench.py", "--out", os.path.join(OUT, "t4.json")] + (["--quick"] if QUICK else [])
    sh(args, log="bench.txt")
    if os.path.exists(os.path.join(OUT, "t4.json")):
        sh([sys.executable, "bench/report.py", os.path.join(OUT, "t4.json")], log="t4.md")
        sh([sys.executable, "bench/plots.py", os.path.join(OUT, "t4.json"), os.path.join(OUT, "figures")])
'''


def bundle():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in INCLUDE:
            tar.add(ROOT / name, arcname=name,
                    filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--slug", default="flash-attention-from-scratch-bench")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    out = ROOT / "kaggle" / "build"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.py").write_text(RUNNER.format(bundle=bundle(), quick=args.quick,
                                              skip_tests=args.skip_tests))
    meta = {
        "id": f"{args.user}/{args.slug}", "title": args.slug, "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
