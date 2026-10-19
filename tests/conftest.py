import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Without a CUDA GPU there is nothing to test; collect no test files.
if not torch.cuda.is_available():
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(scope="session", autouse=True)
def _build_once():
    import attnkernels

    attnkernels.load_extension()  # compile before the first test, not inside it


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def max_err(out, ref):
    return (out.float() - ref.float()).abs().max().item()
