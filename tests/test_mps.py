"""End-to-end tests on an Apple GPU: generated MSL vs the interpreter.

Skipped unless PyTorch with MPS is available (i.e. run these on a Mac).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

torch = pytest.importorskip("torch")
if not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")):
    pytest.skip("needs PyTorch >= 2.5 with MPS", allow_module_level=True)

from softmax import reference as softmax_ref  # noqa: E402
from softmax import softmax  # noqa: E402
from vector_add import add  # noqa: E402

from test_tier1 import fused_kernel, int_ops_kernel  # noqa: E402


@pytest.mark.parametrize("n", [1, 100, 1024, 98432])
def test_vector_add_mps(n):
    x = torch.rand(n, device="mps")
    y = torch.rand(n, device="mps")
    out = add(x, y)
    torch.mps.synchronize()
    assert torch.equal(out.cpu(), (x + y).cpu())


@pytest.mark.parametrize("shape", [(1, 1), (7, 33), (64, 781), (3, 1024)])
def test_softmax_mps(shape):
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    xt = torch.from_numpy(x).to("mps")
    out = softmax(xt, torch.empty_like(xt))
    torch.mps.synchronize()
    np.testing.assert_allclose(out.cpu().numpy(), softmax_ref(x), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("act", ["relu", "sigmoid", "none"])
def test_fused_matches_interpreter(act):
    n = 1000
    x = np.random.default_rng(1).standard_normal(n).astype(np.float32)
    ref = np.empty(n, np.float16)
    fused_kernel[(4,)](x, ref, n, 3.0, BLOCK=256, ACT=act)            # interpreter
    xt = torch.from_numpy(x).to("mps")
    out = torch.empty(n, dtype=torch.float16, device="mps")
    fused_kernel[(4,)](xt, out, n, 3.0, BLOCK=256, ACT=act)           # Metal
    torch.mps.synchronize()
    np.testing.assert_allclose(out.cpu().numpy().astype(np.float32),
                               ref.astype(np.float32), rtol=2e-3, atol=1e-3)


def test_int_ops_mps():
    x = np.arange(-8, 8, dtype=np.int32)
    ref = np.empty_like(x)
    int_ops_kernel[(1,)](x, ref, BLOCK=16)
    xt = torch.from_numpy(x).to("mps")
    out = torch.empty_like(xt)
    int_ops_kernel[(1,)](xt, out, BLOCK=16)
    torch.mps.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy(), ref)
