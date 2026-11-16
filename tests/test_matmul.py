"""tilec: the compiler (no GPU needed) and tilec.ext.matrix_mul on the Apple GPU."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tilec 
import tilec.language as tl  
from tilec import ext


def test_compiles_without_gpu():
    a, b, c = torch.zeros(64, 16), torch.zeros(16, 64), torch.zeros(64, 64)
    ck = ext._matmul_tiled.inspect(a, b, c, 64, 64, 16, BM=64, BN=64, BK=16)
    assert "dot(" in ck.ir
    assert "kernel void _matmul_tiled(" in ck.msl


def test_compile_error_names_the_line():
    @tilec.jit
    def bad(x_ptr):
        y = tl.arange(0, 10)  # noqa: F841

    with pytest.raises(tilec.CompileError, match=r"power of two \(got 10\)\n  at line \d+: y = tl.arange"):
        bad.inspect(torch.zeros(1))


needs_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")


@needs_gpu
@pytest.mark.parametrize("shape", [(1, 1, 1), (5, 7, 3), (65, 17, 63), (128, 300, 130), (512, 1024, 256)])
def test_matrix_mul(shape):
    m, k, n = shape
    x1 = torch.randn(m, k, device="mps")
    x2 = torch.randn(k, n, device="mps")
    got = ext.matrix_mul(x1, x2)
    torch.mps.synchronize()
    np.testing.assert_allclose(got.cpu().numpy(), (x1 @ x2).cpu().numpy(), rtol=1e-3, atol=1e-2)
