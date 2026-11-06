"""tilec: the compiler (no GPU needed) and tilec.ext.sum on the Apple GPU."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tilec  # noqa: E402
import tilec.language as tl  # noqa: E402
from tilec import ext  # noqa: E402


def test_compiles_without_gpu():
    x = torch.zeros(2, 3)                     # CPU tensor: only its type is used
    ck = ext._row_sum_kernel.inspect(x, torch.zeros(2), 3, BLOCK=4, CHUNKS=1)
    assert "reduce(%16) {kind='sum'} : f32" in ck.ir
    assert "kernel void _row_sum_kernel(" in ck.msl
    assert "simd_sum(acc0)" in ck.msl


def test_compile_error_names_the_line():
    @tilec.jit
    def bad(x_ptr):
        y = tl.arange(0, 10)  # noqa: F841

    with pytest.raises(tilec.CompileError, match=r"power of two \(got 10\)\n  at line \d+: y = tl.arange"):
        bad.inspect(torch.zeros(1))


needs_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")


@needs_gpu
@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (4, 8), (3, 1000), (37, 5000), (2, 100000)])
@pytest.mark.parametrize("axis", [0, 1])
def test_sum(shape, axis):
    x = torch.randn(*shape, device="mps")
    got = ext.sum(x, axis)
    torch.mps.synchronize()
    np.testing.assert_allclose(got.cpu().numpy(), x.sum(axis).cpu().numpy(), rtol=1e-4, atol=1e-2)
