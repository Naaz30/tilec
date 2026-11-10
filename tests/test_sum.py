"""tilec: the compiler (no GPU needed) and tilec.ext.sum on the Apple GPU."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tilec 
import tilec.language as tl  
from tilec import ext


def test_compiles_without_gpu():
    x = torch.zeros(2, 3)                     
    ck = ext._row_sum_kernel.inspect(x, torch.zeros(2), 3, 0, BLOCK=4, CHUNKS=1, ROWS=1)
    assert "reduce(%22) {kind='sum'} : f32" in ck.ir
    assert "kernel void _row_sum_kernel(" in ck.msl
    assert "simd_sum(" in ck.msl


def test_compile_error_names_the_line():
    @tilec.jit
    def bad(x_ptr):
        y = tl.arange(0, 10)  # noqa: F841

    with pytest.raises(tilec.CompileError, match=r"power of two \(got 10\)\n  at line \d+: y = tl.arange"):
        bad.inspect(torch.zeros(1))
