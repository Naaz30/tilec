"""Tier 1 tests: front end, interpreter semantics, and MSL generation.

The interpreter tests check numerics against NumPy.  The MSL tests check the
generated source for structure; actually running it needs a Mac (see
tests/test_mps.py, which is skipped elsewhere).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import tilec as triton 
import tilec.language as tl 
from tilec.codegen_msl import UnsupportedError 

from softmax import reference as softmax_ref 
from softmax import softmax, softmax_kernel
from vector_add import add, add_kernel 

rng = np.random.default_rng(1234)


@pytest.mark.parametrize("n", [1, 100, 1024, 98432])
def test_vector_add(n):
    x = rng.random(n, dtype=np.float32)
    y = rng.random(n, dtype=np.float32)
    np.testing.assert_array_equal(add(x, y), x + y)


@pytest.mark.parametrize("shape", [(1, 1), (7, 33), (64, 781), (3, 1024)])
def test_softmax(shape):
    x = rng.standard_normal(shape).astype(np.float32)
    got = softmax(x, np.empty_like(x))
    np.testing.assert_allclose(got, softmax_ref(x), rtol=1e-5, atol=1e-6)


@triton.jit
def fused_kernel(x_ptr, out_ptr, n, scale, BLOCK: tl.constexpr, ACT: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    if ACT == "relu":
        y = tl.maximum(x, 0.0)
    elif ACT == "sigmoid":
        y = tl.sigmoid(x)
    else:
        y = x
    y = tl.where(y > 0.5, y * scale, -y)
    tl.store(out_ptr + offs, y.to(tl.float16), mask=m)


@pytest.mark.parametrize("act", ["relu", "sigmoid", "none"])
def test_constexpr_branch_where_cast(act):
    n = 1000
    x = rng.standard_normal(n).astype(np.float32)
    out = np.empty(n, np.float16)
    fused_kernel[(triton.cdiv(n, 256),)](x, out, n, 3.0, BLOCK=256, ACT=act)
    y = {"relu": np.maximum(x, 0), "sigmoid": 1 / (1 + np.exp(-x)), "none": x}[act]
    ref = np.where(y > 0.5, y * 3.0, -y).astype(np.float16)
    np.testing.assert_allclose(out.astype(np.float32), ref.astype(np.float32), rtol=1e-3)


@triton.jit
def int_ops_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = (x // 3) + (x % 3) * 100 + (x & 1) * 1000
    tl.store(out_ptr + offs, y)


def test_integer_ops_use_c_semantics():
    x = np.arange(-8, 8, dtype=np.int32)
    out = np.empty_like(x)
    int_ops_kernel[(1,)](x, out, BLOCK=16)
    trunc_div = np.trunc(x / 3).astype(np.int32)
    c_mod = np.fmod(x, 3).astype(np.int32)
    np.testing.assert_array_equal(out, trunc_div + c_mod * 100 + (x & 1) * 1000)


@triton.jit
def row_sum_2d_kernel(x_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr):
    rows = tl.arange(0, R)
    cols = tl.arange(0, C)
    x = tl.load(x_ptr + rows[:, None] * C + cols[None, :])
    tl.store(out_ptr + rows, tl.sum(x, axis=1))


def test_2d_tiles_in_interpreter():
    x = rng.standard_normal((8, 16)).astype(np.float32)
    out = np.empty(8, np.float32)
    row_sum_2d_kernel[(1,)](x, out, R=8, C=16)
    np.testing.assert_allclose(out, x.sum(axis=1), rtol=1e-5)


@triton.jit
def unrolled_kernel(out_ptr, N: tl.constexpr):
    acc = tl.zeros((N,), tl.float32)
    for i in tl.static_range(4):
        acc += i
    tl.store(out_ptr + tl.arange(0, N), acc)


def test_static_range_unrolls():
    out = np.empty(32, np.float32)
    unrolled_kernel[(1,)](out, N=32)
    np.testing.assert_array_equal(out, np.full(32, 6.0, np.float32))


def test_unmasked_out_of_bounds_is_reported():
    x = np.zeros(10, np.float32)

    @triton.jit
    def oob(x_ptr, BLOCK: tl.constexpr):
        tl.store(x_ptr + tl.arange(0, BLOCK), 1.0)

    with pytest.raises(IndexError, match="out-of-bounds store"):
        oob[(1,)](x, BLOCK=16)


def test_non_power_of_two_arange_is_rejected():
    @triton.jit
    def bad(x_ptr):
        tl.store(x_ptr + tl.arange(0, 10), 1.0)

    with pytest.raises(triton.CompileError, match="power of two"):
        bad[(1,)](np.zeros(10, np.float32))


def test_runtime_if_is_rejected_with_hint():
    @triton.jit
    def bad(x_ptr, n):
        if n > 0:
            tl.store(x_ptr, 1.0)

    with pytest.raises(triton.CompileError, match="tl.where"):
        bad[(1,)](np.zeros(1, np.float32), 3)


def test_ir_text():
    x = np.zeros(4096, np.float32)
    ck = add_kernel.inspect(x, x, x, 4096, BLOCK_SIZE=1024)
    text = str(ck.ir)
    assert "func @add_kernel" in text
    assert "arange()" in text and "load(" in text and "store(" in text


def test_specialization_cache():
    x = np.zeros(64, np.float32)
    a = add_kernel.inspect(x, x, x, 64, BLOCK_SIZE=64)
    b = add_kernel.inspect(x, x, x, 64, BLOCK_SIZE=64)
    c = add_kernel.inspect(x, x, x, 64, BLOCK_SIZE=128)
    assert a is b and a is not c


def test_msl_vector_add_structure():
    x = np.zeros(4096, np.float32)
    msl = add_kernel.inspect(x, x, x, 4096, BLOCK_SIZE=1024, num_warps=4).msl
    assert "kernel void add_kernel(" in msl
    assert "device float* x_ptr [[buffer(0)]]" in msl
    assert "device const int* n_elements_arg [[buffer(3)]]" in msl
    # 1024 elements over 128 threads -> 8 registers per thread, coalesced indexing
    assert "for (uint r = 0; r < 8u; ++r)" in msl
    assert "(r * 128u + tid)" in msl
    assert msl.count("{") == msl.count("}")


def test_msl_softmax_has_simd_reductions():
    x = np.zeros((4, 781), np.float32)
    ck = softmax_kernel.inspect(x, x, 781, 781, 781, BLOCK_SIZE=1024)
    msl = ck.msl
    assert "simd_max(" in msl and "simd_sum(" in msl
    assert "threadgroup float red0[4];" in msl
    assert "threadgroup_barrier(mem_flags::mem_threadgroup);" in msl
    assert "precise::exp(" in msl


def test_msl_small_tile_is_replicated_and_store_guarded():
    x = np.zeros(32, np.float32)
    msl = add_kernel.inspect(x, x, x, 32, BLOCK_SIZE=32, num_warps=4).msl
    assert "(tid % 32u)" in msl
    assert "if ((tid < 32u) &&" in msl


def test_msl_rejects_2d_with_clear_message():
    x = np.zeros((8, 16), np.float32)
    ck = row_sum_2d_kernel.inspect(x, np.zeros(8, np.float32), R=8, C=16)
    with pytest.raises(NotImplementedError, match="Tier 2"):
        ck.msl


def test_msl_rejects_float64():
    x = np.zeros(16, np.float64)
    ck = add_kernel.inspect(x, x, x, 16, BLOCK_SIZE=16)
    with pytest.raises(UnsupportedError, match="not supported on Apple GPUs"):
        ck.msl


def test_trace_prints_every_stage_and_runs():
    import io
    from tilec.trace import trace_launch
    x = np.arange(1, 11, dtype=np.float32)
    out = np.zeros(10, np.float32)
    buf = io.StringIO()
    trace_launch(add_kernel, (3,), (x, x, out, 10), {"BLOCK_SIZE": 4}, program=2, out=buf)
    text = buf.getvalue()
    for stage in ["STAGE 1 · jit.py", "STAGE 2 · frontend.py", "STAGE 3 · ir.py",
                  "STAGE 4 · layout.py + codegen_msl.py", "STAGE 5 · interpreter.py"]:
        assert stage in text
    assert "AST    │ Assign offsets = BinOp(+, block_start, Call(tl.arange, 0, BLOCK_SIZE))" in text
    assert "writes [18, 20] to output_ptr[[8, 9]]" in text
    np.testing.assert_array_equal(out, 2 * x)        # the kernel really ran


def test_ops_carry_source_lines():
    x = np.zeros(16, np.float32)
    f = add_kernel.inspect(x, x, x, 16, BLOCK_SIZE=16).ir
    assert all(op.loc is not None for op in f.ops)
    store = f.ops[-1]
    assert store.opcode == "store" and "tl.store" in store.loc[1]
