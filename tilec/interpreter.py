"""Reference interpreter: runs tile IR with NumPy, one program at a time.

This defines what every op *means*.  The Metal backend is tested against it,
so keep it simple and obviously correct rather than fast.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import ir


class Ptr:
    """A pointer (or tile of pointers): which buffer, plus element offsets."""

    def __init__(self, buf, offsets):
        self.buf = buf                      
        self.offsets = np.asarray(offsets, dtype=np.int64)


def _np(dt):
    if dt.np is None:
        raise NotImplementedError(f"the interpreter does not support {dt.name}")
    return dt.np


def _c_div(a, b):
    """Integer division truncating toward zero (C / Triton semantics)."""
    q = np.abs(a) // np.abs(b)
    return np.where((a < 0) ^ (b < 0), -q, q).astype(a.dtype)


def _c_mod(a, b):
    return np.fmod(a, b).astype(np.result_type(a, b))


_BINARY = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply,
    "div": np.true_divide, "and": np.bitwise_and, "or": np.bitwise_or,
    "xor": np.bitwise_xor, "shl": np.left_shift, "shr": np.right_shift,
    "max": np.maximum, "min": np.minimum,
}
_COMPARE = {
    "lt": np.less, "le": np.less_equal, "gt": np.greater,
    "ge": np.greater_equal, "eq": np.equal, "ne": np.not_equal,
}
_UNARY = {
    "exp": np.exp, "exp2": np.exp2, "log": np.log, "log2": np.log2,
    "sqrt": np.sqrt, "rsqrt": lambda x: 1 / np.sqrt(x), "abs": np.abs,
    "sin": np.sin, "cos": np.cos, "floor": np.floor, "ceil": np.ceil,
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
}
_REDUCE = {"sum": np.sum, "max": np.max, "min": np.min}


def run(func: ir.Function, grid, buffers, scalars, observer=None, observe_pid=None):
    """Execute ``func`` over ``grid``.

    buffers: list of flat numpy arrays, one per pointer parameter (in order).
    scalars: dict param-name -> python number for scalar parameters.
    Writes happen in place on ``buffers``.
    observer(op, args, value), if given, is called after every op of the program
    whose id equals ``observe_pid`` (a 3-tuple); used for tracing.
    """
    grid = tuple(grid) + (1,) * (3 - len(grid))
    for pz, py, px in itertools.product(range(grid[2]), range(grid[1]), range(grid[0])):
        pid = (px, py, pz)
        obs = observer if (observer is not None and pid == observe_pid) else None
        _run_program(func, pid, grid, buffers, scalars, obs)


def _run_program(func, pid, grid, buffers, scalars, observer=None):
    env = {}
    buf_i = 0
    for name, v in func.params:
        if v.type.is_ptr:
            env[v.id] = Ptr(buf_i, 0)
            buf_i += 1
        else:
            env[v.id] = np.asarray(scalars[name], dtype=_np(v.type.elem))

    for op in func.ops:
        args = [env[o.id] for o in op.operands]
        out = _execute(op, args, pid, grid, buffers)
        if op.result is not None:
            env[op.result.id] = out
        if observer is not None:
            observer(op, args, out)


def _execute(op, args, pid, grid, buffers):
    a = op.attrs
    rt = op.result.type if op.result is not None else None
    code = op.opcode

    if code == "program_id":
        return np.int32(pid[a["axis"]])
    if code == "num_programs":
        return np.int32(grid[a["axis"]])
    if code == "constant":
        return np.asarray(a["value"], dtype=_np(rt.elem))
    if code == "arange":
        return np.arange(a["start"], a["end"], dtype=np.int32)
    if code == "broadcast":
        x = args[0]
        if isinstance(x, Ptr):
            return Ptr(x.buf, np.broadcast_to(x.offsets, rt.shape))
        return np.broadcast_to(x, rt.shape)
    if code == "expand_dims":
        x = args[0]
        if isinstance(x, Ptr):
            return Ptr(x.buf, np.expand_dims(x.offsets, a["axis"]))
        return np.expand_dims(x, a["axis"])
    if code == "cast":
        return np.asarray(args[0]).astype(_np(rt.elem))
    if code == "binary":
        x, y = args
        dt = _np(rt.elem)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if a["kind"] == "floordiv":
                return _c_div(np.asarray(x), np.asarray(y))
            if a["kind"] == "mod":
                if np.issubdtype(np.asarray(x).dtype, np.floating):
                    return np.fmod(x, y).astype(dt)
                return _c_mod(np.asarray(x), np.asarray(y))
            return np.asarray(_BINARY[a["kind"]](x, y)).astype(dt)
    if code == "compare":
        return _COMPARE[a["kind"]](args[0], args[1])
    if code == "select":
        return np.where(args[0], args[1], args[2]).astype(_np(rt.elem))
    if code == "unary":
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return np.asarray(_UNARY[a["fn"]](args[0])).astype(_np(rt.elem))
    if code == "addptr":
        p, off = args
        return Ptr(p.buf, p.offsets + np.asarray(off, dtype=np.int64))
    if code == "load":
        return _load(args, a, rt, buffers)
    if code == "store":
        _store(args, a, buffers)
        return None
    if code == "reduce":
        x = np.asarray(args[0])
        return np.asarray(_REDUCE[a["kind"]](x, axis=a["axis"])).astype(_np(rt.elem))
    raise NotImplementedError(f"interpreter: unknown op {code}")


def _check_bounds(buf, offs, what):
    if offs.size and (offs.min() < 0 or offs.max() >= buf.size):
        bad = offs[(offs < 0) | (offs >= buf.size)][0]
        raise IndexError(f"out-of-bounds {what} at element offset {bad} "
                         f"(buffer has {buf.size} elements); did you forget a mask?")


def _load(args, attrs, rt, buffers):
    ptr = args[0]
    buf = buffers[ptr.buf]
    offs = np.broadcast_to(ptr.offsets, rt.shape)
    if attrs["has_mask"]:
        mask = np.broadcast_to(args[1], rt.shape)
        _check_bounds(buf, offs[mask], "load")
        safe = np.where(mask, offs, 0)
        vals = buf[safe] if buf.size else np.zeros(rt.shape, buf.dtype)
        other = args[2] if attrs["has_other"] else np.zeros((), buf.dtype)
        return np.where(mask, vals, other).astype(buf.dtype)
    _check_bounds(buf, offs, "load")
    return buf[offs].copy() if offs.ndim else buf[int(offs)].copy()


def _store(args, attrs, buffers):
    ptr, val = args[0], np.asarray(args[1])
    buf = buffers[ptr.buf]
    offs = np.broadcast_to(ptr.offsets, val.shape)
    if attrs["has_mask"]:
        mask = np.broadcast_to(args[2], val.shape)
        offs, val = offs[mask], val[mask]
    _check_bounds(buf, np.atleast_1d(offs), "store")
    buf[offs] = val.astype(buf.dtype)
