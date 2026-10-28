"""The kernel-side language: a Triton-compatible surface (Tier 1).

Kernels import this as ``tl`` exactly like ``triton.language``.  Nothing in
here executes on its own: the front end recognises these objects while it
walks a kernel's source and turns them into IR.  Calling them from ordinary
Python raises, except where a host-side meaning exists (``cdiv``).
"""

from __future__ import annotations

import numpy as np


class dtype:
    """A scalar element type.  Instances are singletons compared by identity."""

    def __init__(self, name, np_dtype, msl, kind, bits):
        self.name = name          
        self.np = np_dtype       
        self.msl = msl           
        self.kind = kind          
        self.bits = bits

    @property
    def is_float(self):
        return self.kind == "float"

    @property
    def is_int(self):
        return self.kind == "int"

    @property
    def is_bool(self):
        return self.kind == "bool"

    def __repr__(self):
        return f"tl.{_LONG_NAMES[self.name]}"

    def __str__(self):
        return self.name


int1 = dtype("i1", np.bool_, "bool", "bool", 1)
int8 = dtype("i8", np.int8, "char", "int", 8)
int16 = dtype("i16", np.int16, "short", "int", 16)
int32 = dtype("i32", np.int32, "int", "int", 32)
int64 = dtype("i64", np.int64, "long", "int", 64)
float16 = dtype("f16", np.float16, "half", "float", 16)
bfloat16 = dtype("bf16", None, "bfloat", "float", 16)   
float32 = dtype("f32", np.float32, "float", "float", 32)
float64 = dtype("f64", np.float64, None, "float", 64)  

_LONG_NAMES = {
    "i1": "int1", "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
    "f16": "float16", "bf16": "bfloat16", "f32": "float32", "f64": "float64",
}
ALL_DTYPES = [int1, int8, int16, int32, int64, float16, bfloat16, float32, float64]

bool_ = int1
float = float32 


def dtype_from_numpy(np_dtype):
    np_dtype = np.dtype(np_dtype)
    for d in ALL_DTYPES:
        if d.np is not None and np.dtype(d.np) == np_dtype:
            return d
    raise TypeError(f"unsupported array dtype {np_dtype}")


class constexpr:
    """Marks a kernel parameter as a compile-time constant.

    Used as an annotation (``BLOCK: tl.constexpr``), the same as in Triton.
    Each distinct value of a constexpr parameter produces its own compiled kernel.
    """

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"constexpr({self.value!r})"


class Builtin:
    """A function that only has meaning inside a kernel."""

    def __init__(self, name, host_impl=None):
        self.name = name
        self.host_impl = host_impl

    def __call__(self, *args, **kwargs):
        if self.host_impl is not None:
            return self.host_impl(*args, **kwargs)
        raise RuntimeError(
            f"tl.{self.name} can only be called inside a @jit kernel")

    def __repr__(self):
        return f"<tl.{self.name}>"


def _cdiv(a, b):
    return (a + b - 1) // b


_BUILTIN_NAMES = [
    
    "program_id", "num_programs",
    
    "arange", "zeros", "full", "cast",
   
    "load", "store",
    
    "where", "maximum", "minimum", "exp", "exp2", "log", "log2", "sqrt",
    "rsqrt", "abs", "sin", "cos", "floor", "ceil", "sigmoid", "fma",
    
    "sum", "max", "min",
    
    "static_range", "static_assert", "static_print",
    
    "multiple_of", "max_contiguous", "max_constancy",
]

BUILTINS = {name: Builtin(name) for name in _BUILTIN_NAMES}
BUILTINS["cdiv"] = Builtin("cdiv", host_impl=_cdiv)
globals().update(BUILTINS)
