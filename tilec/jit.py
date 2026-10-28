"""@jit decorator, specialization cache and launch (``kernel[grid](...)``)."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field

import numpy as np

from . import interpreter, ir
from . import language as tl
from .frontend import build_ir

_LAUNCH_OPTIONS = {"num_warps", "num_stages", "backend"}


def _is_torch_tensor(x):
    return type(x).__module__.startswith("torch") and hasattr(x, "device")


def _arg_type(x) -> ir.Type:
    if isinstance(x, np.ndarray):
        return ir.Type(ir.PtrType(tl.dtype_from_numpy(x.dtype)))
    if _is_torch_tensor(x):
        from .runtime_mps import torch_dtype_to_tl
        return ir.Type(ir.PtrType(torch_dtype_to_tl(x.dtype)))
    if isinstance(x, (bool, np.bool_)):
        return ir.Type(tl.int1)
    if isinstance(x, (int, np.integer)):
        return ir.Type(tl.int32 if -2**31 <= int(x) < 2**31 else tl.int64)
    if isinstance(x, (float, np.floating)):
        return ir.Type(tl.float32)
    raise TypeError(f"unsupported kernel argument {x!r}")


@dataclass
class CompiledKernel:
    ir: ir.Function
    num_warps: int
    _msl: str | None = field(default=None, repr=False)

    @property
    def msl(self) -> str:
        if self._msl is None:
            from .codegen_msl import generate_msl
            self._msl = generate_msl(self.ir, self.num_warps)
        return self._msl


class JITFunction:
    __tilec_jit__ = True

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.signature = inspect.signature(fn)
        self.constexpr_names = set()
        for name, p in self.signature.parameters.items():
            ann = p.annotation
            if ann is tl.constexpr or (isinstance(ann, str) and ann.split(".")[-1] == "constexpr"):
                self.constexpr_names.add(name)
        self.cache: dict = {}

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"launch kernels with {self.__name__}[grid](...)")

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(grid, *args, **kwargs)

    @property
    def trace(self):
        """``kernel.trace[grid](args..., program=0)``: run and print every stage."""
        jit_fn = self

        class _Tracer:
            def __getitem__(self, grid):
                def launch(*args, program=0, **kwargs):
                    from .trace import trace_launch
                    return trace_launch(jit_fn, grid, args, kwargs, program=program)
                return launch
        return _Tracer()

    def _bind(self, args, kwargs):
        options = {k: kwargs.pop(k) for k in list(kwargs) if k in _LAUNCH_OPTIONS}
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)
        constexprs = {}
        for name in self.constexpr_names:
            v = params[name]
            constexprs[name] = v.value if isinstance(v, tl.constexpr) else v
        runtime = {k: v for k, v in params.items() if k not in self.constexpr_names}
        return runtime, constexprs, options

    def specialize(self, runtime, constexprs, num_warps) -> CompiledKernel:
        arg_types = {k: _arg_type(v) for k, v in runtime.items()}
        key = (tuple(sorted((k, str(t)) for k, t in arg_types.items())),
               tuple(sorted((k, repr(v)) for k, v in constexprs.items())), num_warps)
        ck = self.cache.get(key)
        if ck is None:
            ck = CompiledKernel(build_ir(self.fn, arg_types, constexprs), num_warps)
            self.cache[key] = ck
        return ck

    def inspect(self, *args, num_warps=4, **kwargs) -> CompiledKernel:
        """Compile for these arguments without running; returns .ir and .msl."""
        runtime, constexprs, _ = self._bind(args, kwargs)
        return self.specialize(runtime, constexprs, num_warps)

    def run(self, grid, *args, **kwargs):
        if os.environ.get("TILEC_TRACE", "") not in ("", "0"):
            from .trace import trace_launch
            program = int(os.environ.get("TILEC_TRACE_PROGRAM", "0"))
            return trace_launch(self, grid, args, kwargs, program=program)

        runtime, constexprs, options = self._bind(args, dict(kwargs))
        num_warps = int(options.get("num_warps", 4))
        grid = self.resolve_grid(grid, runtime, constexprs)
        ck = self.specialize(runtime, constexprs, num_warps)
        backend = self.choose_backend(options, runtime)
        self.launch(ck, grid, runtime, backend, num_warps)
        return ck

    @staticmethod
    def resolve_grid(grid, runtime, constexprs):
        if callable(grid):
            grid = grid(dict(constexprs, **runtime))
        if isinstance(grid, int):
            grid = (grid,)
        return tuple(int(g) for g in grid)

    def choose_backend(self, options, runtime):
        return options.get("backend") or os.environ.get("TILEC_BACKEND") \
            or self._default_backend(runtime)

    def launch(self, ck, grid, runtime, backend, num_warps, observer=None, observe_pid=None):
        if backend == "interpreter":
            self._run_interpreter(ck, grid, runtime, observer, observe_pid)
        elif backend == "mps":
            from .runtime_mps import launch
            ordered = [runtime[name] for name, _ in ck.ir.params]
            launch(ck.msl, ck.ir.name, ck.ir.params, ordered, grid, num_warps)
        else:
            raise ValueError(f"unknown backend {backend!r} (use 'interpreter' or 'mps')")

    @staticmethod
    def _default_backend(runtime):
        for v in runtime.values():
            if _is_torch_tensor(v) and v.device.type == "mps":
                return "mps"
        return "interpreter"

    @staticmethod
    def _run_interpreter(ck, grid, runtime, observer=None, observe_pid=None):
        buffers, write_back, scalars = [], [], {}
        for name, v in ck.ir.params:
            a = runtime[name]
            if not v.type.is_ptr:
                scalars[name] = a
                continue
            if isinstance(a, np.ndarray):
                if not a.flags.c_contiguous:
                    raise ValueError(f"argument '{name}' must be contiguous")
                buffers.append(a.reshape(-1))
            elif _is_torch_tensor(a):
                if a.device.type == "cpu":
                    buffers.append(a.contiguous().view(-1).numpy())
                else:                              # copy device tensors through the host
                    host = a.detach().cpu().contiguous().view(-1).numpy().copy()
                    buffers.append(host)
                    write_back.append((a, host))
            else:
                raise TypeError(f"argument '{name}' must be an array or tensor")
        interpreter.run(ck.ir, grid, buffers, scalars, observer, observe_pid)
        for tensor, host in write_back:
            import torch
            tensor.copy_(torch.from_numpy(host).view(tensor.shape))


def jit(fn=None, **_ignored):
    """Decorator turning a Python function into a tile kernel (like triton.jit)."""
    if fn is None:
        return lambda f: JITFunction(f)
    return JITFunction(fn)
