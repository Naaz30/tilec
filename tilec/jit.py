import ast
import inspect
import os
import textwrap
from dataclasses import dataclass

from . import language as tl
from ._core import compile as _compile

os.environ.setdefault("TILEC_OPT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                "mlir", "build", "tilec-opt"))

# each threadgroup is further devided into 4 SIMD groups
# as name suggests each SIMD group runs same instruction on diff data.
# while one SIMD group of a threadgroup is waiting the core
# might schedule a different SIMD group to execute

# 4 SIMD-groups in a theadgroup x 32 threads in an SIMD group = 128 threads per threadgroup
NUM_WARPS = 4                    
_LIBRARIES: dict = {}


def _arg_type(x) -> str:
    if str(getattr(x, "dtype", "")) == "torch.float32":
        return "ptr<f32>"
    if isinstance(x, int) and not isinstance(x, bool):
        return "i32"
    raise TypeError(f"unsupported kernel argument {x!r} (use float32 tensors and ints)")


@dataclass
class CompiledKernel:
    name: str
    ir: str                     
    msl: str                    


class JITFunction:
    def __init__(self, fn):
        self.fn = fn
        self.signature = inspect.signature(fn)
        self.constexpr_names = {n for n, p in self.signature.parameters.items()
                                if p.annotation is tl.constexpr}
        self.cache: dict = {}

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(grid, *args, **kwargs)

    def _bind(self, args, kwargs):
        params = dict(self.signature.bind(*args, **kwargs).arguments)
        constexprs = {k: v for k, v in params.items() if k in self.constexpr_names}
        runtime = {k: v for k, v in params.items() if k not in self.constexpr_names}
        return runtime, constexprs

    def compile(self, runtime, constexprs) -> CompiledKernel:
        types = {k: _arg_type(v) for k, v in runtime.items()}
        key = (tuple(sorted(types.items())), tuple(sorted((k, repr(v)) for k, v in constexprs.items())))
        if key not in self.cache:
            lines, first_line = inspect.getsourcelines(self.fn)
            source = textwrap.dedent("".join(lines))
            fdef = ast.parse(source).body[0]                    
            ir, msl = _compile(fdef, types, constexprs, self.fn.__globals__,
                               source, first_line, NUM_WARPS)
            self.cache[key] = CompiledKernel(self.fn.__name__, ir, msl)
        return self.cache[key]

    def inspect(self, *args, **kwargs) -> CompiledKernel:
        return self.compile(*self._bind(args, kwargs))

    def run(self, grid, *args, **kwargs):
        runtime, constexprs = self._bind(args, kwargs)
        ck = self.compile(runtime, constexprs)
        _launch(ck, grid, list(runtime.values()))
        return ck


def _launch(ck, grid, args):
    import torch
    if ck.msl not in _LIBRARIES:
        _LIBRARIES[ck.msl] = torch.mps.compile_shader(ck.msl)
    kernel = getattr(_LIBRARIES[ck.msl], ck.name)
    packed = [torch.tensor([a], dtype=torch.int32, device="mps") if isinstance(a, int) else a
              for a in args]                                   
    threads = NUM_WARPS * 32
    gx, gy, gz = (tuple(grid) + (1, 1))[:3]
    kernel(*packed, threads=[gx * threads, gy, gz], group_size=[threads, 1, 1])


def jit(fn):
    return JITFunction(fn)
