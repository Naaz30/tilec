"""tilec: a Triton-compatible tile language with an Apple GPU (Metal) backend.

    import tilec as triton
    import tilec.language as tl
"""

from . import language
from .frontend import CompileError
from .jit import CompiledKernel, JITFunction, jit
from .language import cdiv

__all__ = ["jit", "cdiv", "language", "CompileError", "CompiledKernel", "JITFunction"]
__version__ = "0.1.0"
