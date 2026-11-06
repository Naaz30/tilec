"""tilec: a Triton-like tile language for Apple GPUs.
Python parses a @jit kernel into an AST; the Rust core (tilec._core) turns
the AST into IR and the IR into Metal, which runs through PyTorch's MPS backend.
"""

from . import ext, language
from ._core import CompileError
from .jit import jit

__all__ = ["ext", "jit", "language", "CompileError"]
__version__ = "0.1.0"