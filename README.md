# tilec

A Triton-like tile language for Apple GPUs.

```python
import torch, tilec
x = torch.randn(1000, 3000, device="mps")
out = tilec.ext.sum(x, axis=1)        # (1000,)
```

## How it fits together

```
your kernel (@tilec.jit)
   │  Python: inspect.getsource + ast.parse            tilec/jit.py
   ▼
Python AST  ──►  Rust front end: AST -> IR            src/frontend.rs, src/ir.rs
                 Rust backend:   IR  -> Metal          src/codegen.rs
   ▼
Metal source ──► GPU, through torch.mps.compile_shader  tilec/jit.py
```

| file | job |
|---|---|
| `tilec/ext.py` | `sum(x, axis)`: picks BLOCK and CHUNKS, launches `_row_sum_kernel` |
| `tilec/language.py` | `tl`: `constexpr` and the builtins (`program_id`, `arange`, `load`, `store`, `sum`, `static_range`) |
| `tilec/jit.py` | `@jit`: parses the kernel into an AST, calls the Rust compiler, caches, launches |
| `src/lib.rs` | `tilec._core.compile(...)`: the entry point Python calls |
| `src/frontend.rs` | walks the Python AST, builds the IR |
| `src/ir.rs` | the IR: types, values (%0, %1, ...), ops |
| `src/codegen.rs` | IR -> Metal source (and which thread holds which element) |

## Build and run (needs Rust: https://rustup.rs)

```bash
pip install maturin
maturin develop --release         # compile src/*.rs into tilec/_core and install tilec
python examples/sum_demo.py
pip install pytest numpy && python -m pytest -q tests
```

After changing a `.rs` file, run `maturin develop --release` again.
