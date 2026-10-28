# tilec

A Triton-compatible tile language with an Apple GPU (Metal) backend.
Status: **Tier 1 skeleton** — runs Triton's vector-add and fused-softmax
tutorials unchanged except for the imports.

```python
import tilec as triton
import tilec.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK_SIZE=1024)
```

## Quick start (Mac)

```bash
pip install -e ".[mps,test]"       # needs PyTorch >= 2.5 for torch.mps.compile_shader
python examples/vector_add.py --mps
python examples/softmax.py --mps
pytest tests                        # tests/test_mps.py runs only where MPS exists
```

Without a Mac, everything except `--mps` runs on the NumPy interpreter.
`python examples/vector_add.py --show` prints the IR and the generated Metal code.

## Tracing: see every stage

```bash
python examples/trace_demo.py               # follows program 2 of a tiny kernel
python examples/trace_demo.py --mps         # same, running on the Apple GPU
TILEC_TRACE=1 python examples/softmax.py    # trace every launch in any script
TILEC_TRACE=1 TILEC_TRACE_PROGRAM=5 ...     # follow a different program instance
```

or in code: `kernel.trace[grid](args..., program=2)`. It runs the kernel and prints:

| Stage | File | Shows |
| --- | --- | --- |
| 1 | `jit.py` | each argument: runtime (and its IR type) or constexpr; grid, threads, backend |
| 2 | `frontend.py` | for each source line: its AST, the IR ops it produced, the names it bound |
| 3 | `ir.py` | the complete IR, every op tagged with its source line |
| 4 | `layout.py` + `codegen_msl.py` | the Metal code, with each block labelled by source line, IR op and layout |
| 5 | `interpreter.py` or `runtime_mps.py` | the actual value of every op for one program, or the GPU launch configuration |
| ✓ | | the output buffers |

## How it works

```
@jit Python function
  └─ frontend.py      Python AST → tile IR (constexpr folding, broadcasting, type promotion)
       ├─ interpreter.py   NumPy reference semantics (any machine; used as the test oracle)
       └─ codegen_msl.py   tile IR → Metal Shading Language, placed by layout.py
            └─ runtime_mps.py   torch.mps.compile_shader → Apple GPU
```

| File | Role |
| --- | --- |
| `tilec/language.py` | The `tl.*` surface: dtypes, `constexpr`, builtins |
| `tilec/ir.py` | SSA tile IR: types, values, ops, printer |
| `tilec/frontend.py` | Parses the kernel's Python AST into IR |
| `tilec/interpreter.py` | Executes IR with NumPy, one program at a time |
| `tilec/layout.py` | Which thread holds which tile element (1-D blocked today) |
| `tilec/codegen_msl.py` | IR → MSL |
| `tilec/runtime_mps.py` | Compiles and launches MSL through PyTorch MPS |
| `tilec/jit.py` | `@jit`, specialization cache, `kernel[grid](...)` launch |

**Mapping to Apple GPUs:** one program = one threadgroup; `num_warps` =
SIMD-groups per threadgroup (32 threads each); a tile lives in per-thread
register arrays placed by its layout (consecutive threads hold consecutive
elements, so memory accesses coalesce); reductions use `simd_sum/max/min`
then threadgroup memory across SIMD-groups. Runtime scalars are passed as
1-element buffers.

## What works (Tier 1)

| Area | Supported |
| --- | --- |
| Program model | `@jit`, `program_id`, `num_programs`, `constexpr`, `cdiv`, `kernel[grid]` incl. `grid=lambda meta: ...`, `num_warps` |
| Creation | `arange`, `zeros`, `full`, `cast`, `.to()`, `x[:, None]` |
| Memory | `load` (`mask`, `other`), `store` (`mask`), pointer arithmetic |
| Math | `+ - * / // % & \| ^ << >>`, comparisons, `where`, `maximum`, `minimum`, `exp`, `exp2`, `log`, `log2`, `sqrt`, `rsqrt`, `abs`, `sin`, `cos`, `floor`, `ceil`, `sigmoid`, `fma` |
| Reductions | `sum`, `max`, `min` (`axis`, `keep_dims`) |
| Control flow | `if` on constexpr values, `tl.static_range` / constant `range` (unrolled) |
| Types | i1, i8, i16, i32, i64, f16, bf16 (Metal only), f32; f64 is refused on Metal |

The **interpreter** runs N-D tiles. The **Metal backend** supports scalars
and 1-D tiles; N-D tiles raise a clear error pointing at Tier 2.

Integer `//` and `%` follow C semantics (truncate toward zero), like Triton.

## Next (Tier 2)

1. Runtime `for`/`while` loops and `if` (region ops in the IR).
2. N-D layouts (linear layouts) and `convert_layout`.
3. `tl.dot` lowered generically to `simdgroup_matrix` with threadgroup-memory staging.
4. `reshape`/`trans`, `make_block_ptr`, `autotune`.

## Not yet verified

The Metal path has been generated and inspected but not yet executed:
it was built on Linux. Run `pytest tests/test_mps.py` on a Mac to confirm
it, including the `threads=`/`group_size=` launch convention of
`torch.mps.compile_shader`.
