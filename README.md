# tilec

GPU ops for Apple Silicon, built on a Triton-like tile compiler.

**Using ops.** Call an op on PyTorch MPS tensors. No GPU knowledge is needed: tilec defines the kernel, the tiling and the launch configuration, and picks the configuration for each input shape with a model trained on this machine.

```python
import torch, tilec
x1 = torch.randn(512, 4096, device="mps")
x2 = torch.randn(4096, 64, device="mps")
out = tilec.ext.matrix_mul(x1, x2)
s = tilec.ext.sum(x1, axis=1)
```

**Writing ops.** A new op is a short Python kernel plus host code in `tilec/ext.py`. The kernel describes what one program instance (one threadgroup) does with 1-D tiles. The compiler handles everything below that: which thread holds which element, coalesced memory access, reductions across SIMD groups and threadgroup memory, barriers, and keeping loop-invariant loads in registers. It emits Metal that PyTorch's MPS backend runs directly on the tensors' memory.


## Pipeline

```
@tilec.jit kernel
   │  inspect + ast.parse                                        tilec/jit.py
   ▼
Python AST ──► frontend: AST → tile IR (SSA, 1-D tiles)          src/frontend.rs, src/ir.rs
   │
   ├─► MLIR path (default)
   │     tile IR → per-thread MLIR (gpu/scf/arith/memref)        src/mlir.rs
   │     canonicalize, CSE, LICM, load hoisting → Metal          mlir/tilec_opt.cpp  (tilec-opt)
   │
   └─► direct path (TILEC_OPT= empty)
         tile IR → Metal                                         src/codegen.rs
   ▼
Metal source ──► torch.mps.compile_shader ──► GPU                tilec/jit.py
```


## Optimizations implemented

### Frontend and specialization

- **JIT specialization.** Kernels are compiled per (argument types, `constexpr` values). Constexpr parameters are partially evaluated: tile sizes, chunk counts and row counts are compile-time constants in the generated code.
- **Constant folding.** Python-level integer and float expressions on constants are folded in the frontend and never reach the IR.
- **Full unrolling.** `tl.static_range` loops are unrolled at compile time; `range` loops stay as runtime loops (`loop_begin`/`loop_end` → `scf.for`).
- **Implicit broadcasting.** NumPy-style shape broadcasting is resolved statically; broadcasting a scalar costs nothing after lowering (every register slot aliases the same SSA value).

### Lowering: data layout and reductions

- **Static layout assignment.** Each 1-D tile gets a blocked-cyclic distribution over the threadgroup: element `e = r·T + tid` lives in register slot `r` of thread `tid` (`T` = 128). Consecutive threads touch consecutive addresses, so global loads and stores are coalesced.
- **Replicated layout with ownership predication.** Tiles smaller than the threadgroup are replicated (`tid mod n`); side effects and reductions are predicated on `tid < n` so each element is stored and counted once.
- **Scalarization.** Tiles are lowered to independent per-thread SSA scalars, one per register slot, rather than to local arrays, which exposes every element to scalar optimizations.
- **Hierarchical reduction.** `tl.sum` lowers to a thread-local reduction over register slots, a SIMD-group reduction (`gpu.subgroup_reduce` → `simd_sum`), and a cross-SIMD-group combine through threadgroup memory with a single barrier. Inside runtime loops a second barrier is inserted to remove the write-after-read hazard on the scratch buffer.
- **If-conversion of masked loads.** A masked load lowers to branch-free code: the index is clamped with `select` so the access is always in bounds, and the loaded value is selected against the `other` value. This makes every load safe to speculate, which the hoisting pass depends on.

### MLIR mid-end (`tilec-opt`)

- **Canonicalization**: constant folding, algebraic simplification (`x + 0`, `x · 1`), dead code elimination, constant hoisting to the entry block.
- **Common subexpression elimination**: removes recomputed index arithmetic and duplicate mask comparisons (for example the identical column masks of both matmul operands).
- **Loop-invariant code motion** for side-effect-free ops.
- **Load hoisting (custom pass).** MLIR's LICM does not move memory reads, because a store in the loop may alias. `tilec-opt` adds a memory-aware LICM: a `memref.load` is moved out of an `scf.for` when its operands are loop-invariant and its buffer is read-only in the kernel, assuming distinct kernel arguments do not alias (C `restrict` semantics). In `_matrix_mul` this keeps a row of `x1` in registers across the loop over output columns, cutting `x1` traffic by a factor of `M2`.
- **Read-only inference.** Buffers that are never stored to are emitted as `device const`, giving Apple's compiler aliasing and caching freedom.

### Schedule-level transformations

Exposed as kernel parameters and applied by the host code in `ext.py`:

- **Tile size selection** (`BLOCK`, `CHUNKS`): register footprint vs. number of reduction rounds and masked-lane waste.
- **Split-K / reduction splitting** (`split`): a reduction dimension is partitioned across threadgroups to raise parallelism when the outer dimension is small; partial results are combined in a second reduction pass.
- **Thread-block coarsening** (`join`): one threadgroup processes several rows to amortize scheduling cost when there are many short rows.

## Future work

- **Loop-carried values.** Support `scf.for` `iter_args` so accumulators can live in registers across runtime loops. This enables the next item.
- **Reduction deferral.** Accumulate partial products per thread across all chunks and perform a single cross-thread reduction per output, instead of a full SIMD + threadgroup reduction (with barriers) for every chunk.
- **Loop interchange and register tiling for matmul.** Reorder loops so one loaded `x2` chunk is reused across all coarsened rows, giving an outer-product formulation with reuse of both operands in registers.
- **Threadgroup-memory tiling.** Cooperative loads of `x1`/`x2` tiles into threadgroup memory, with software pipelining (double buffering) to overlap loads with compute.
- **Matrix intrinsics.** Lower tile matmuls to Apple's `simdgroup_matrix` (8×8 SIMD-group matrix multiply-accumulate).
- **Vectorized memory access.** Widen contiguous per-thread accesses to `float4` loads and stores.
- **Barrier elimination.** Remove redundant `threadgroup_barrier`s (back-to-back or with no intervening shared-memory access).
- **A tile-level dialect.** Keep tiles and their layouts as first-class types in MLIR (as Triton's TritonGPU dialect does), so layout propagation, layout conversion, split-K and coarsening become IR rewrites instead of hand-written kernel variants.
- **Multi-dimensional grids.** `tl.program_id(1)` to run split-K in a single launch, with the combine fused into the kernel (atomics or a last-block reduction).
- **Segmented reductions.** Reduce several short rows within one threadgroup in parallel, so coarsened short rows keep all threads busy.
- **Finer alias analysis.** Replace the argument-level no-alias assumption with access-range analysis, so loads can be hoisted even when the same buffer is also written.
