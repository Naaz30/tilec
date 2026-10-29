"""Layouts: which thread holds which element of a tile.

One layout, ``Blocked``, for tiles of any rank, over a threadgroup of
``threads`` threads:

* The tile is flattened in row-major order: element (i0, i1, ..., ik) has
  flat index  e = i0 * s0 + i1 * s1 + ... + ik,  where s are the row-major
  strides of the shape.
* Flat indices are dealt out to threads round-robin: register slot r of
  thread tid holds  e = r * threads + tid.  Consecutive threads hold
  consecutive elements, so loads and stores coalesce.
* Each thread has ceil(n / threads) slots.  Three cases:

    n % threads == 0, n >= threads   every slot is a real element
    ragged  (n > threads, uneven)    the last slot of some threads is empty
                                     (e >= n); they "wrap around" one fewer time
    replicated  (n < threads)        one slot, e = tid % n: several threads hold
                                     the same element; only tid < n "own" it

``valid`` gives the condition for a slot holding a real, owned element.
Stores and reductions must respect it; elementwise ops may ignore it.

Ops that change shape (broadcast between tiles, reduce along an axis) still
need code-generation support before N-D tiles run on the GPU; see
codegen_msl.py.  This module only answers "who holds what".
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Blocked:
    shape: tuple
    threads: int

    #----------------- returns actual sizes, dims, and bools ----------------- #
    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def n(self) -> int:
        """Total number of elements."""
        return math.prod(self.shape)

    @property
    def regs(self) -> int:
        """Register slots per thread: ceil(n / threads), at least 1."""
        return max(1, -(-self.n // self.threads))

    @property
    def replicated(self) -> bool:
        """Fewer elements than threads: several threads share each element."""
        return self.n < self.threads

    @property
    def ragged(self) -> bool:
        """More elements than threads, but not a multiple: some last slots are empty."""
        return self.n > self.threads and self.n % self.threads != 0

    @property
    def strides(self) -> tuple:
        """Row-major strides of the shape, e.g. (2, 4) -> (4, 1)."""
        out, acc = [], 1
        for size in reversed(self.shape):
            out.append(acc)
            acc *= size
        return tuple(reversed(out))

    # ----------------- returns MSL expressions --------------------- #
    def flat(self, r: str, tid: str) -> str:
        """Flat (row-major) element index held in register slot r."""
        if self.replicated:
            return f"({tid} % {self.n}u)"
        return f"({r} * {self.threads}u + {tid})"

    def coord(self, axis: int, r: str, tid: str) -> str:
        """Coordinate along ``axis`` of the element held in slot r."""
        if not 0 <= axis < self.rank:
            raise ValueError(f"axis {axis} out of range for a rank-{self.rank} tile")
        e = self.flat(r, tid)
        stride, size = self.strides[axis], self.shape[axis]
        if self.rank == 1:
            return e
        if axis == 0:
            return f"({e} / {stride}u)"
        if stride == 1:
            return f"({e} % {size}u)"
        return f"(({e} / {stride}u) % {size}u)"

    def valid(self, r: str, tid: str) -> str | None:
        """Condition for slot r holding a real element this thread owns (None = always)."""
        if self.replicated:
            return f"({tid} < {self.n}u)"
        if self.ragged:
            return f"({self.flat(r, tid)} < {self.n}u)"
        return None

    def element(self, r: str, tid: str) -> str:
        """Element index of a 1-D tile (same as flat)."""
        return self.flat(r, tid)

    def owner_guard(self, tid: str, r: str = "r") -> str | None:
        """Guard for stores and reductions; codegen calls it inside its `for r` loops."""
        return self.valid(r, tid)

    # helperr for debugging
    def holder(self, index) -> tuple:
        """(tid, r) of the first thread holding the element at ``index`` (for tests/docs)."""
        if isinstance(index, int):
            index = (index,)
        e = sum(i * s for i, s in zip(index, self.strides))
        if not 0 <= e < self.n:
            raise IndexError(f"index {index} out of range for shape {self.shape}")
        if self.replicated:
            return (e, 0)
        return (e % self.threads, e // self.threads)


# Older name, kept so existing imports keep working.
Blocked1D = Blocked


def layout_for(shape, threads):
    """Layout for a value of this shape (None for scalars)."""
    if len(shape) == 0:
        return None
    if any(s <= 0 for s in shape):
        raise ValueError(f"tile shape {tuple(shape)} has an empty dimension")
    return Blocked(tuple(shape), threads)