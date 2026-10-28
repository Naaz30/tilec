"""Layouts: which thread holds which element of a tile.

Tier 1 has one layout, a 1-D blocked layout over a threadgroup of
``threads`` threads:

* ``n >= threads``: each thread holds ``n / threads`` elements in registers,
  element ``e = r * threads + tid`` for register slot ``r``.  Consecutive
  threads hold consecutive elements, so loads and stores coalesce.
* ``n < threads``: element ``tid % n``, so the tile is *replicated* across
  threads (as in Triton).  Only "owner" threads (``tid < n``) store or
  contribute to reductions, so nothing is written or counted twice.

Tier 2 replaces this with general N-D layouts (linear layouts) and adds
layout conversions; this module is where that work goes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Blocked1D:
    n: int
    threads: int

    @property
    def regs(self) -> int:
        """Elements held per thread."""
        return max(1, self.n // self.threads)

    @property
    def replicated(self) -> bool:
        return self.n < self.threads

    def element(self, r: str, tid: str) -> str:
        """MSL expression for the element index held in register slot ``r``."""
        if self.replicated:
            return f"({tid} % {self.n}u)"
        return f"({r} * {self.threads}u + {tid})"

    def owner_guard(self, tid: str) -> str | None:
        """MSL condition for threads that own (not just replicate) their element."""
        return f"({tid} < {self.n}u)" if self.replicated else None


def layout_for(shape, threads):
    if len(shape) == 0:
        return None
    if len(shape) == 1:
        n = shape[0]
        if n % threads != 0 and n > threads:
            raise NotImplementedError(
                f"tile of {n} elements does not divide evenly over {threads} threads")
        return Blocked1D(n, threads)
    raise NotImplementedError(
        f"{len(shape)}-D tiles need the N-D layout pass (Tier 2); the Metal "
        "backend currently supports scalars and 1-D tiles. "
        "Use backend='interpreter' to run this kernel today.")
