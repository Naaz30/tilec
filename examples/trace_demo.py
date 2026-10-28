"""Run a small kernel and print what every stage of the compiler does.

    python examples/trace_demo.py                 # interpreter, follows program 2
    python examples/trace_demo.py --program 0     # follow a different program
    python examples/trace_demo.py --mps           # run on the Apple GPU instead

The same trace is available for any launch in any script:

    TILEC_TRACE=1 python examples/softmax.py
"""

import sys

import numpy as np

import tilec as triton
import tilec.language as tl


@triton.jit
def scale_kernel(x_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * alpha, mask=mask)


def main():
    program = int(sys.argv[sys.argv.index("--program") + 1]) if "--program" in sys.argv else 2
    n, block = 10, 4                       # tiny, so every value fits on screen
    grid = (triton.cdiv(n, block),)        # 3 programs

    if "--mps" in sys.argv:
        import torch
        x = torch.arange(1, n + 1, dtype=torch.float32, device="mps")
        out = torch.zeros(n, device="mps")
    else:
        x = np.arange(1, n + 1, dtype=np.float32)    # [1, 2, ..., 10]
        out = np.zeros(n, np.float32)

    scale_kernel.trace[grid](x, out, n, 2.0, BLOCK=block, program=program)


if __name__ == "__main__":
    main()
