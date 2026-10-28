"""Triton tutorial 01 (vector add), unchanged except for the imports.

    python examples/vector_add.py            # interpreter (any machine)
    python examples/vector_add.py --mps      # Apple GPU via Metal
    python examples/vector_add.py --show     # print the IR and generated MSL
"""

import sys

import numpy as np

import tilec as triton
import tilec.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x, y):
    output = x * 0 if hasattr(x, "device") else np.empty_like(x)
    n = output.numel() if hasattr(output, "numel") else output.size
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output


def main():
    n = 98432
    if "--mps" in sys.argv:
        import torch
        x = torch.rand(n, device="mps")
        y = torch.rand(n, device="mps")
        out = add(x, y)
        torch.mps.synchronize()
        err = (out - (x + y)).abs().max().item()
    else:
        rng = np.random.default_rng(0)
        x = rng.random(n, dtype=np.float32)
        y = rng.random(n, dtype=np.float32)
        out = add(x, y)
        err = float(np.abs(out - (x + y)).max())
    print(f"max abs error: {err}")

    if "--show" in sys.argv:
        x = np.zeros(n, np.float32)
        ck = add_kernel.inspect(x, x, x, n, BLOCK_SIZE=1024)
        print("\n--- IR ---\n" + str(ck.ir))
        print("\n--- MSL ---\n" + ck.msl)


if __name__ == "__main__":
    main()
