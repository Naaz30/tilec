"""Triton tutorial 02 (fused softmax), one program per row.

    python examples/softmax.py          # interpreter
    python examples/softmax.py --mps    # Apple GPU via Metal
"""

import sys

import numpy as np

import tilec as triton
import tilec.language as tl


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def softmax(x, out):
    n_rows, n_cols = x.shape
    block = next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](out, x, n_cols, n_cols, n_cols, BLOCK_SIZE=block)
    return out


def reference(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 781)).astype(np.float32)
    if "--mps" in sys.argv:
        import torch
        xt = torch.from_numpy(x).to("mps")
        out = softmax(xt, torch.empty_like(xt))
        torch.mps.synchronize()
        got = out.cpu().numpy()
    else:
        got = softmax(x, np.empty_like(x))
    print(f"max abs error: {np.abs(got - reference(x)).max():.3e}")


if __name__ == "__main__":
    main()
