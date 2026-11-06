import numpy as np
import torch

import tilec as triton
import tilec.language as tl


@triton.jit
def row_sum(x_ptr, out_ptr, R: tl.constexpr, C: tl.constexpr):
    rows = tl.arange(0, R)
    cols = tl.arange(0, C)
    x = tl.load(x_ptr + rows[:, None] * C + cols[None, :])
    tl.store(out_ptr + rows, tl.sum(x, axis=1))

    out_ptr = tl.ext.sum(x_ptr, axis = 1)


x = np.arange(32, dtype=np.float32).reshape(4, 8)
print("input:")
print(x)


x_gpu = torch.from_numpy(x).to("mps")
out_gpu = torch.zeros(4, dtype=torch.float32, device="mps")
row_sum[(1,)](x_gpu, out_gpu, R=4, C=8)
torch.mps.synchronize()                  
print("GPU (Metal): ", out_gpu.cpu().numpy())

print("numpy:      ", x.sum(axis=1))