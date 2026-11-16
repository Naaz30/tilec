"""Multiply two matrices on the Apple GPU"""

import torch

import tilec

x1 = torch.arange(4096, dtype=torch.float32, device="mps").reshape(64, 64)
x2 = torch.arange(1024, dtype=torch.float32, device="mps").reshape(64, 16)

out = tilec.ext._matrix_mul_tiled(x1,x2)
print(out)
