import sys

import torch

import tilec

x1 = torch.randn(4096, 4096, device="mps")
x2 = torch.randn(4096, 4096, device="mps")

tilec.ext.matrix_mul(x1, x2)
x1 @ x2
torch.mps.synchronize()

if len(sys.argv) > 1 and sys.argv[1] == "capture":
    with torch.mps.profiler.metal_capture("mm"):
        tilec.ext.matrix_mul(x1, x2)
        x1 @ x2
        torch.mps.synchronize()
else:
    with torch.mps.profiler.profile(mode="interval", wait_until_completed=True):
        for _ in range(5):
            tilec.ext.matrix_mul(x1, x2)
            x1 @ x2
        torch.mps.synchronize()