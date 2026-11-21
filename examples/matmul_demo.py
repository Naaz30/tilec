import torch

import tilec

M, K, N = 4096, 4096, 4096
x1 = torch.randn(M, K, device="mps")
x2 = torch.randn(K, N, device="mps")


def gflops(f, reps=20):
    f()
    torch.mps.synchronize()
    start = torch.mps.Event(enable_timing=True)
    end = torch.mps.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        f()
    end.record()
    torch.mps.synchronize()
    sec = start.elapsed_time(end) / 1000 / reps
    return 2 * M * N * K / sec / 1e9, sec * 1000


out = tilec.ext.matrix_mul(x1, x2)
torch.mps.synchronize()
print("max error:", (out - x1 @ x2).abs().max().item())

from pathlib import Path

for ck in tilec.ext._matmul_tiled.cache.values():
    Path(__file__).with_name(f"{ck.name}.metal").write_text(ck.msl)

for name, f in [("tilec", lambda: tilec.ext.matrix_mul(x1, x2)), ("torch", lambda: x1 @ x2)]:
    g, ms = gflops(f)
    print(f"{name}: {ms:.2f} ms  {g:.1f} GFLOPS")