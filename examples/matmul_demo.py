from pathlib import Path

import torch

import tilec
from tilec.ext import _matmul_tiled, _matrix_mul_tiled

SIZES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (3072, 3072, 3072),
    (4096, 4096, 4096),
    (6144, 6144, 6144),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
    (4096, 1024, 4096),
    (1024, 8192, 1024),
    (8192, 512, 8192),
]
TILES = [(64, 64, 16), (64, 64, 32), (32, 32, 16), (128, 64, 16), (64, 128, 16)]


def gflops(f, M, N, K):
    flops = 2 * M * N * K
    reps = max(3, min(20, int(2e12 / flops)))
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
    return flops / sec / 1e9, sec * 1000


for M, K, N in SIZES:
    print(f"\nM={M} K={K} N={N}")
    try:
        x1 = torch.randn(M, K, device="mps")
        x2 = torch.randn(K, N, device="mps")
        want = x1 @ x2
        torch.mps.synchronize()
    except RuntimeError as e:
        print(f"  skipped: {str(e).splitlines()[0]}")
        torch.mps.empty_cache()
        continue
    g, ms = gflops(lambda: x1 @ x2, M, N, K)
    print(f"  {'torch':<28} {ms:9.2f} ms  {g:7.1f} GFLOPS")

    for BM, BN, BK in TILES:
        label = f"tilec BM={BM} BN={BN} BK={BK}"
        try:
            out = _matrix_mul_tiled(x1, x2, BM=BM, BN=BN, BK=BK)
            torch.mps.synchronize()
        except (ValueError, RuntimeError, tilec.CompileError) as e:
            print(f"  {label:<28} skipped: {str(e).splitlines()[0]}")
            continue
        err = (out - want).abs().max().item()
        del out
        g, ms = gflops(lambda: _matrix_mul_tiled(x1, x2, BM=BM, BN=BN, BK=BK), M, N, K)
        print(f"  {label:<28} {ms:9.2f} ms  {g:7.1f} GFLOPS  max error {err:.1e}")

    del x1, x2, want
    torch.mps.empty_cache()

for key, ck in _matmul_tiled.cache.items():
    tiles = "_".join(f"{k}{v}" for k, v in key[1])
    Path(__file__).with_name(f"{ck.name}_{tiles}.metal").write_text(ck.msl)
