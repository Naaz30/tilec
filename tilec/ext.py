from . import language as tl
from .jit import jit
from .language import cdiv

# one threadgroup on one GPU core sharing
# that core's shared memory

# largest amount of data (input tiles + output tile) that can be held by a thread group
# 128 threads x 32 registers each
MAX_BLOCK = 4096


@jit
def _matmul_tiled(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN))
    for k in range(0, K, BK):
        ka = k + rk
        x = tl.load(a + rm[:, None] * K + ka[None, :])
        y = tl.load(b + ka[:, None] * N + rn[None, :])
        acc = acc + tl.dot(x, y)
    tl.store(c + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _matrix_mul_tiled(x1, x2, BM=64, BN=64, BK=16):
    M, K = x1.shape
    K2, N = x2.shape
    if K != K2:
        raise ValueError(f"matrix_mul: shapes {tuple(x1.shape)} and {tuple(x2.shape)} don't match")
    if M % BM or N % BN or K % BK:
        raise ValueError(f"matrix_mul: needs M % {BM} == 0, N % {BN} == 0 and K % {BK} == 0, "
                         f"got M={M}, N={N}, K={K}")
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    if x1.storage_offset() % 4:
        x1 = x1.clone()
    if x2.storage_offset() % 4:
        x2 = x2.clone()
    out = x1.new_empty(M, N)
    _matmul_tiled[(cdiv(M, BM), cdiv(N, BN))](x1, x2, out, M, N, K, BM=BM, BN=BN, BK=BK)
    return out


def matrix_mul(x1, x2):
    return _matrix_mul_tiled(x1, x2)
