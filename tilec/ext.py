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
        x = tl.load(a + rm[:, None] * K + ka[None, :], mask=(rm[:, None] < M) & (ka[None, :] < K), other=0.0)
        y = tl.load(b + ka[:, None] * N + rn[None, :], mask=(ka[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = acc + tl.dot(x, y)
    tl.store(c + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

# return the maximum possible memory that can be help by a
# unit of execution (threadgroup) aka BLOCK
# and the number of such units of execution aka CHUNKS
def plan(N, block=None):
    if block is None:
        block = 1
        while block < N and block < MAX_BLOCK:
            block *= 4
    return block, cdiv(N, block)


SPLITS = (1, 2, 4, 8, 16, 32, 64)
JOINS = (1, 2, 4, 8)
BLOCKS = (256, 1024, 4096)

# we iterate through every possible factor by which the
# number of rows can be divided or combined.
def _blocks(n):
    top = plan(n)[0]
    return [b for b in BLOCKS if b < top and cdiv(n, b) <= 16] + [top]


def mm_configs(M1, N, M2):
    return [(p, r, b) for p in SPLITS for r in JOINS
            if (p == 1 or N // p >= 32) and (r == 1 or M1 >= r)
            for b in _blocks(cdiv(N, p))]


def _matrix_mul_tiled(x1, x2, BM=64, BN=64, BK=16):
    M, K = x1.shape
    K2, N = x2.shape
    if K != K2:
        raise ValueError(f"matrix_mul: shapes {tuple(x1.shape)} and {tuple(x2.shape)} don't match")
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    out = x1.new_empty(M, N)
    _matmul_tiled[(cdiv(M, BM), cdiv(N, BN))](x1, x2, out, M, N, K, BM=BM, BN=BN, BK=BK)
    return out


def matrix_mul(x1, x2):
    return _matrix_mul_tiled(x1, x2)
