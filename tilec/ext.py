from . import language as tl
from .jit import jit
from .language import cdiv

# one threadgroup on one GPU code sharing
# that core's shared memory

# largest tile size that can be help by a thread group
# 128 threads x 32 registers each
MAX_BLOCK = 4096


@jit
def _row_sum_kernel(x_ptr, out_ptr, N, ROW0, BLOCK: tl.constexpr, CHUNKS: tl.constexpr, ROWS: tl.constexpr):
    # we are assigning one row per threadgroup
    # each row is then divided into CHUNKS of size BLOCK that
    # that threadgroup processes one after the other
    pid = tl.program_id(0)
    cols0 = tl.arange(0, BLOCK)
    for i in tl.static_range(ROWS):
        row = ROW0 + pid * ROWS + i
        acc_t = tl.load(x_ptr + row * N + cols0, mask=cols0 < 0, other=0.0)
        for c in tl.static_range(CHUNKS):
            cols = c * BLOCK + cols0
            acc_t = acc_t + tl.load(x_ptr + row * N + cols, mask=cols < N, other=0.0)
        tl.store(out_ptr + row, tl.sum(acc_t))


@jit
def _col_sum_kernel(x_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc_t = tl.load(x_ptr + cols, mask=cols < 0, other=0.0)
    for i in range(M):
        acc_t = acc_t + tl.load(x_ptr + i * N + cols, mask=cols < N, other=0.0)
    tl.store(out_ptr + cols, acc_t)


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


def configs(M, N):
    return [(p, r, b) for p in SPLITS for r in JOINS
            if N % p == 0 and (p == 1 or N // p >= 32) and (r == 1 or M * p >= r)
            for b in _blocks(N // p)]


def mm_configs(M1, N, M2):
    return [(p, r, b) for p in SPLITS for r in JOINS
            if (p == 1 or N // p >= 32) and (r == 1 or M1 >= r)
            for b in _blocks(cdiv(N, p))]


def _row_sum(x, split, join, block=None):
    M, N = x.shape
    x = x.view(M * split, N // split)
    m, n = x.shape
    BLOCK, CHUNKS = plan(n, block)
    out = x.new_empty(m)
    full = m // join * join
    if full:
        _row_sum_kernel[(full // join,)](x, out, n, 0, BLOCK=BLOCK, CHUNKS=CHUNKS, ROWS=join)
    if full < m:
        _row_sum_kernel[(m - full,)](x, out, n, full, BLOCK=BLOCK, CHUNKS=CHUNKS, ROWS=1)
    if split > 1:
        out = _row_sum(out.view(M, split), 1, 1)
    return out


def _col_sum(x, block=128):
    M, N = x.shape
    grid = cdiv(N, block)
    out = x.new_empty(grid * block)
    _col_sum_kernel[(grid,)](x, out, M, N, BLOCK=block)
    return out[:N]


# sum a 2-D float32 MPS tensor along axis 0 or 1
def sum(x, axis, config=(1, 1, None)):
    x = x.contiguous()
    if axis == 0:
        return _col_sum(x)
    return _row_sum(x, *config)


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


def matrix_mul(x1,x2, config=None):
    if config is None:
        return _matrix_mul_tiled(x1, x2)
    M1, N = x1.shape
    N, M2 = x2.shape
    #so both matrices are traversed row major
    x1 = x1.contiguous()
    x2 = x2.T
    x2 = x2.contiguous()
    split, join, block = config
    L = cdiv(N, split)
    BLOCK, CHUNKS = plan(L, block)
    out = x1.new_empty(split, M1, M2)
    full = M1 // join * join
    for k in range(split):
        col0 = k * L
        length = min(L, N - col0)
        out0 = k * M1 * M2
        if full:
            _matrix_mul[(full // join,)](x1, x2, out, N, M2, 0, col0, length, out0,
                                         BLOCK=BLOCK, CHUNKS=CHUNKS, ROWS=join)
        if full < M1:
            _matrix_mul[(M1 - full,)](x1, x2, out, N, M2, full, col0, length, out0,
                                      BLOCK=BLOCK, CHUNKS=CHUNKS, ROWS=1)
    if split > 1:
        return sum(out.view(split, M1 * M2), 0, (1, 1, None)).view(M1, M2)
    return out.view(M1, M2)
