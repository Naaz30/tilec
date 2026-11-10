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
    for i in tl.static_range(ROWS):
        row = ROW0 + pid * ROWS + i
        acc = 0.0
        for c in tl.static_range(CHUNKS):
            cols = c * BLOCK + tl.arange(0, BLOCK)
            v = tl.load(x_ptr + row * N + cols, mask=cols < N, other=0.0)
            acc = acc + tl.sum(v)
        tl.store(out_ptr + row, acc)

@jit
def _matrix_mul(x1, x2, out_ptr, N, M2, ROW0, COL0, LEN, OUT0,
                BLOCK: tl.constexpr, CHUNKS: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    for i in tl.static_range(ROWS):
        row = ROW0 + pid * ROWS + i
        for r2 in range(M2):
            acc = 0.0
            for c in tl.static_range(CHUNKS):
               cols = c * BLOCK + tl.arange(0, BLOCK)
               v1 = tl.load(x1 + row * N + COL0 + cols, mask=cols < LEN, other=0.0)
               v2 = tl.load(x2 + r2 * N + COL0 + cols, mask=cols < LEN, other=0.0)
               acc = acc + tl.sum(tl.mul(v1,v2))
               tl.barrier()
            tl.store(out_ptr + OUT0 + row * M2 + r2, acc)

# return the maximum possible memory that can be help by a
# unit of execution (threadgroup) aka BLOCK
# and the number of such units of execution aka CHUNKS
def plan(N, block=None):
    if block is None:
        block = 1 << (N - 1).bit_length()
        block = block if block < MAX_BLOCK else MAX_BLOCK
    return block, cdiv(N, block)


SPLITS = (1, 2, 4, 8, 16, 32, 64)
JOINS = (1, 2, 4, 8)
BLOCKS = (128, 256, 512, 1024, 2048, 4096)

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


# sum a 2-D float32 MPS tensor along axis 0 or 1
def sum(x, axis, config=(1, 1, None)):
    if axis == 0:
        x = x.T
    x = x.contiguous()
    return _row_sum(x, *config)


def matrix_mul(x1,x2, config=None):
    M1, N = x1.shape
    N, M2 = x2.shape
    if config is None:
        from .tune import pick
        config = pick(M1, N, M2)
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
