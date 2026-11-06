from . import language as tl
from .jit import jit
from .language import cdiv

# one threadgroup on one GPU code sharing 
# that core's shared memory

# largest tile size that can be help by a thread group
# 128 threads x 32 registers each
MAX_BLOCK = 4096 


@jit
def _row_sum_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr, CHUNKS: tl.constexpr):
    # we are assigning one row per threadgroup
    # each row is then divided into CHUNKS of size BLOCK that 
    # that threadgroup processes one after the other
    row = tl.program_id(0)                      
    acc = 0.0
    for c in tl.static_range(CHUNKS):           
        cols = c * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + row * N + cols, mask=cols < N, other=0.0)
        acc = acc + tl.sum(v)
    tl.store(out_ptr + row, acc)

@jit
def _matrix_mul(x1, x2, out_ptr, N, M2, BLOCK: tl.constexpr, CHUNKS: tl.constexpr):
    row = tl.program_id(0)
    for r2 in range(M2):                          
        acc = 0.0                             
        for c in tl.static_range(CHUNKS):       
           cols = c * BLOCK + tl.arange(0, BLOCK)
           v1 = tl.load(x1 + row * N + cols, mask=cols < N, other=0.0)
           v2 = tl.load(x2 + r2 * N + cols, mask=cols < N, other=0.0)
           acc = acc + tl.sum(tl.mul(v1,v2))
           tl.barrier()        
        tl.store(out_ptr + row * M2 + r2, acc) 

# return the maximum possible memory that can be help by a 
# unit of execution (threadgroup) aka BLOCK 
# and the number of such units of execution aka CHUNKS
def plan(N):
    block = 1 << (N - 1).bit_length() 
    block = block if block < MAX_BLOCK else MAX_BLOCK
    return block, cdiv(N, block)

# sum a 2-D float32 MPS tensor along axis 0 or 1
def sum(x, axis):
    if axis == 0:
        x = x.T                                  
    x = x.contiguous()                           
    M, N = x.shape

    # we pass the column count bc elems need to be 
    # distributed in row major fashion between threads of a threadgroup (opt 1)
    BLOCK, CHUNKS = plan(N)
    out = x.new_empty(M)                        
    _row_sum_kernel[(M,)](x, out, N, BLOCK=BLOCK, CHUNKS=CHUNKS)
    return out


def matrix_mul(x1,x2):
    M1, N = x1.shape
    N, M2 = x2.shape
    #so both matrices are traversed row major
    x2 = x2.T
    x2 = x2.contiguous()
    BLOCK, CHUNKS = plan(N)
    out = x1.new_empty(M1,M2)
    _matrix_mul[(M1,)](x1, x2, out, N, M2, BLOCK=BLOCK, CHUNKS=CHUNKS)
    return out

