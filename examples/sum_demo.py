"""Sum a 2-D tensor along an axis on the Apple GPU"""

import torch

import tilec

# x = torch.arange(6, dtype=torch.float32, device="mps").reshape(2, 3) 
# print("input:\n", x.cpu().numpy())
# out1 = tilec.ext.sum(x, axis=1)
# out0 = tilec.ext.sum(x, axis=0)
# torch.mps.synchronize()                      
# print("sum axis=1:", out1.cpu().numpy())       
# print("sum axis=0:", out0.cpu().numpy())


x1 = torch.arange(6, dtype=torch.float32, device="mps").reshape(2, 3)
x2 = torch.arange(12, dtype=torch.float32, device="mps").reshape(3, 4)

out = tilec.ext.matrix_mul(x1,x2)
print(out)



# ck = tilec.ext._row_sum_kernel.inspect(x, out1, 3, BLOCK=4, CHUNKS=1)
# print(ck.ir)
# print(ck.msl)
