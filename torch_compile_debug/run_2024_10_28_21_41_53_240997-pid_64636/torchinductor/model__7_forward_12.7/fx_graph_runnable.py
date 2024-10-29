
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.suppress_errors = True

torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.debug_partitioner = True



isolate_fails_code_str = None



# torch version: 2.6.0.dev20241027+cu124
# torch cuda version: 12.4
# torch git version: cef671f99bdf4a484f1cf7cb46b008af870582f5


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA GeForce RTX 4090 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8):
        index = torch.ops.aten.index.Tensor(primals_2, [primals_3, primals_4]);  primals_2 = primals_4 = None
        index_put = torch.ops.aten.index_put.default(primals_1, [primals_3, primals_5, primals_6], index);  primals_1 = primals_3 = primals_5 = primals_6 = index = None
        mul = torch.ops.aten.mul.Tensor(primals_7, 0.975);  primals_7 = None
        add = torch.ops.aten.add.Tensor(index_put, mul);  index_put = mul = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_8, 3);  primals_8 = None
        permute = torch.ops.aten.permute.default(unsqueeze, [0, 1, 3, 2]);  unsqueeze = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(add, 3);  add = None
        permute_1 = torch.ops.aten.permute.default(unsqueeze_1, [3, 0, 2, 1]);  unsqueeze_1 = None
        permute_2 = torch.ops.aten.permute.default(permute, [1, 0, 3, 2]);  permute = None
        view = torch.ops.aten.view.default(permute_2, [6, 12, 256]);  permute_2 = None
        permute_3 = torch.ops.aten.permute.default(permute_1, [1, 3, 2, 0]);  permute_1 = None
        view_1 = torch.ops.aten.view.default(permute_3, [6, 256, 256]);  permute_3 = None
        bmm = torch.ops.aten.bmm.default(view, view_1);  view = None
        view_2 = torch.ops.aten.view.default(bmm, [6, 12, 1, 256]);  bmm = None
        permute_4 = torch.ops.aten.permute.default(view_2, [1, 0, 3, 2]);  view_2 = None
        view_3 = torch.ops.aten.view.default(permute_4, [12, 6, 256]);  permute_4 = None
        permute_6 = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return (view_3, permute_6)
        
def load_args(reader):
    buf0 = reader.storage(None, 3145728, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf0, (6, 256, 256), dtype=torch.float64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 144, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf1, (6, 3), dtype=torch.float64, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 110304, dtype_hint=torch.int64)
    reader.tensor(buf2, (4596,), dtype=torch.int64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 36768, dtype_hint=torch.int64)
    reader.tensor(buf3, (4596,), dtype=torch.int64, is_leaf=True)  # primals_4
    reader.tensor(buf2, (4596,), dtype=torch.int64, storage_offset=4596, is_leaf=True)  # primals_5
    reader.tensor(buf2, (4596,), dtype=torch.int64, storage_offset=9192, is_leaf=True)  # primals_6
    buf4 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf4, (256, 256), dtype=torch.float64, is_leaf=True)  # primals_7
    buf5 = reader.storage(None, 147456, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf5, (12, 6, 256), dtype=torch.float64, is_leaf=True)  # primals_8
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)