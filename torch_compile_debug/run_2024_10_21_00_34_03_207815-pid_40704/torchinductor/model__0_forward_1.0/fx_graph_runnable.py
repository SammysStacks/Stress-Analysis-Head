
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


torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.4.1+cu124
# torch cuda version: 12.4
# torch git version: 38b96d3399a695e704ed39b60dac733c3fbf20e2


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA GeForce RTX 4090 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4):
        mul = torch.ops.aten.mul.Tensor(primals_2, primals_1);  primals_1 = None
        add = torch.ops.aten.add.Tensor(mul, primals_3);  mul = primals_3 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_4, 3);  primals_4 = None
        permute = torch.ops.aten.permute.default(unsqueeze, [0, 1, 3, 2]);  unsqueeze = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(add, 3);  add = None
        permute_1 = torch.ops.aten.permute.default(unsqueeze_1, [3, 0, 2, 1]);  unsqueeze_1 = None
        permute_2 = torch.ops.aten.permute.default(permute, [0, 3, 1, 2]);  permute = None
        view = torch.ops.aten.view.default(permute_2, [1, 3080, 129]);  permute_2 = None
        permute_3 = torch.ops.aten.permute.default(permute_1, [3, 1, 2, 0]);  permute_1 = None
        view_1 = torch.ops.aten.view.default(permute_3, [1, 129, 129]);  permute_3 = None
        bmm = torch.ops.aten.bmm.default(view, view_1)
        view_2 = torch.ops.aten.view.default(bmm, [3080, 1, 1, 129]);  bmm = None
        permute_4 = torch.ops.aten.permute.default(view_2, [0, 2, 3, 1]);  view_2 = None
        view_3 = torch.ops.aten.view.default(permute_4, [3080, 1, 129]);  permute_4 = None
        permute_6 = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        permute_7 = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return [view_3, primals_2, permute_6, permute_7]
        
def load_args(reader):
    buf0 = reader.storage(None, 133128, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf0, (1, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 133128, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf1, (1, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 133128, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf2, (129, 129), dtype=torch.float64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 3178560, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf3, (3080, 1, 129), dtype=torch.float64, is_leaf=True)  # primals_4
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)