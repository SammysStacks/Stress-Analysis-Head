
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        full_default = torch.ops.aten.full.default([12, 6, 256], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1 = torch.ops.aten.full.default([12, 32, 256], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2 = torch.ops.aten.full.default([12, 256], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        return (arg1_1, arg0_1, arg2_1, full_default, full_default_1, full_default_2)
        
def load_args(reader):
    buf0 = reader.storage(None, 138240, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf0, (12, 6, 120, 2), dtype=torch.float64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 864, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf1, (12, 6, 3), dtype=torch.int32, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 192, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf2, (12, 2), dtype=torch.float64, is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)