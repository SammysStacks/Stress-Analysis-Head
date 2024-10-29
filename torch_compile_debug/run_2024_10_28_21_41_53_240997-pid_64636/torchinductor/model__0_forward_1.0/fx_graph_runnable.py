
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

torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4):
        full_default = torch.ops.aten.full.default([12, 6, 256], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1 = torch.ops.aten.full.default([12, 32, 256], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2 = torch.ops.aten.full.default([12, 256], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select = torch.ops.aten.select.int(primals_2, 2, 2);  primals_2 = None
        select_1 = torch.ops.aten.select.int(select, 1, 0);  select = None
        select_2 = torch.ops.aten.select.int(primals_1, 3, 1)
        select_3 = torch.ops.aten.select.int(primals_1, 3, 0);  primals_1 = None
        eq = torch.ops.aten.eq.Scalar(select_2, 0)
        eq_1 = torch.ops.aten.eq.Scalar(select_3, 0)
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(eq, eq_1);  eq = eq_1 = None
        logical_not = torch.ops.aten.logical_not.default(bitwise_and)
        any_1 = torch.ops.aten.any.dim(logical_not, -1);  logical_not = None
        logical_not_1 = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        view = torch.ops.aten.view.default(logical_not_1, [12, 6, 1]);  logical_not_1 = None
        bitwise_not = torch.ops.aten.bitwise_not.default(view);  view = None
        index = torch.ops.aten.index.Tensor(primals_4, [select_1])
        select_4 = torch.ops.aten.select.int(primals_4, 0, 0);  primals_4 = None
        slice_10 = torch.ops.aten.slice.Tensor(select_4, 0, 0, 5);  select_4 = None
        device_put = torch.ops.prims.device_put.default(slice_10, device(type='cpu'));  slice_10 = None
        return (device_put, primals_3, full_default, full_default_1, full_default_2, select_2, select_3, bitwise_and, bitwise_not, index, select_1)
        
def load_args(reader):
    buf0 = reader.storage(None, 138240, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf0, (12, 6, 120, 2), dtype=torch.float64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 864, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf1, (12, 6, 3), dtype=torch.int32, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 192, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf2, (12, 2), dtype=torch.float64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 122880, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf3, (60, 256), dtype=torch.float64, is_leaf=True)  # primals_4
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)