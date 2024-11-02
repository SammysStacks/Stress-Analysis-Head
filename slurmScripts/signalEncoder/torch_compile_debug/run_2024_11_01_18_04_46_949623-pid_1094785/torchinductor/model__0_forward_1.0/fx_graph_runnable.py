
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



# torch version: 2.5.1+cu124
# torch cuda version: 12.4
# torch git version: a8d6afb511a69687bbb2b7e88a3cf67917e1697e


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# Tesla P100-PCIE-16GB : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4):
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_2, torch.int32);  primals_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(primals_3, torch.int32);  primals_3 = None
        full_default = torch.ops.aten.full.default([11, 6, 128], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1 = torch.ops.aten.full.default([11, 32, 128], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2 = torch.ops.aten.full.default([11, 128], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select = torch.ops.aten.select.int(convert_element_type, 2, 0)
        iota = torch.ops.prims.iota.default(120, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        expand = torch.ops.aten.expand.default(iota, [11, 64, 120]);  iota = None
        unsqueeze = torch.ops.aten.unsqueeze.default(select, -1);  select = None
        lt = torch.ops.aten.lt.Tensor(expand, unsqueeze);  expand = unsqueeze = None
        select_1 = torch.ops.aten.select.int(convert_element_type, 2, 2);  convert_element_type = None
        select_2 = torch.ops.aten.select.int(select_1, 1, 0);  select_1 = None
        index = torch.ops.aten.index.Tensor(primals_4, [select_2]);  primals_4 = None
        return (primals_1, convert_element_type_1, full_default, full_default_1, full_default_2, lt, index, select_2)
        
def load_args(reader):
    buf0 = reader.storage(None, 1351680, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf0, (11, 64, 120, 2), dtype=torch.float64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 704000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (11, 64, 3), (16000, 250, 2), storage_offset=240, is_leaf=True)  # primals_2
    reader.tensor(buf1, (11, 2), (16000, 2), storage_offset=246, is_leaf=True)  # primals_3
    buf2 = reader.storage(None, 61440, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf2, (60, 128), dtype=torch.float64, is_leaf=True)  # primals_4
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)