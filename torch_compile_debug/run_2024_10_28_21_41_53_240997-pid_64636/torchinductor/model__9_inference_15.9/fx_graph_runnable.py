
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

    
    
    def forward(self, arg0_1, arg1_1):
        iota = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul = torch.ops.aten.mul.Tensor(iota, 1);  iota = None
        add = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        convert_element_type = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type, 2);  convert_element_type = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, 3.141592653589793);  mul_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg0_1);  mul_2 = arg0_1 = None
        add_1 = torch.ops.aten.add.Tensor(mul_3, arg1_1);  mul_3 = arg1_1 = None
        sin = torch.ops.aten.sin.default(add_1);  add_1 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sin, 2);  sin = None
        mul_4 = torch.ops.aten.mul.Tensor(pow_1, 0.1);  pow_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_4, 1);  mul_4 = None
        sub = torch.ops.aten.sub.Tensor(add_2, 0.05);  add_2 = None
        return (sub,)
        
def load_args(reader):
    buf0 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf0, (), dtype=torch.float64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf1, (), dtype=torch.float64, is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)