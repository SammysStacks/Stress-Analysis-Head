
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

    
    
    def forward(self, primals_1):
        _fft_r2c = torch.ops.aten._fft_r2c.default(primals_1, [2], 1, True);  primals_1 = None
        view_as_real = torch.ops.aten.view_as_real.default(_fft_r2c)
        select = torch.ops.aten.select.int(view_as_real, 3, 1);  view_as_real = None
        view_as_real_1 = torch.ops.aten.view_as_real.default(_fft_r2c)
        select_1 = torch.ops.aten.select.int(view_as_real_1, 3, 0);  view_as_real_1 = None
        return (select_1, select, _fft_r2c)
        
def load_args(reader):
    buf0 = reader.storage(None, 147456, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf0, (12, 6, 256), (256, 3072, 1), dtype=torch.float64, is_leaf=True)  # primals_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)