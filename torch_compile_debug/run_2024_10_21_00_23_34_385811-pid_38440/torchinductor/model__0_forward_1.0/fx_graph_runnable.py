
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10):
        device_put = torch.ops.prims.device_put.default(primals_5, device(type='cuda', index=0));  primals_5 = None
        device_put_1 = torch.ops.prims.device_put.default(primals_7, device(type='cuda', index=0));  primals_7 = None
        iota = torch.ops.prims.iota.default(129, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul = torch.ops.aten.mul.Tensor(iota, 1);  iota = None
        add = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_2, 2);  convert_element_type_2 = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, 3.141592653589793);  mul_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_9);  mul_2 = None
        add_1 = torch.ops.aten.add.Tensor(mul_3, primals_10);  mul_3 = None
        sin = torch.ops.aten.sin.default(add_1);  add_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(sin, 1000000.0);  sin = None
        round_1 = torch.ops.aten.round.default(mul_4);  mul_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(round_1, 1e-06);  round_1 = None
        mul_6 = torch.ops.aten.mul.Tensor(primals_8, mul_5);  primals_8 = mul_5 = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, 1)
        sub = torch.ops.aten.sub.Tensor(primals_6, mul_7);  primals_6 = None
        mul_8 = torch.ops.aten.mul.Tensor(device_put, primals_1);  primals_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_8, device_put_1);  mul_8 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(sub, 3);  sub = None
        permute = torch.ops.aten.permute.default(unsqueeze, [0, 1, 3, 2]);  unsqueeze = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(add_2, 3);  add_2 = None
        permute_1 = torch.ops.aten.permute.default(unsqueeze_1, [3, 0, 2, 1]);  unsqueeze_1 = None
        permute_2 = torch.ops.aten.permute.default(permute, [1, 0, 3, 2]);  permute = None
        view = torch.ops.aten.view.default(permute_2, [110, 26, 129]);  permute_2 = None
        permute_3 = torch.ops.aten.permute.default(permute_1, [1, 3, 2, 0]);  permute_1 = None
        view_1 = torch.ops.aten.view.default(permute_3, [110, 129, 129]);  permute_3 = None
        bmm = torch.ops.aten.bmm.default(view, view_1)
        view_2 = torch.ops.aten.view.default(bmm, [110, 26, 1, 129]);  bmm = None
        permute_4 = torch.ops.aten.permute.default(view_2, [1, 0, 3, 2]);  view_2 = None
        view_3 = torch.ops.aten.view.default(permute_4, [26, 110, 129]);  permute_4 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_6, -1);  mul_6 = None
        sub_1 = torch.ops.aten.sub.Tensor(view_3, mul_16);  view_3 = None
        mul_17 = torch.ops.aten.mul.Tensor(device_put, primals_2);  primals_2 = None
        add_5 = torch.ops.aten.add.Tensor(mul_17, device_put_1);  mul_17 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(sub_1, 3);  sub_1 = None
        permute_5 = torch.ops.aten.permute.default(unsqueeze_2, [0, 1, 3, 2]);  unsqueeze_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(add_5, 3);  add_5 = None
        permute_6 = torch.ops.aten.permute.default(unsqueeze_3, [3, 0, 2, 1]);  unsqueeze_3 = None
        permute_7 = torch.ops.aten.permute.default(permute_5, [1, 0, 3, 2]);  permute_5 = None
        view_4 = torch.ops.aten.view.default(permute_7, [110, 26, 129]);  permute_7 = None
        permute_8 = torch.ops.aten.permute.default(permute_6, [1, 3, 2, 0]);  permute_6 = None
        view_5 = torch.ops.aten.view.default(permute_8, [110, 129, 129]);  permute_8 = None
        bmm_1 = torch.ops.aten.bmm.default(view_4, view_5)
        view_6 = torch.ops.aten.view.default(bmm_1, [110, 26, 1, 129]);  bmm_1 = None
        permute_9 = torch.ops.aten.permute.default(view_6, [1, 0, 3, 2]);  view_6 = None
        view_7 = torch.ops.aten.view.default(permute_9, [26, 110, 129]);  permute_9 = None
        sub_2 = torch.ops.aten.sub.Tensor(view_7, mul_7);  view_7 = mul_7 = None
        mul_26 = torch.ops.aten.mul.Tensor(device_put, primals_3);  primals_3 = None
        add_8 = torch.ops.aten.add.Tensor(mul_26, device_put_1);  mul_26 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(sub_2, 3);  sub_2 = None
        permute_10 = torch.ops.aten.permute.default(unsqueeze_4, [0, 1, 3, 2]);  unsqueeze_4 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(add_8, 3);  add_8 = None
        permute_11 = torch.ops.aten.permute.default(unsqueeze_5, [3, 0, 2, 1]);  unsqueeze_5 = None
        permute_12 = torch.ops.aten.permute.default(permute_10, [1, 0, 3, 2]);  permute_10 = None
        view_8 = torch.ops.aten.view.default(permute_12, [110, 26, 129]);  permute_12 = None
        permute_13 = torch.ops.aten.permute.default(permute_11, [1, 3, 2, 0]);  permute_11 = None
        view_9 = torch.ops.aten.view.default(permute_13, [110, 129, 129]);  permute_13 = None
        bmm_2 = torch.ops.aten.bmm.default(view_8, view_9)
        view_10 = torch.ops.aten.view.default(bmm_2, [110, 26, 1, 129]);  bmm_2 = None
        permute_14 = torch.ops.aten.permute.default(view_10, [1, 0, 3, 2]);  view_10 = None
        view_11 = torch.ops.aten.view.default(permute_14, [26, 110, 129]);  permute_14 = None
        sub_3 = torch.ops.aten.sub.Tensor(view_11, mul_16);  view_11 = mul_16 = None
        mul_35 = torch.ops.aten.mul.Tensor(device_put, primals_4);  primals_4 = None
        add_11 = torch.ops.aten.add.Tensor(mul_35, device_put_1);  mul_35 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(sub_3, 3);  sub_3 = None
        permute_15 = torch.ops.aten.permute.default(unsqueeze_6, [0, 1, 3, 2]);  unsqueeze_6 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(add_11, 3);  add_11 = None
        permute_16 = torch.ops.aten.permute.default(unsqueeze_7, [3, 0, 2, 1]);  unsqueeze_7 = None
        permute_17 = torch.ops.aten.permute.default(permute_15, [1, 0, 3, 2]);  permute_15 = None
        view_12 = torch.ops.aten.view.default(permute_17, [110, 26, 129]);  permute_17 = None
        permute_18 = torch.ops.aten.permute.default(permute_16, [1, 3, 2, 0]);  permute_16 = None
        view_13 = torch.ops.aten.view.default(permute_18, [110, 129, 129]);  permute_18 = None
        bmm_3 = torch.ops.aten.bmm.default(view_12, view_13)
        view_14 = torch.ops.aten.view.default(bmm_3, [110, 26, 1, 129]);  bmm_3 = None
        permute_19 = torch.ops.aten.permute.default(view_14, [1, 0, 3, 2]);  view_14 = None
        view_15 = torch.ops.aten.view.default(permute_19, [26, 110, 129]);  permute_19 = None
        permute_21 = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_22 = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_28 = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
        permute_29 = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_35 = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
        permute_36 = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
        permute_42 = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        permute_43 = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return [view_15, device_put, device_put_1, primals_9, primals_10, device_put, permute_21, permute_22, permute_28, permute_29, permute_35, permute_36, permute_42, permute_43]
        
def load_args(reader):
    buf0 = reader.storage(None, 14644080, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf0, (110, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 14644080, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf1, (110, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 14644080, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf2, (110, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 14644080, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf3, (110, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 133128, dtype_hint=torch.float64)
    reader.tensor(buf4, (1, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 2951520, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf5, (26, 110, 129), dtype=torch.float64, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 133128, dtype_hint=torch.float64)
    reader.tensor(buf6, (129, 129), dtype=torch.float64, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf7, (), dtype=torch.float64, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf8, (), dtype=torch.float64, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf9, (), dtype=torch.float64, is_leaf=True)  # primals_10
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)