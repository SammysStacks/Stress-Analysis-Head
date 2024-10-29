
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16):
        index = torch.ops.aten.index.Tensor(primals_4, [primals_5, primals_6]);  primals_4 = None
        index_put = torch.ops.aten.index_put.default(primals_1, [primals_5, primals_7, primals_8], index);  index = None
        mul = torch.ops.aten.mul.Tensor(primals_3, 0.975)
        add = torch.ops.aten.add.Tensor(index_put, mul);  index_put = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_2, 3);  primals_2 = None
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
        iota = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1 = torch.ops.aten.mul.Tensor(iota, 1);  iota = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, 0);  mul_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type, 2);  convert_element_type = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, 3.141592653589793);  mul_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, primals_9)
        add_2 = torch.ops.aten.add.Tensor(mul_4, primals_10);  mul_4 = None
        sin = torch.ops.aten.sin.default(add_2);  add_2 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sin, 2);  sin = None
        mul_5 = torch.ops.aten.mul.Tensor(pow_1, 0.1);  pow_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_5, 1);  mul_5 = None
        sub = torch.ops.aten.sub.Tensor(add_3, 0.05);  add_3 = None
        div = torch.ops.aten.div.Tensor(view_3, sub);  view_3 = sub = None
        index_1 = torch.ops.aten.index.Tensor(primals_11, [primals_5, primals_6]);  primals_11 = None
        index_put_1 = torch.ops.aten.index_put.default(primals_1, [primals_5, primals_7, primals_8], index_1);  index_1 = None
        add_4 = torch.ops.aten.add.Tensor(index_put_1, mul);  index_put_1 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(div, 3);  div = None
        permute_5 = torch.ops.aten.permute.default(unsqueeze_2, [0, 1, 3, 2]);  unsqueeze_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(add_4, 3);  add_4 = None
        permute_6 = torch.ops.aten.permute.default(unsqueeze_3, [3, 0, 2, 1]);  unsqueeze_3 = None
        permute_7 = torch.ops.aten.permute.default(permute_5, [1, 0, 3, 2]);  permute_5 = None
        view_4 = torch.ops.aten.view.default(permute_7, [6, 12, 256]);  permute_7 = None
        permute_8 = torch.ops.aten.permute.default(permute_6, [1, 3, 2, 0]);  permute_6 = None
        view_5 = torch.ops.aten.view.default(permute_8, [6, 256, 256]);  permute_8 = None
        bmm_1 = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = None
        view_6 = torch.ops.aten.view.default(bmm_1, [6, 12, 1, 256]);  bmm_1 = None
        permute_9 = torch.ops.aten.permute.default(view_6, [1, 0, 3, 2]);  view_6 = None
        view_7 = torch.ops.aten.view.default(permute_9, [12, 6, 256]);  permute_9 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_3, primals_12);  primals_12 = None
        add_6 = torch.ops.aten.add.Tensor(mul_10, primals_13);  mul_10 = primals_13 = None
        sin_1 = torch.ops.aten.sin.default(add_6);  add_6 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(sin_1, 2);  sin_1 = None
        mul_11 = torch.ops.aten.mul.Tensor(pow_2, 0.1);  pow_2 = None
        add_7 = torch.ops.aten.add.Tensor(mul_11, 1);  mul_11 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_7, 0.05);  add_7 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_7, sub_1);  view_7 = None
        index_2 = torch.ops.aten.index.Tensor(primals_14, [primals_5, primals_6]);  primals_14 = primals_6 = None
        index_put_2 = torch.ops.aten.index_put.default(primals_1, [primals_5, primals_7, primals_8], index_2);  primals_5 = primals_7 = primals_8 = index_2 = None
        add_8 = torch.ops.aten.add.Tensor(index_put_2, mul);  index_put_2 = mul = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(mul_12, 3);  mul_12 = None
        permute_10 = torch.ops.aten.permute.default(unsqueeze_4, [0, 1, 3, 2]);  unsqueeze_4 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(add_8, 3);  add_8 = None
        permute_11 = torch.ops.aten.permute.default(unsqueeze_5, [3, 0, 2, 1]);  unsqueeze_5 = None
        permute_12 = torch.ops.aten.permute.default(permute_10, [1, 0, 3, 2]);  permute_10 = None
        view_8 = torch.ops.aten.view.default(permute_12, [6, 12, 256]);  permute_12 = None
        permute_13 = torch.ops.aten.permute.default(permute_11, [1, 3, 2, 0]);  permute_11 = None
        view_9 = torch.ops.aten.view.default(permute_13, [6, 256, 256]);  permute_13 = None
        bmm_2 = torch.ops.aten.bmm.default(view_8, view_9);  view_8 = None
        view_10 = torch.ops.aten.view.default(bmm_2, [6, 12, 1, 256]);  bmm_2 = None
        permute_14 = torch.ops.aten.permute.default(view_10, [1, 0, 3, 2]);  view_10 = None
        view_11 = torch.ops.aten.view.default(permute_14, [12, 6, 256]);  permute_14 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_3, primals_15);  mul_3 = primals_15 = None
        add_10 = torch.ops.aten.add.Tensor(mul_17, primals_16);  mul_17 = primals_16 = None
        sin_2 = torch.ops.aten.sin.default(add_10);  add_10 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(sin_2, 2);  sin_2 = None
        mul_18 = torch.ops.aten.mul.Tensor(pow_3, 0.1);  pow_3 = None
        add_11 = torch.ops.aten.add.Tensor(mul_18, 1);  mul_18 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_11, 0.05);  add_11 = None
        div_1 = torch.ops.aten.div.Tensor(view_11, sub_2);  view_11 = None
        permute_16 = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_20 = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
        permute_24 = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return (div_1, primals_3, primals_1, primals_9, primals_10, sub_1, sub_2, permute_16, permute_20, permute_24)
        
def load_args(reader):
    buf0 = reader.storage(None, 3145728, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf0, (6, 256, 256), dtype=torch.float64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 147456, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf1, (12, 6, 256), dtype=torch.float64, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf2, (256, 256), dtype=torch.float64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 144, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf3, (6, 3), dtype=torch.float64, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 110304, dtype_hint=torch.int64)
    reader.tensor(buf4, (4596,), dtype=torch.int64, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 36768, dtype_hint=torch.int64)
    reader.tensor(buf5, (4596,), dtype=torch.int64, is_leaf=True)  # primals_6
    reader.tensor(buf4, (4596,), dtype=torch.int64, storage_offset=4596, is_leaf=True)  # primals_7
    reader.tensor(buf4, (4596,), dtype=torch.int64, storage_offset=9192, is_leaf=True)  # primals_8
    buf6 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf6, (), dtype=torch.float64, is_leaf=True)  # primals_9
    buf7 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf7, (), dtype=torch.float64, is_leaf=True)  # primals_10
    buf8 = reader.storage(None, 144, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf8, (6, 3), dtype=torch.float64, is_leaf=True)  # primals_11
    buf9 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf9, (), dtype=torch.float64, is_leaf=True)  # primals_12
    buf10 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf10, (), dtype=torch.float64, is_leaf=True)  # primals_13
    buf11 = reader.storage(None, 144, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf11, (6, 3), dtype=torch.float64, is_leaf=True)  # primals_14
    buf12 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf12, (), dtype=torch.float64, is_leaf=True)  # primals_15
    buf13 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf13, (), dtype=torch.float64, is_leaf=True)  # primals_16
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)