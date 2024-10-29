
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20):
        constant_pad_nd = torch.ops.aten.constant_pad_nd.default(primals_1, [0, 0], 0.0);  primals_1 = None
        _fft_r2c = torch.ops.aten._fft_r2c.default(constant_pad_nd, [2], 1, True);  constant_pad_nd = None
        view_as_real = torch.ops.aten.view_as_real.default(_fft_r2c);  _fft_r2c = None
        select = torch.ops.aten.select.int(view_as_real, 3, 1)
        select_1 = torch.ops.aten.select.int(view_as_real, 3, 0);  view_as_real = None
        add = torch.ops.aten.add.Tensor(select_1, 0);  select_1 = None
        add_1 = torch.ops.aten.add.Tensor(select, 0);  select = None
        unsqueeze = torch.ops.aten.unsqueeze.default(add_1, 3)
        permute = torch.ops.aten.permute.default(unsqueeze, [0, 1, 3, 2]);  unsqueeze = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(primals_4, 3);  primals_4 = None
        permute_1 = torch.ops.aten.permute.default(unsqueeze_1, [3, 0, 2, 1]);  unsqueeze_1 = None
        permute_2 = torch.ops.aten.permute.default(permute, [1, 0, 3, 2]);  permute = None
        view = torch.ops.aten.view.default(permute_2, [6, 12, 129]);  permute_2 = None
        permute_3 = torch.ops.aten.permute.default(permute_1, [1, 3, 2, 0]);  permute_1 = None
        view_1 = torch.ops.aten.view.default(permute_3, [6, 129, 129]);  permute_3 = None
        bmm = torch.ops.aten.bmm.default(view, view_1);  view = None
        view_2 = torch.ops.aten.view.default(bmm, [6, 12, 1, 129]);  bmm = None
        permute_4 = torch.ops.aten.permute.default(view_2, [1, 0, 3, 2]);  view_2 = None
        view_3 = torch.ops.aten.view.default(permute_4, [12, 6, 129]);  permute_4 = None
        sub = torch.ops.aten.sub.Tensor(add, view_3);  add = view_3 = None
        div = torch.ops.aten.div.Tensor(sub, 1.15);  sub = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(div, 3)
        permute_5 = torch.ops.aten.permute.default(unsqueeze_2, [0, 1, 3, 2]);  unsqueeze_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(primals_3, 3);  primals_3 = None
        permute_6 = torch.ops.aten.permute.default(unsqueeze_3, [3, 0, 2, 1]);  unsqueeze_3 = None
        permute_7 = torch.ops.aten.permute.default(permute_5, [1, 0, 3, 2]);  permute_5 = None
        view_4 = torch.ops.aten.view.default(permute_7, [6, 12, 129]);  permute_7 = None
        permute_8 = torch.ops.aten.permute.default(permute_6, [1, 3, 2, 0]);  permute_6 = None
        view_5 = torch.ops.aten.view.default(permute_8, [6, 129, 129]);  permute_8 = None
        bmm_1 = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = None
        view_6 = torch.ops.aten.view.default(bmm_1, [6, 12, 1, 129]);  bmm_1 = None
        permute_9 = torch.ops.aten.permute.default(view_6, [1, 0, 3, 2]);  view_6 = None
        view_7 = torch.ops.aten.view.default(permute_9, [12, 6, 129]);  permute_9 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_1, view_7);  add_1 = view_7 = None
        div_1 = torch.ops.aten.div.Tensor(sub_1, 1.15);  sub_1 = None
        iota = torch.ops.prims.iota.default(129, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul = torch.ops.aten.mul.Tensor(iota, 1);  iota = None
        add_2 = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        convert_element_type = torch.ops.prims.convert_element_type.default(add_2, torch.float32);  add_2 = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type, 2);  convert_element_type = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, 3.141592653589793);  mul_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_5)
        add_3 = torch.ops.aten.add.Tensor(mul_3, primals_6);  mul_3 = None
        sin = torch.ops.aten.sin.default(add_3);  add_3 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sin, 2);  sin = None
        mul_4 = torch.ops.aten.mul.Tensor(pow_1, 0.1);  pow_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_4, 1);  mul_4 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_4, 0.05);  add_4 = None
        div_2 = torch.ops.aten.div.Tensor(div, sub_2);  div = sub_2 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_2, primals_7)
        add_6 = torch.ops.aten.add.Tensor(mul_8, primals_8);  mul_8 = None
        sin_1 = torch.ops.aten.sin.default(add_6);  add_6 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(sin_1, 2);  sin_1 = None
        mul_9 = torch.ops.aten.mul.Tensor(pow_2, 0.1);  pow_2 = None
        add_7 = torch.ops.aten.add.Tensor(mul_9, 1);  mul_9 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_7, 0.05);  add_7 = None
        div_3 = torch.ops.aten.div.Tensor(div_1, sub_3);  div_1 = sub_3 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(div_2, 3)
        permute_10 = torch.ops.aten.permute.default(unsqueeze_4, [0, 1, 3, 2]);  unsqueeze_4 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(primals_10, 3);  primals_10 = None
        permute_11 = torch.ops.aten.permute.default(unsqueeze_5, [3, 0, 2, 1]);  unsqueeze_5 = None
        permute_12 = torch.ops.aten.permute.default(permute_10, [1, 0, 3, 2]);  permute_10 = None
        view_8 = torch.ops.aten.view.default(permute_12, [6, 12, 129]);  permute_12 = None
        permute_13 = torch.ops.aten.permute.default(permute_11, [1, 3, 2, 0]);  permute_11 = None
        view_9 = torch.ops.aten.view.default(permute_13, [6, 129, 129]);  permute_13 = None
        bmm_2 = torch.ops.aten.bmm.default(view_8, view_9);  view_8 = None
        view_10 = torch.ops.aten.view.default(bmm_2, [6, 12, 1, 129]);  bmm_2 = None
        permute_14 = torch.ops.aten.permute.default(view_10, [1, 0, 3, 2]);  view_10 = None
        view_11 = torch.ops.aten.view.default(permute_14, [12, 6, 129]);  permute_14 = None
        sub_4 = torch.ops.aten.sub.Tensor(div_3, view_11);  div_3 = view_11 = None
        div_4 = torch.ops.aten.div.Tensor(sub_4, 1.15);  sub_4 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(div_4, 3)
        permute_15 = torch.ops.aten.permute.default(unsqueeze_6, [0, 1, 3, 2]);  unsqueeze_6 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(primals_9, 3);  primals_9 = None
        permute_16 = torch.ops.aten.permute.default(unsqueeze_7, [3, 0, 2, 1]);  unsqueeze_7 = None
        permute_17 = torch.ops.aten.permute.default(permute_15, [1, 0, 3, 2]);  permute_15 = None
        view_12 = torch.ops.aten.view.default(permute_17, [6, 12, 129]);  permute_17 = None
        permute_18 = torch.ops.aten.permute.default(permute_16, [1, 3, 2, 0]);  permute_16 = None
        view_13 = torch.ops.aten.view.default(permute_18, [6, 129, 129]);  permute_18 = None
        bmm_3 = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = None
        view_14 = torch.ops.aten.view.default(bmm_3, [6, 12, 1, 129]);  bmm_3 = None
        permute_19 = torch.ops.aten.permute.default(view_14, [1, 0, 3, 2]);  view_14 = None
        view_15 = torch.ops.aten.view.default(permute_19, [12, 6, 129]);  permute_19 = None
        sub_5 = torch.ops.aten.sub.Tensor(div_2, view_15);  div_2 = view_15 = None
        div_5 = torch.ops.aten.div.Tensor(sub_5, 1.15);  sub_5 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_2, primals_11);  primals_11 = None
        add_9 = torch.ops.aten.add.Tensor(mul_13, primals_12);  mul_13 = primals_12 = None
        sin_2 = torch.ops.aten.sin.default(add_9);  add_9 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(sin_2, 2);  sin_2 = None
        mul_14 = torch.ops.aten.mul.Tensor(pow_3, 0.1);  pow_3 = None
        add_10 = torch.ops.aten.add.Tensor(mul_14, 1);  mul_14 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_10, 0.05);  add_10 = None
        mul_15 = torch.ops.aten.mul.Tensor(div_5, sub_6);  div_5 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_2, primals_13);  primals_13 = None
        add_12 = torch.ops.aten.add.Tensor(mul_19, primals_14);  mul_19 = primals_14 = None
        sin_3 = torch.ops.aten.sin.default(add_12);  add_12 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(sin_3, 2);  sin_3 = None
        mul_20 = torch.ops.aten.mul.Tensor(pow_4, 0.1);  pow_4 = None
        add_13 = torch.ops.aten.add.Tensor(mul_20, 1);  mul_20 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_13, 0.05);  add_13 = None
        mul_21 = torch.ops.aten.mul.Tensor(div_4, sub_7);  div_4 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(mul_21, 3)
        permute_20 = torch.ops.aten.permute.default(unsqueeze_8, [0, 1, 3, 2]);  unsqueeze_8 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(primals_16, 3);  primals_16 = None
        permute_21 = torch.ops.aten.permute.default(unsqueeze_9, [3, 0, 2, 1]);  unsqueeze_9 = None
        permute_22 = torch.ops.aten.permute.default(permute_20, [1, 0, 3, 2]);  permute_20 = None
        view_16 = torch.ops.aten.view.default(permute_22, [6, 12, 129]);  permute_22 = None
        permute_23 = torch.ops.aten.permute.default(permute_21, [1, 3, 2, 0]);  permute_21 = None
        view_17 = torch.ops.aten.view.default(permute_23, [6, 129, 129]);  permute_23 = None
        bmm_4 = torch.ops.aten.bmm.default(view_16, view_17);  view_16 = None
        view_18 = torch.ops.aten.view.default(bmm_4, [6, 12, 1, 129]);  bmm_4 = None
        permute_24 = torch.ops.aten.permute.default(view_18, [1, 0, 3, 2]);  view_18 = None
        view_19 = torch.ops.aten.view.default(permute_24, [12, 6, 129]);  permute_24 = None
        sub_8 = torch.ops.aten.sub.Tensor(mul_15, view_19);  mul_15 = view_19 = None
        div_6 = torch.ops.aten.div.Tensor(sub_8, 1.15);  sub_8 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(div_6, 3)
        permute_25 = torch.ops.aten.permute.default(unsqueeze_10, [0, 1, 3, 2]);  unsqueeze_10 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(primals_15, 3);  primals_15 = None
        permute_26 = torch.ops.aten.permute.default(unsqueeze_11, [3, 0, 2, 1]);  unsqueeze_11 = None
        permute_27 = torch.ops.aten.permute.default(permute_25, [1, 0, 3, 2]);  permute_25 = None
        view_20 = torch.ops.aten.view.default(permute_27, [6, 12, 129]);  permute_27 = None
        permute_28 = torch.ops.aten.permute.default(permute_26, [1, 3, 2, 0]);  permute_26 = None
        view_21 = torch.ops.aten.view.default(permute_28, [6, 129, 129]);  permute_28 = None
        bmm_5 = torch.ops.aten.bmm.default(view_20, view_21);  view_20 = None
        view_22 = torch.ops.aten.view.default(bmm_5, [6, 12, 1, 129]);  bmm_5 = None
        permute_29 = torch.ops.aten.permute.default(view_22, [1, 0, 3, 2]);  view_22 = None
        view_23 = torch.ops.aten.view.default(permute_29, [12, 6, 129]);  permute_29 = None
        sub_9 = torch.ops.aten.sub.Tensor(mul_21, view_23);  mul_21 = view_23 = None
        div_7 = torch.ops.aten.div.Tensor(sub_9, 1.15);  sub_9 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_2, primals_17);  primals_17 = None
        add_15 = torch.ops.aten.add.Tensor(mul_25, primals_18);  mul_25 = primals_18 = None
        sin_4 = torch.ops.aten.sin.default(add_15);  add_15 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(sin_4, 2);  sin_4 = None
        mul_26 = torch.ops.aten.mul.Tensor(pow_5, 0.1);  pow_5 = None
        add_16 = torch.ops.aten.add.Tensor(mul_26, 1);  mul_26 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_16, 0.05);  add_16 = None
        div_8 = torch.ops.aten.div.Tensor(div_6, sub_10);  div_6 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_2, primals_19);  mul_2 = primals_19 = None
        add_18 = torch.ops.aten.add.Tensor(mul_30, primals_20);  mul_30 = primals_20 = None
        sin_5 = torch.ops.aten.sin.default(add_18);  add_18 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(sin_5, 2);  sin_5 = None
        mul_31 = torch.ops.aten.mul.Tensor(pow_6, 0.1);  pow_6 = None
        add_19 = torch.ops.aten.add.Tensor(mul_31, 1);  mul_31 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_19, 0.05);  add_19 = None
        div_9 = torch.ops.aten.div.Tensor(div_7, sub_11);  div_7 = None
        mul_32 = torch.ops.aten.mul.Tensor(div_9, 1j);  div_9 = None
        add_20 = torch.ops.aten.add.Tensor(div_8, mul_32);  div_8 = mul_32 = None
        _fft_c2r = torch.ops.aten._fft_c2r.default(add_20, [2], 1, 256);  add_20 = None
        permute_31 = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
        permute_35 = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
        permute_39 = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_43 = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_47 = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
        permute_51 = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return (_fft_c2r, primals_2, primals_5, primals_6, primals_7, primals_8, sub_6, sub_7, sub_10, sub_11, permute_31, permute_35, permute_39, permute_43, permute_47, permute_51)
        
def load_args(reader):
    buf0 = reader.storage(None, 147456, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf0, (12, 6, 256), (256, 3072, 1), dtype=torch.float64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 133128, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf1, (1, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 798768, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf2, (6, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 798768, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf3, (6, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf4, (), dtype=torch.float64, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf5, (), dtype=torch.float64, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf6, (), dtype=torch.float64, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf7, (), dtype=torch.float64, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 798768, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf8, (6, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 798768, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf9, (6, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf10, (), dtype=torch.float64, is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf11, (), dtype=torch.float64, is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf12, (), dtype=torch.float64, is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf13, (), dtype=torch.float64, is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 798768, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf14, (6, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 798768, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf15, (6, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf16, (), dtype=torch.float64, is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf17, (), dtype=torch.float64, is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf18, (), dtype=torch.float64, is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf19, (), dtype=torch.float64, is_leaf=True)  # primals_20
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)