
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37):
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
        mul_17 = torch.ops.aten.mul.Tensor(mul_3, primals_15);  primals_15 = None
        add_10 = torch.ops.aten.add.Tensor(mul_17, primals_16);  mul_17 = primals_16 = None
        sin_2 = torch.ops.aten.sin.default(add_10);  add_10 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(sin_2, 2);  sin_2 = None
        mul_18 = torch.ops.aten.mul.Tensor(pow_3, 0.1);  pow_3 = None
        add_11 = torch.ops.aten.add.Tensor(mul_18, 1);  mul_18 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_11, 0.05);  add_11 = None
        div_1 = torch.ops.aten.div.Tensor(view_11, sub_2);  view_11 = None
        constant_pad_nd = torch.ops.aten.constant_pad_nd.default(div_1, [0, 0], 0.0);  div_1 = None
        _fft_r2c = torch.ops.aten._fft_r2c.default(constant_pad_nd, [2], 1, True);  constant_pad_nd = None
        view_as_real = torch.ops.aten.view_as_real.default(_fft_r2c);  _fft_r2c = None
        select = torch.ops.aten.select.int(view_as_real, 3, 1)
        select_1 = torch.ops.aten.select.int(view_as_real, 3, 0);  view_as_real = None
        add_12 = torch.ops.aten.add.Tensor(select_1, 0);  select_1 = None
        add_13 = torch.ops.aten.add.Tensor(select, 0);  select = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(add_13, 3)
        permute_15 = torch.ops.aten.permute.default(unsqueeze_6, [0, 1, 3, 2]);  unsqueeze_6 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(primals_19, 3);  primals_19 = None
        permute_16 = torch.ops.aten.permute.default(unsqueeze_7, [3, 0, 2, 1]);  unsqueeze_7 = None
        permute_17 = torch.ops.aten.permute.default(permute_15, [1, 0, 3, 2]);  permute_15 = None
        view_12 = torch.ops.aten.view.default(permute_17, [6, 12, 129]);  permute_17 = None
        permute_18 = torch.ops.aten.permute.default(permute_16, [1, 3, 2, 0]);  permute_16 = None
        view_13 = torch.ops.aten.view.default(permute_18, [6, 129, 129]);  permute_18 = None
        bmm_3 = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = None
        view_14 = torch.ops.aten.view.default(bmm_3, [6, 12, 1, 129]);  bmm_3 = None
        permute_19 = torch.ops.aten.permute.default(view_14, [1, 0, 3, 2]);  view_14 = None
        view_15 = torch.ops.aten.view.default(permute_19, [12, 6, 129]);  permute_19 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_12, view_15);  add_12 = view_15 = None
        div_2 = torch.ops.aten.div.Tensor(sub_3, 1.15);  sub_3 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(div_2, 3)
        permute_20 = torch.ops.aten.permute.default(unsqueeze_8, [0, 1, 3, 2]);  unsqueeze_8 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(primals_18, 3);  primals_18 = None
        permute_21 = torch.ops.aten.permute.default(unsqueeze_9, [3, 0, 2, 1]);  unsqueeze_9 = None
        permute_22 = torch.ops.aten.permute.default(permute_20, [1, 0, 3, 2]);  permute_20 = None
        view_16 = torch.ops.aten.view.default(permute_22, [6, 12, 129]);  permute_22 = None
        permute_23 = torch.ops.aten.permute.default(permute_21, [1, 3, 2, 0]);  permute_21 = None
        view_17 = torch.ops.aten.view.default(permute_23, [6, 129, 129]);  permute_23 = None
        bmm_4 = torch.ops.aten.bmm.default(view_16, view_17);  view_16 = None
        view_18 = torch.ops.aten.view.default(bmm_4, [6, 12, 1, 129]);  bmm_4 = None
        permute_24 = torch.ops.aten.permute.default(view_18, [1, 0, 3, 2]);  view_18 = None
        view_19 = torch.ops.aten.view.default(permute_24, [12, 6, 129]);  permute_24 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_13, view_19);  add_13 = view_19 = None
        div_3 = torch.ops.aten.div.Tensor(sub_4, 1.15);  sub_4 = None
        iota_3 = torch.ops.prims.iota.default(129, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_19 = torch.ops.aten.mul.Tensor(iota_3, 1);  iota_3 = None
        add_14 = torch.ops.aten.add.Tensor(mul_19, 0);  mul_19 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(add_14, torch.float32);  add_14 = None
        mul_20 = torch.ops.aten.mul.Tensor(convert_element_type_3, 2);  convert_element_type_3 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, 3.141592653589793);  mul_20 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, primals_20)
        add_15 = torch.ops.aten.add.Tensor(mul_22, primals_21);  mul_22 = None
        sin_3 = torch.ops.aten.sin.default(add_15);  add_15 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(sin_3, 2);  sin_3 = None
        mul_23 = torch.ops.aten.mul.Tensor(pow_4, 0.1);  pow_4 = None
        add_16 = torch.ops.aten.add.Tensor(mul_23, 1);  mul_23 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_16, 0.05);  add_16 = None
        div_4 = torch.ops.aten.div.Tensor(div_2, sub_5);  div_2 = sub_5 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_21, primals_22)
        add_18 = torch.ops.aten.add.Tensor(mul_27, primals_23);  mul_27 = None
        sin_4 = torch.ops.aten.sin.default(add_18);  add_18 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(sin_4, 2);  sin_4 = None
        mul_28 = torch.ops.aten.mul.Tensor(pow_5, 0.1);  pow_5 = None
        add_19 = torch.ops.aten.add.Tensor(mul_28, 1);  mul_28 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_19, 0.05);  add_19 = None
        div_5 = torch.ops.aten.div.Tensor(div_3, sub_6);  div_3 = sub_6 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(div_4, 3)
        permute_25 = torch.ops.aten.permute.default(unsqueeze_10, [0, 1, 3, 2]);  unsqueeze_10 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(primals_25, 3);  primals_25 = None
        permute_26 = torch.ops.aten.permute.default(unsqueeze_11, [3, 0, 2, 1]);  unsqueeze_11 = None
        permute_27 = torch.ops.aten.permute.default(permute_25, [1, 0, 3, 2]);  permute_25 = None
        view_20 = torch.ops.aten.view.default(permute_27, [6, 12, 129]);  permute_27 = None
        permute_28 = torch.ops.aten.permute.default(permute_26, [1, 3, 2, 0]);  permute_26 = None
        view_21 = torch.ops.aten.view.default(permute_28, [6, 129, 129]);  permute_28 = None
        bmm_5 = torch.ops.aten.bmm.default(view_20, view_21);  view_20 = None
        view_22 = torch.ops.aten.view.default(bmm_5, [6, 12, 1, 129]);  bmm_5 = None
        permute_29 = torch.ops.aten.permute.default(view_22, [1, 0, 3, 2]);  view_22 = None
        view_23 = torch.ops.aten.view.default(permute_29, [12, 6, 129]);  permute_29 = None
        sub_7 = torch.ops.aten.sub.Tensor(div_5, view_23);  div_5 = view_23 = None
        div_6 = torch.ops.aten.div.Tensor(sub_7, 1.15);  sub_7 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(div_6, 3)
        permute_30 = torch.ops.aten.permute.default(unsqueeze_12, [0, 1, 3, 2]);  unsqueeze_12 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(primals_24, 3);  primals_24 = None
        permute_31 = torch.ops.aten.permute.default(unsqueeze_13, [3, 0, 2, 1]);  unsqueeze_13 = None
        permute_32 = torch.ops.aten.permute.default(permute_30, [1, 0, 3, 2]);  permute_30 = None
        view_24 = torch.ops.aten.view.default(permute_32, [6, 12, 129]);  permute_32 = None
        permute_33 = torch.ops.aten.permute.default(permute_31, [1, 3, 2, 0]);  permute_31 = None
        view_25 = torch.ops.aten.view.default(permute_33, [6, 129, 129]);  permute_33 = None
        bmm_6 = torch.ops.aten.bmm.default(view_24, view_25);  view_24 = None
        view_26 = torch.ops.aten.view.default(bmm_6, [6, 12, 1, 129]);  bmm_6 = None
        permute_34 = torch.ops.aten.permute.default(view_26, [1, 0, 3, 2]);  view_26 = None
        view_27 = torch.ops.aten.view.default(permute_34, [12, 6, 129]);  permute_34 = None
        sub_8 = torch.ops.aten.sub.Tensor(div_4, view_27);  div_4 = view_27 = None
        div_7 = torch.ops.aten.div.Tensor(sub_8, 1.15);  sub_8 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_21, primals_26);  primals_26 = None
        add_21 = torch.ops.aten.add.Tensor(mul_32, primals_27);  mul_32 = primals_27 = None
        sin_5 = torch.ops.aten.sin.default(add_21);  add_21 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(sin_5, 2);  sin_5 = None
        mul_33 = torch.ops.aten.mul.Tensor(pow_6, 0.1);  pow_6 = None
        add_22 = torch.ops.aten.add.Tensor(mul_33, 1);  mul_33 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_22, 0.05);  add_22 = None
        mul_34 = torch.ops.aten.mul.Tensor(div_7, sub_9);  div_7 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_21, primals_28);  primals_28 = None
        add_24 = torch.ops.aten.add.Tensor(mul_38, primals_29);  mul_38 = primals_29 = None
        sin_6 = torch.ops.aten.sin.default(add_24);  add_24 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(sin_6, 2);  sin_6 = None
        mul_39 = torch.ops.aten.mul.Tensor(pow_7, 0.1);  pow_7 = None
        add_25 = torch.ops.aten.add.Tensor(mul_39, 1);  mul_39 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_25, 0.05);  add_25 = None
        mul_40 = torch.ops.aten.mul.Tensor(div_6, sub_10);  div_6 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(mul_40, 3)
        permute_35 = torch.ops.aten.permute.default(unsqueeze_14, [0, 1, 3, 2]);  unsqueeze_14 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(primals_31, 3);  primals_31 = None
        permute_36 = torch.ops.aten.permute.default(unsqueeze_15, [3, 0, 2, 1]);  unsqueeze_15 = None
        permute_37 = torch.ops.aten.permute.default(permute_35, [1, 0, 3, 2]);  permute_35 = None
        view_28 = torch.ops.aten.view.default(permute_37, [6, 12, 129]);  permute_37 = None
        permute_38 = torch.ops.aten.permute.default(permute_36, [1, 3, 2, 0]);  permute_36 = None
        view_29 = torch.ops.aten.view.default(permute_38, [6, 129, 129]);  permute_38 = None
        bmm_7 = torch.ops.aten.bmm.default(view_28, view_29);  view_28 = None
        view_30 = torch.ops.aten.view.default(bmm_7, [6, 12, 1, 129]);  bmm_7 = None
        permute_39 = torch.ops.aten.permute.default(view_30, [1, 0, 3, 2]);  view_30 = None
        view_31 = torch.ops.aten.view.default(permute_39, [12, 6, 129]);  permute_39 = None
        sub_11 = torch.ops.aten.sub.Tensor(mul_34, view_31);  mul_34 = view_31 = None
        div_8 = torch.ops.aten.div.Tensor(sub_11, 1.15);  sub_11 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(div_8, 3)
        permute_40 = torch.ops.aten.permute.default(unsqueeze_16, [0, 1, 3, 2]);  unsqueeze_16 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(primals_30, 3);  primals_30 = None
        permute_41 = torch.ops.aten.permute.default(unsqueeze_17, [3, 0, 2, 1]);  unsqueeze_17 = None
        permute_42 = torch.ops.aten.permute.default(permute_40, [1, 0, 3, 2]);  permute_40 = None
        view_32 = torch.ops.aten.view.default(permute_42, [6, 12, 129]);  permute_42 = None
        permute_43 = torch.ops.aten.permute.default(permute_41, [1, 3, 2, 0]);  permute_41 = None
        view_33 = torch.ops.aten.view.default(permute_43, [6, 129, 129]);  permute_43 = None
        bmm_8 = torch.ops.aten.bmm.default(view_32, view_33);  view_32 = None
        view_34 = torch.ops.aten.view.default(bmm_8, [6, 12, 1, 129]);  bmm_8 = None
        permute_44 = torch.ops.aten.permute.default(view_34, [1, 0, 3, 2]);  view_34 = None
        view_35 = torch.ops.aten.view.default(permute_44, [12, 6, 129]);  permute_44 = None
        sub_12 = torch.ops.aten.sub.Tensor(mul_40, view_35);  mul_40 = view_35 = None
        div_9 = torch.ops.aten.div.Tensor(sub_12, 1.15);  sub_12 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_21, primals_32);  primals_32 = None
        add_27 = torch.ops.aten.add.Tensor(mul_44, primals_33);  mul_44 = primals_33 = None
        sin_7 = torch.ops.aten.sin.default(add_27);  add_27 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(sin_7, 2);  sin_7 = None
        mul_45 = torch.ops.aten.mul.Tensor(pow_8, 0.1);  pow_8 = None
        add_28 = torch.ops.aten.add.Tensor(mul_45, 1);  mul_45 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_28, 0.05);  add_28 = None
        div_10 = torch.ops.aten.div.Tensor(div_8, sub_13);  div_8 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_21, primals_34);  mul_21 = primals_34 = None
        add_30 = torch.ops.aten.add.Tensor(mul_49, primals_35);  mul_49 = primals_35 = None
        sin_8 = torch.ops.aten.sin.default(add_30);  add_30 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(sin_8, 2);  sin_8 = None
        mul_50 = torch.ops.aten.mul.Tensor(pow_9, 0.1);  pow_9 = None
        add_31 = torch.ops.aten.add.Tensor(mul_50, 1);  mul_50 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_31, 0.05);  add_31 = None
        div_11 = torch.ops.aten.div.Tensor(div_9, sub_14);  div_9 = None
        mul_51 = torch.ops.aten.mul.Tensor(div_11, 1j);  div_11 = None
        add_32 = torch.ops.aten.add.Tensor(div_10, mul_51);  div_10 = mul_51 = None
        _fft_c2r = torch.ops.aten._fft_c2r.default(add_32, [2], 1, 256);  add_32 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_3, primals_36);  mul_3 = primals_36 = None
        add_34 = torch.ops.aten.add.Tensor(mul_55, primals_37);  mul_55 = primals_37 = None
        sin_9 = torch.ops.aten.sin.default(add_34);  add_34 = None
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(sin_9, 2);  sin_9 = None
        mul_56 = torch.ops.aten.mul.Tensor(pow_10, 0.1);  pow_10 = None
        add_35 = torch.ops.aten.add.Tensor(mul_56, 1);  mul_56 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_35, 0.05);  add_35 = None
        div_12 = torch.ops.aten.div.Tensor(_fft_c2r, sub_15);  _fft_c2r = None
        permute_46 = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        permute_50 = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
        permute_54 = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
        permute_58 = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
        permute_62 = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
        permute_66 = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_70 = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_74 = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
        permute_78 = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return (div_12, primals_3, primals_1, primals_17, primals_9, primals_10, primals_20, primals_21, primals_22, primals_23, sub_1, sub_2, sub_9, sub_10, sub_13, sub_14, sub_15, permute_46, permute_50, permute_54, permute_58, permute_62, permute_66, permute_70, permute_74, permute_78)
        
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
    buf14 = reader.storage(None, 133128, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf14, (1, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_17
    buf15 = reader.storage(None, 798768, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf15, (6, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_18
    buf16 = reader.storage(None, 798768, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf16, (6, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_19
    buf17 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf17, (), dtype=torch.float64, is_leaf=True)  # primals_20
    buf18 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf18, (), dtype=torch.float64, is_leaf=True)  # primals_21
    buf19 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf19, (), dtype=torch.float64, is_leaf=True)  # primals_22
    buf20 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf20, (), dtype=torch.float64, is_leaf=True)  # primals_23
    buf21 = reader.storage(None, 798768, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf21, (6, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_24
    buf22 = reader.storage(None, 798768, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf22, (6, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_25
    buf23 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf23, (), dtype=torch.float64, is_leaf=True)  # primals_26
    buf24 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf24, (), dtype=torch.float64, is_leaf=True)  # primals_27
    buf25 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf25, (), dtype=torch.float64, is_leaf=True)  # primals_28
    buf26 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf26, (), dtype=torch.float64, is_leaf=True)  # primals_29
    buf27 = reader.storage(None, 798768, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf27, (6, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_30
    buf28 = reader.storage(None, 798768, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf28, (6, 129, 129), dtype=torch.float64, is_leaf=True)  # primals_31
    buf29 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf29, (), dtype=torch.float64, is_leaf=True)  # primals_32
    buf30 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf30, (), dtype=torch.float64, is_leaf=True)  # primals_33
    buf31 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf31, (), dtype=torch.float64, is_leaf=True)  # primals_34
    buf32 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf32, (), dtype=torch.float64, is_leaf=True)  # primals_35
    buf33 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf33, (), dtype=torch.float64, is_leaf=True)  # primals_36
    buf34 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float64)
    reader.tensor(buf34, (), dtype=torch.float64, is_leaf=True)  # primals_37
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)