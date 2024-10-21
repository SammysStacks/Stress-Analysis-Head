class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[110, 129, 129]", primals_2: "f64[110, 129, 129]", primals_3: "f64[110, 129, 129]", primals_4: "f64[110, 129, 129]", primals_5: "f64[1, 129, 129]", primals_6: "f64[26, 110, 129]", primals_7: "f64[129, 129]", primals_8: "f64[]", primals_9: "f64[]", primals_10: "f64[]"):
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleLinearLayer.py:36 in forward, code: self.restrictedWindowMask = self.restrictedWindowMask.to(inputData.device)
        device_put: "f64[1, 129, 129]" = torch.ops.prims.device_put.default(primals_5, device(type='cuda', index=0));  primals_5 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleLinearLayer.py:37 in forward, code: self.stabilityTerm = self.stabilityTerm.to(inputData.device)
        device_put_1: "f64[129, 129]" = torch.ops.prims.device_put.default(primals_7, device(type='cuda', index=0));  primals_7 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:221 in getNonLinearity, code: positions = torch.arange(start=0, end=self.sequenceLength, step=1, dtype=torch.float32, device=device)
        iota: "i64[129]" = torch.ops.prims.iota.default(129, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul: "i64[129]" = torch.ops.aten.mul.Tensor(iota, 1);  iota = None
        add: "i64[129]" = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        convert_element_type_2: "f32[129]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:223 in getNonLinearity, code: return self.learnableAmplitude*(positions*2*torch.pi*self.learnableFrequency + self.learnablePhaseShift).sin().round(decimals=6)
        mul_1: "f32[129]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 2);  convert_element_type_2 = None
        mul_2: "f32[129]" = torch.ops.aten.mul.Tensor(mul_1, 3.141592653589793);  mul_1 = None
        mul_3: "f32[129]" = torch.ops.aten.mul.Tensor(mul_2, primals_9);  mul_2 = None
        add_1: "f32[129]" = torch.ops.aten.add.Tensor(mul_3, primals_10);  mul_3 = None
        sin: "f32[129]" = torch.ops.aten.sin.default(add_1);  add_1 = None
        mul_4: "f32[129]" = torch.ops.aten.mul.Tensor(sin, 1000000.0);  sin = None
        round_1: "f32[129]" = torch.ops.aten.round.default(mul_4);  mul_4 = None
        mul_5: "f32[129]" = torch.ops.aten.mul.Tensor(round_1, 1e-06);  round_1 = None
        mul_6: "f32[129]" = torch.ops.aten.mul.Tensor(primals_8, mul_5);  primals_8 = mul_5 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:207 in forward, code: nonLinearityTerm = self.getNonLinearity(device=x.device)*coefficient
        mul_7: "f32[129]" = torch.ops.aten.mul.Tensor(mul_6, 1)
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:218 in inversePass, code: return y - nonLinearityTerm
        sub: "f64[26, 110, 129]" = torch.ops.aten.sub.Tensor(primals_6, mul_7);  primals_6 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleLinearLayer.py:57 in applyLayer, code: if self.kernelSize != self.sequenceLength: neuralWeights = self.restrictedWindowMask * neuralWeights + self.stabilityTerm
        mul_8: "f64[110, 129, 129]" = torch.ops.aten.mul.Tensor(device_put, primals_1);  primals_1 = None
        add_2: "f64[110, 129, 129]" = torch.ops.aten.add.Tensor(mul_8, device_put_1);  mul_8 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleLinearLayer.py:64 in applyLayer, code: outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)
        unsqueeze: "f64[26, 110, 129, 1]" = torch.ops.aten.unsqueeze.default(sub, 3);  sub = None
        permute: "f64[26, 110, 1, 129]" = torch.ops.aten.permute.default(unsqueeze, [0, 1, 3, 2]);  unsqueeze = None
        unsqueeze_1: "f64[110, 129, 129, 1]" = torch.ops.aten.unsqueeze.default(add_2, 3);  add_2 = None
        permute_1: "f64[1, 110, 129, 129]" = torch.ops.aten.permute.default(unsqueeze_1, [3, 0, 2, 1]);  unsqueeze_1 = None
        permute_2: "f64[110, 26, 129, 1]" = torch.ops.aten.permute.default(permute, [1, 0, 3, 2]);  permute = None
        view: "f64[110, 26, 129]" = torch.ops.aten.view.default(permute_2, [110, 26, 129]);  permute_2 = None
        permute_3: "f64[110, 129, 129, 1]" = torch.ops.aten.permute.default(permute_1, [1, 3, 2, 0]);  permute_1 = None
        view_1: "f64[110, 129, 129]" = torch.ops.aten.view.default(permute_3, [110, 129, 129]);  permute_3 = None
        bmm: "f64[110, 26, 129]" = torch.ops.aten.bmm.default(view, view_1)
        view_2: "f64[110, 26, 1, 129]" = torch.ops.aten.view.default(bmm, [110, 26, 1, 129]);  bmm = None
        permute_4: "f64[26, 110, 129, 1]" = torch.ops.aten.permute.default(view_2, [1, 0, 3, 2]);  view_2 = None
        view_3: "f64[26, 110, 129]" = torch.ops.aten.view.default(permute_4, [26, 110, 129]);  permute_4 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:207 in forward, code: nonLinearityTerm = self.getNonLinearity(device=x.device)*coefficient
        mul_16: "f32[129]" = torch.ops.aten.mul.Tensor(mul_6, -1);  mul_6 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:218 in inversePass, code: return y - nonLinearityTerm
        sub_1: "f64[26, 110, 129]" = torch.ops.aten.sub.Tensor(view_3, mul_16);  view_3 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleLinearLayer.py:57 in applyLayer, code: if self.kernelSize != self.sequenceLength: neuralWeights = self.restrictedWindowMask * neuralWeights + self.stabilityTerm
        mul_17: "f64[110, 129, 129]" = torch.ops.aten.mul.Tensor(device_put, primals_2);  primals_2 = None
        add_5: "f64[110, 129, 129]" = torch.ops.aten.add.Tensor(mul_17, device_put_1);  mul_17 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleLinearLayer.py:64 in applyLayer, code: outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)
        unsqueeze_2: "f64[26, 110, 129, 1]" = torch.ops.aten.unsqueeze.default(sub_1, 3);  sub_1 = None
        permute_5: "f64[26, 110, 1, 129]" = torch.ops.aten.permute.default(unsqueeze_2, [0, 1, 3, 2]);  unsqueeze_2 = None
        unsqueeze_3: "f64[110, 129, 129, 1]" = torch.ops.aten.unsqueeze.default(add_5, 3);  add_5 = None
        permute_6: "f64[1, 110, 129, 129]" = torch.ops.aten.permute.default(unsqueeze_3, [3, 0, 2, 1]);  unsqueeze_3 = None
        permute_7: "f64[110, 26, 129, 1]" = torch.ops.aten.permute.default(permute_5, [1, 0, 3, 2]);  permute_5 = None
        view_4: "f64[110, 26, 129]" = torch.ops.aten.view.default(permute_7, [110, 26, 129]);  permute_7 = None
        permute_8: "f64[110, 129, 129, 1]" = torch.ops.aten.permute.default(permute_6, [1, 3, 2, 0]);  permute_6 = None
        view_5: "f64[110, 129, 129]" = torch.ops.aten.view.default(permute_8, [110, 129, 129]);  permute_8 = None
        bmm_1: "f64[110, 26, 129]" = torch.ops.aten.bmm.default(view_4, view_5)
        view_6: "f64[110, 26, 1, 129]" = torch.ops.aten.view.default(bmm_1, [110, 26, 1, 129]);  bmm_1 = None
        permute_9: "f64[26, 110, 129, 1]" = torch.ops.aten.permute.default(view_6, [1, 0, 3, 2]);  view_6 = None
        view_7: "f64[26, 110, 129]" = torch.ops.aten.view.default(permute_9, [26, 110, 129]);  permute_9 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:218 in inversePass, code: return y - nonLinearityTerm
        sub_2: "f64[26, 110, 129]" = torch.ops.aten.sub.Tensor(view_7, mul_7);  view_7 = mul_7 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleLinearLayer.py:57 in applyLayer, code: if self.kernelSize != self.sequenceLength: neuralWeights = self.restrictedWindowMask * neuralWeights + self.stabilityTerm
        mul_26: "f64[110, 129, 129]" = torch.ops.aten.mul.Tensor(device_put, primals_3);  primals_3 = None
        add_8: "f64[110, 129, 129]" = torch.ops.aten.add.Tensor(mul_26, device_put_1);  mul_26 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleLinearLayer.py:64 in applyLayer, code: outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)
        unsqueeze_4: "f64[26, 110, 129, 1]" = torch.ops.aten.unsqueeze.default(sub_2, 3);  sub_2 = None
        permute_10: "f64[26, 110, 1, 129]" = torch.ops.aten.permute.default(unsqueeze_4, [0, 1, 3, 2]);  unsqueeze_4 = None
        unsqueeze_5: "f64[110, 129, 129, 1]" = torch.ops.aten.unsqueeze.default(add_8, 3);  add_8 = None
        permute_11: "f64[1, 110, 129, 129]" = torch.ops.aten.permute.default(unsqueeze_5, [3, 0, 2, 1]);  unsqueeze_5 = None
        permute_12: "f64[110, 26, 129, 1]" = torch.ops.aten.permute.default(permute_10, [1, 0, 3, 2]);  permute_10 = None
        view_8: "f64[110, 26, 129]" = torch.ops.aten.view.default(permute_12, [110, 26, 129]);  permute_12 = None
        permute_13: "f64[110, 129, 129, 1]" = torch.ops.aten.permute.default(permute_11, [1, 3, 2, 0]);  permute_11 = None
        view_9: "f64[110, 129, 129]" = torch.ops.aten.view.default(permute_13, [110, 129, 129]);  permute_13 = None
        bmm_2: "f64[110, 26, 129]" = torch.ops.aten.bmm.default(view_8, view_9)
        view_10: "f64[110, 26, 1, 129]" = torch.ops.aten.view.default(bmm_2, [110, 26, 1, 129]);  bmm_2 = None
        permute_14: "f64[26, 110, 129, 1]" = torch.ops.aten.permute.default(view_10, [1, 0, 3, 2]);  view_10 = None
        view_11: "f64[26, 110, 129]" = torch.ops.aten.view.default(permute_14, [26, 110, 129]);  permute_14 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:218 in inversePass, code: return y - nonLinearityTerm
        sub_3: "f64[26, 110, 129]" = torch.ops.aten.sub.Tensor(view_11, mul_16);  view_11 = mul_16 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleLinearLayer.py:57 in applyLayer, code: if self.kernelSize != self.sequenceLength: neuralWeights = self.restrictedWindowMask * neuralWeights + self.stabilityTerm
        mul_35: "f64[110, 129, 129]" = torch.ops.aten.mul.Tensor(device_put, primals_4);  primals_4 = None
        add_11: "f64[110, 129, 129]" = torch.ops.aten.add.Tensor(mul_35, device_put_1);  mul_35 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleLinearLayer.py:64 in applyLayer, code: outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)
        unsqueeze_6: "f64[26, 110, 129, 1]" = torch.ops.aten.unsqueeze.default(sub_3, 3);  sub_3 = None
        permute_15: "f64[26, 110, 1, 129]" = torch.ops.aten.permute.default(unsqueeze_6, [0, 1, 3, 2]);  unsqueeze_6 = None
        unsqueeze_7: "f64[110, 129, 129, 1]" = torch.ops.aten.unsqueeze.default(add_11, 3);  add_11 = None
        permute_16: "f64[1, 110, 129, 129]" = torch.ops.aten.permute.default(unsqueeze_7, [3, 0, 2, 1]);  unsqueeze_7 = None
        permute_17: "f64[110, 26, 129, 1]" = torch.ops.aten.permute.default(permute_15, [1, 0, 3, 2]);  permute_15 = None
        view_12: "f64[110, 26, 129]" = torch.ops.aten.view.default(permute_17, [110, 26, 129]);  permute_17 = None
        permute_18: "f64[110, 129, 129, 1]" = torch.ops.aten.permute.default(permute_16, [1, 3, 2, 0]);  permute_16 = None
        view_13: "f64[110, 129, 129]" = torch.ops.aten.view.default(permute_18, [110, 129, 129]);  permute_18 = None
        bmm_3: "f64[110, 26, 129]" = torch.ops.aten.bmm.default(view_12, view_13)
        view_14: "f64[110, 26, 1, 129]" = torch.ops.aten.view.default(bmm_3, [110, 26, 1, 129]);  bmm_3 = None
        permute_19: "f64[26, 110, 129, 1]" = torch.ops.aten.permute.default(view_14, [1, 0, 3, 2]);  view_14 = None
        view_15: "f64[26, 110, 129]" = torch.ops.aten.view.default(permute_19, [26, 110, 129]);  permute_19 = None
        permute_21: "f64[110, 129, 26]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_22: "f64[110, 129, 129]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_28: "f64[110, 129, 26]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
        permute_29: "f64[110, 129, 129]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_35: "f64[110, 129, 26]" = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
        permute_36: "f64[110, 129, 129]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
        permute_42: "f64[110, 129, 26]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        permute_43: "f64[110, 129, 129]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return [view_15, device_put, device_put_1, primals_9, primals_10, device_put, permute_21, permute_22, permute_28, permute_29, permute_35, permute_36, permute_42, permute_43]
        