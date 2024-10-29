class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[6, 256, 256]", primals_2: "f64[12, 6, 256]", primals_3: "f64[256, 256]", primals_4: "f64[6, 3]", primals_5: "i64[4596]", primals_6: "i64[4596]", primals_7: "i64[4596]", primals_8: "i64[4596]", primals_9: "f64[]", primals_10: "f64[]", primals_11: "f64[6, 3]", primals_12: "f64[]", primals_13: "f64[]", primals_14: "f64[6, 3]", primals_15: "f64[]", primals_16: "f64[]"):
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:80 in applyLayer, code: neuralWeights[self.signalInds, self.rowInds, self.colInds] = kernelWeights[self.signalInds, self.kernelInds]
        index: "f64[4596]" = torch.ops.aten.index.Tensor(primals_4, [primals_5, primals_6]);  primals_4 = None
        index_put: "f64[6, 256, 256]" = torch.ops.aten.index_put.default(primals_1, [primals_5, primals_7, primals_8], index);  index = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:83 in applyLayer, code: neuralWeights = neuralWeights + self.stabilityTerm*0.975
        mul: "f64[256, 256]" = torch.ops.aten.mul.Tensor(primals_3, 0.975)
        add: "f64[6, 256, 256]" = torch.ops.aten.add.Tensor(index_put, mul);  index_put = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:89 in applyLayer, code: outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)
        unsqueeze: "f64[12, 6, 256, 1]" = torch.ops.aten.unsqueeze.default(primals_2, 3);  primals_2 = None
        permute: "f64[12, 6, 1, 256]" = torch.ops.aten.permute.default(unsqueeze, [0, 1, 3, 2]);  unsqueeze = None
        unsqueeze_1: "f64[6, 256, 256, 1]" = torch.ops.aten.unsqueeze.default(add, 3);  add = None
        permute_1: "f64[1, 6, 256, 256]" = torch.ops.aten.permute.default(unsqueeze_1, [3, 0, 2, 1]);  unsqueeze_1 = None
        permute_2: "f64[6, 12, 256, 1]" = torch.ops.aten.permute.default(permute, [1, 0, 3, 2]);  permute = None
        view: "f64[6, 12, 256]" = torch.ops.aten.reshape.default(permute_2, [6, 12, 256]);  permute_2 = None
        permute_3: "f64[6, 256, 256, 1]" = torch.ops.aten.permute.default(permute_1, [1, 3, 2, 0]);  permute_1 = None
        view_1: "f64[6, 256, 256]" = torch.ops.aten.reshape.default(permute_3, [6, 256, 256]);  permute_3 = None
        bmm: "f64[6, 12, 256]" = torch.ops.aten.bmm.default(view, view_1);  view = None
        view_2: "f64[6, 12, 1, 256]" = torch.ops.aten.reshape.default(bmm, [6, 12, 1, 256]);  bmm = None
        permute_4: "f64[12, 6, 256, 1]" = torch.ops.aten.permute.default(view_2, [1, 0, 3, 2]);  view_2 = None
        view_3: "f64[12, 6, 256]" = torch.ops.aten.reshape.default(permute_4, [12, 6, 256]);  permute_4 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:234 in getNonLinearity, code: positions = torch.arange(start=0, end=self.sequenceLength, step=1, dtype=torch.float32, device=device)
        iota: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_1: "i64[256]" = torch.ops.aten.mul.Tensor(iota, 1);  iota = None
        add_1: "i64[256]" = torch.ops.aten.add.Tensor(mul_1, 0);  mul_1 = None
        convert_element_type: "f32[256]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:236 in getNonLinearity, code: return self.amplitude*(positions*2*torch.pi*self.learnableFrequency + self.learnablePhaseShift).sin().pow(2) + 1 - self.amplitude/2
        mul_2: "f32[256]" = torch.ops.aten.mul.Tensor(convert_element_type, 2);  convert_element_type = None
        mul_3: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2, 3.141592653589793);  mul_2 = None
        mul_4: "f32[256]" = torch.ops.aten.mul.Tensor(mul_3, primals_9)
        add_2: "f32[256]" = torch.ops.aten.add.Tensor(mul_4, primals_10);  mul_4 = None
        sin: "f32[256]" = torch.ops.aten.sin.default(add_2);  add_2 = None
        pow_1: "f32[256]" = torch.ops.aten.pow.Tensor_Scalar(sin, 2);  sin = None
        mul_5: "f32[256]" = torch.ops.aten.mul.Tensor(pow_1, 0.1);  pow_1 = None
        add_3: "f32[256]" = torch.ops.aten.add.Tensor(mul_5, 1);  mul_5 = None
        sub: "f32[256]" = torch.ops.aten.sub.Tensor(add_3, 0.05);  add_3 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:231 in inversePass, code: return y / nonLinearityTerm
        div: "f64[12, 6, 256]" = torch.ops.aten.div.Tensor(view_3, sub);  view_3 = sub = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:80 in applyLayer, code: neuralWeights[self.signalInds, self.rowInds, self.colInds] = kernelWeights[self.signalInds, self.kernelInds]
        index_1: "f64[4596]" = torch.ops.aten.index.Tensor(primals_11, [primals_5, primals_6]);  primals_11 = None
        index_put_1: "f64[6, 256, 256]" = torch.ops.aten.index_put.default(primals_1, [primals_5, primals_7, primals_8], index_1);  index_1 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:83 in applyLayer, code: neuralWeights = neuralWeights + self.stabilityTerm*0.975
        add_4: "f64[6, 256, 256]" = torch.ops.aten.add.Tensor(index_put_1, mul);  index_put_1 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:89 in applyLayer, code: outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)
        unsqueeze_2: "f64[12, 6, 256, 1]" = torch.ops.aten.unsqueeze.default(div, 3);  div = None
        permute_5: "f64[12, 6, 1, 256]" = torch.ops.aten.permute.default(unsqueeze_2, [0, 1, 3, 2]);  unsqueeze_2 = None
        unsqueeze_3: "f64[6, 256, 256, 1]" = torch.ops.aten.unsqueeze.default(add_4, 3);  add_4 = None
        permute_6: "f64[1, 6, 256, 256]" = torch.ops.aten.permute.default(unsqueeze_3, [3, 0, 2, 1]);  unsqueeze_3 = None
        permute_7: "f64[6, 12, 256, 1]" = torch.ops.aten.permute.default(permute_5, [1, 0, 3, 2]);  permute_5 = None
        view_4: "f64[6, 12, 256]" = torch.ops.aten.reshape.default(permute_7, [6, 12, 256]);  permute_7 = None
        permute_8: "f64[6, 256, 256, 1]" = torch.ops.aten.permute.default(permute_6, [1, 3, 2, 0]);  permute_6 = None
        view_5: "f64[6, 256, 256]" = torch.ops.aten.reshape.default(permute_8, [6, 256, 256]);  permute_8 = None
        bmm_1: "f64[6, 12, 256]" = torch.ops.aten.bmm.default(view_4, view_5);  view_4 = None
        view_6: "f64[6, 12, 1, 256]" = torch.ops.aten.reshape.default(bmm_1, [6, 12, 1, 256]);  bmm_1 = None
        permute_9: "f64[12, 6, 256, 1]" = torch.ops.aten.permute.default(view_6, [1, 0, 3, 2]);  view_6 = None
        view_7: "f64[12, 6, 256]" = torch.ops.aten.reshape.default(permute_9, [12, 6, 256]);  permute_9 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:236 in getNonLinearity, code: return self.amplitude*(positions*2*torch.pi*self.learnableFrequency + self.learnablePhaseShift).sin().pow(2) + 1 - self.amplitude/2
        mul_10: "f32[256]" = torch.ops.aten.mul.Tensor(mul_3, primals_12);  primals_12 = None
        add_6: "f32[256]" = torch.ops.aten.add.Tensor(mul_10, primals_13);  mul_10 = primals_13 = None
        sin_1: "f32[256]" = torch.ops.aten.sin.default(add_6);  add_6 = None
        pow_2: "f32[256]" = torch.ops.aten.pow.Tensor_Scalar(sin_1, 2);  sin_1 = None
        mul_11: "f32[256]" = torch.ops.aten.mul.Tensor(pow_2, 0.1);  pow_2 = None
        add_7: "f32[256]" = torch.ops.aten.add.Tensor(mul_11, 1);  mul_11 = None
        sub_1: "f32[256]" = torch.ops.aten.sub.Tensor(add_7, 0.05);  add_7 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:227 in forwardPass, code: return x * nonLinearityTerm
        mul_12: "f64[12, 6, 256]" = torch.ops.aten.mul.Tensor(view_7, sub_1);  view_7 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:80 in applyLayer, code: neuralWeights[self.signalInds, self.rowInds, self.colInds] = kernelWeights[self.signalInds, self.kernelInds]
        index_2: "f64[4596]" = torch.ops.aten.index.Tensor(primals_14, [primals_5, primals_6]);  primals_14 = primals_6 = None
        index_put_2: "f64[6, 256, 256]" = torch.ops.aten.index_put.default(primals_1, [primals_5, primals_7, primals_8], index_2);  primals_5 = primals_7 = primals_8 = index_2 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:83 in applyLayer, code: neuralWeights = neuralWeights + self.stabilityTerm*0.975
        add_8: "f64[6, 256, 256]" = torch.ops.aten.add.Tensor(index_put_2, mul);  index_put_2 = mul = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:89 in applyLayer, code: outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)
        unsqueeze_4: "f64[12, 6, 256, 1]" = torch.ops.aten.unsqueeze.default(mul_12, 3);  mul_12 = None
        permute_10: "f64[12, 6, 1, 256]" = torch.ops.aten.permute.default(unsqueeze_4, [0, 1, 3, 2]);  unsqueeze_4 = None
        unsqueeze_5: "f64[6, 256, 256, 1]" = torch.ops.aten.unsqueeze.default(add_8, 3);  add_8 = None
        permute_11: "f64[1, 6, 256, 256]" = torch.ops.aten.permute.default(unsqueeze_5, [3, 0, 2, 1]);  unsqueeze_5 = None
        permute_12: "f64[6, 12, 256, 1]" = torch.ops.aten.permute.default(permute_10, [1, 0, 3, 2]);  permute_10 = None
        view_8: "f64[6, 12, 256]" = torch.ops.aten.reshape.default(permute_12, [6, 12, 256]);  permute_12 = None
        permute_13: "f64[6, 256, 256, 1]" = torch.ops.aten.permute.default(permute_11, [1, 3, 2, 0]);  permute_11 = None
        view_9: "f64[6, 256, 256]" = torch.ops.aten.reshape.default(permute_13, [6, 256, 256]);  permute_13 = None
        bmm_2: "f64[6, 12, 256]" = torch.ops.aten.bmm.default(view_8, view_9);  view_8 = None
        view_10: "f64[6, 12, 1, 256]" = torch.ops.aten.reshape.default(bmm_2, [6, 12, 1, 256]);  bmm_2 = None
        permute_14: "f64[12, 6, 256, 1]" = torch.ops.aten.permute.default(view_10, [1, 0, 3, 2]);  view_10 = None
        view_11: "f64[12, 6, 256]" = torch.ops.aten.reshape.default(permute_14, [12, 6, 256]);  permute_14 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:236 in getNonLinearity, code: return self.amplitude*(positions*2*torch.pi*self.learnableFrequency + self.learnablePhaseShift).sin().pow(2) + 1 - self.amplitude/2
        mul_17: "f32[256]" = torch.ops.aten.mul.Tensor(mul_3, primals_15);  mul_3 = primals_15 = None
        add_10: "f32[256]" = torch.ops.aten.add.Tensor(mul_17, primals_16);  mul_17 = primals_16 = None
        sin_2: "f32[256]" = torch.ops.aten.sin.default(add_10);  add_10 = None
        pow_3: "f32[256]" = torch.ops.aten.pow.Tensor_Scalar(sin_2, 2);  sin_2 = None
        mul_18: "f32[256]" = torch.ops.aten.mul.Tensor(pow_3, 0.1);  pow_3 = None
        add_11: "f32[256]" = torch.ops.aten.add.Tensor(mul_18, 1);  mul_18 = None
        sub_2: "f32[256]" = torch.ops.aten.sub.Tensor(add_11, 0.05);  add_11 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:231 in inversePass, code: return y / nonLinearityTerm
        div_1: "f64[12, 6, 256]" = torch.ops.aten.div.Tensor(view_11, sub_2);  view_11 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:89 in applyLayer, code: outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)
        permute_16: "f64[6, 256, 256]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_20: "f64[6, 256, 256]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
        permute_24: "f64[6, 256, 256]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return (div_1, primals_3, primals_1, primals_9, primals_10, sub_1, sub_2, permute_16, permute_20, permute_24)
        