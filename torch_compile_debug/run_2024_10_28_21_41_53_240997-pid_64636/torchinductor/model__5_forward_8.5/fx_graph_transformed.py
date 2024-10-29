class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[6, 256, 256]", primals_2: "f64[12, 6, 256]", primals_3: "f64[256, 256]", primals_4: "f64[6, 3]", primals_5: "i64[4596]", primals_6: "i64[4596]", primals_7: "i64[4596]", primals_8: "i64[4596]", primals_9: "f64[]", primals_10: "f64[]", primals_11: "f64[6, 3]", primals_12: "f64[]", primals_13: "f64[]", primals_14: "f64[6, 3]", primals_15: "f64[]", primals_16: "f64[]", primals_17: "f64[1, 129, 129]", primals_18: "f64[6, 129, 129]", primals_19: "f64[6, 129, 129]", primals_20: "f64[]", primals_21: "f64[]", primals_22: "f64[]", primals_23: "f64[]", primals_24: "f64[6, 129, 129]", primals_25: "f64[6, 129, 129]", primals_26: "f64[]", primals_27: "f64[]", primals_28: "f64[]", primals_29: "f64[]", primals_30: "f64[6, 129, 129]", primals_31: "f64[6, 129, 129]", primals_32: "f64[]", primals_33: "f64[]", primals_34: "f64[]", primals_35: "f64[]", primals_36: "f64[]", primals_37: "f64[]"):
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
        mul_17: "f32[256]" = torch.ops.aten.mul.Tensor(mul_3, primals_15);  primals_15 = None
        add_10: "f32[256]" = torch.ops.aten.add.Tensor(mul_17, primals_16);  mul_17 = primals_16 = None
        sin_2: "f32[256]" = torch.ops.aten.sin.default(add_10);  add_10 = None
        pow_3: "f32[256]" = torch.ops.aten.pow.Tensor_Scalar(sin_2, 2);  sin_2 = None
        mul_18: "f32[256]" = torch.ops.aten.mul.Tensor(pow_3, 0.1);  pow_3 = None
        add_11: "f32[256]" = torch.ops.aten.add.Tensor(mul_18, 1);  mul_18 = None
        sub_2: "f32[256]" = torch.ops.aten.sub.Tensor(add_11, 0.05);  add_11 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:231 in inversePass, code: return y / nonLinearityTerm
        div_1: "f64[12, 6, 256]" = torch.ops.aten.div.Tensor(view_11, sub_2);  view_11 = None
        
         # File: C:\Users\sasol.SQUIRTLE\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\functional.py:5106 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "f64[12, 6, 256]" = torch.ops.aten.constant_pad_nd.default(div_1, [0, 0], 0.0);  div_1 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\neuralOperators\fourierOperator\fourierNeuralOperatorWeights.py:72 in forwardFFT, code: fourierData = torch.fft.rfft(inputData, n=self.sequenceLength, dim=-1, norm='ortho')
        _fft_r2c: "c128[12, 6, 129]" = torch.ops.aten._fft_r2c.default(constant_pad_nd, [2], 1, True);  constant_pad_nd = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\neuralOperators\fourierOperator\fourierNeuralOperatorWeights.py:73 in forwardFFT, code: imaginaryFourierData = fourierData.imag
        view_as_real: "f64[12, 6, 129, 2]" = torch.ops.aten.view_as_real.default(_fft_r2c);  _fft_r2c = None
        select: "f64[12, 6, 129]" = torch.ops.aten.select.int(view_as_real, 3, 1)
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\neuralOperators\fourierOperator\fourierNeuralOperatorWeights.py:74 in forwardFFT, code: realFourierData = fourierData.real
        select_1: "f64[12, 6, 129]" = torch.ops.aten.select.int(view_as_real, 3, 0);  view_as_real = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\neuralOperators\fourierOperator\fourierNeuralOperatorLayer.py:79 in fourierNeuralOperator, code: realFourierData, imaginaryFourierData = self.dualFrequencyWeights(realFourierData + (realFrequencyTerms or 0), imaginaryFourierData + (imaginaryFrequencyTerms or 0))
        add_12: "f64[12, 6, 129]" = torch.ops.aten.add.Tensor(select_1, 0);  select_1 = None
        add_13: "f64[12, 6, 129]" = torch.ops.aten.add.Tensor(select, 0);  select = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleDualLinearLayer.py:91 in applyLayer, code: y2 = (x2 - torch.einsum('bns,nsi->bni', x1, A2)) / self.stabilityFactor
        unsqueeze_6: "f64[12, 6, 129, 1]" = torch.ops.aten.unsqueeze.default(add_13, 3)
        permute_15: "f64[12, 6, 1, 129]" = torch.ops.aten.permute.default(unsqueeze_6, [0, 1, 3, 2]);  unsqueeze_6 = None
        unsqueeze_7: "f64[6, 129, 129, 1]" = torch.ops.aten.unsqueeze.default(primals_19, 3);  primals_19 = None
        permute_16: "f64[1, 6, 129, 129]" = torch.ops.aten.permute.default(unsqueeze_7, [3, 0, 2, 1]);  unsqueeze_7 = None
        permute_17: "f64[6, 12, 129, 1]" = torch.ops.aten.permute.default(permute_15, [1, 0, 3, 2]);  permute_15 = None
        view_12: "f64[6, 12, 129]" = torch.ops.aten.reshape.default(permute_17, [6, 12, 129]);  permute_17 = None
        permute_18: "f64[6, 129, 129, 1]" = torch.ops.aten.permute.default(permute_16, [1, 3, 2, 0]);  permute_16 = None
        view_13: "f64[6, 129, 129]" = torch.ops.aten.reshape.default(permute_18, [6, 129, 129]);  permute_18 = None
        bmm_3: "f64[6, 12, 129]" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = None
        view_14: "f64[6, 12, 1, 129]" = torch.ops.aten.reshape.default(bmm_3, [6, 12, 1, 129]);  bmm_3 = None
        permute_19: "f64[12, 6, 129, 1]" = torch.ops.aten.permute.default(view_14, [1, 0, 3, 2]);  view_14 = None
        view_15: "f64[12, 6, 129]" = torch.ops.aten.reshape.default(permute_19, [12, 6, 129]);  permute_19 = None
        sub_3: "f64[12, 6, 129]" = torch.ops.aten.sub.Tensor(add_12, view_15);  add_12 = view_15 = None
        div_2: "f64[12, 6, 129]" = torch.ops.aten.div.Tensor(sub_3, 1.15);  sub_3 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleDualLinearLayer.py:92 in applyLayer, code: y1 = (x1 - torch.einsum('bns,nsi->bni', y2, A1)) / self.stabilityFactor
        unsqueeze_8: "f64[12, 6, 129, 1]" = torch.ops.aten.unsqueeze.default(div_2, 3)
        permute_20: "f64[12, 6, 1, 129]" = torch.ops.aten.permute.default(unsqueeze_8, [0, 1, 3, 2]);  unsqueeze_8 = None
        unsqueeze_9: "f64[6, 129, 129, 1]" = torch.ops.aten.unsqueeze.default(primals_18, 3);  primals_18 = None
        permute_21: "f64[1, 6, 129, 129]" = torch.ops.aten.permute.default(unsqueeze_9, [3, 0, 2, 1]);  unsqueeze_9 = None
        permute_22: "f64[6, 12, 129, 1]" = torch.ops.aten.permute.default(permute_20, [1, 0, 3, 2]);  permute_20 = None
        view_16: "f64[6, 12, 129]" = torch.ops.aten.reshape.default(permute_22, [6, 12, 129]);  permute_22 = None
        permute_23: "f64[6, 129, 129, 1]" = torch.ops.aten.permute.default(permute_21, [1, 3, 2, 0]);  permute_21 = None
        view_17: "f64[6, 129, 129]" = torch.ops.aten.reshape.default(permute_23, [6, 129, 129]);  permute_23 = None
        bmm_4: "f64[6, 12, 129]" = torch.ops.aten.bmm.default(view_16, view_17);  view_16 = None
        view_18: "f64[6, 12, 1, 129]" = torch.ops.aten.reshape.default(bmm_4, [6, 12, 1, 129]);  bmm_4 = None
        permute_24: "f64[12, 6, 129, 1]" = torch.ops.aten.permute.default(view_18, [1, 0, 3, 2]);  view_18 = None
        view_19: "f64[12, 6, 129]" = torch.ops.aten.reshape.default(permute_24, [12, 6, 129]);  permute_24 = None
        sub_4: "f64[12, 6, 129]" = torch.ops.aten.sub.Tensor(add_13, view_19);  add_13 = view_19 = None
        div_3: "f64[12, 6, 129]" = torch.ops.aten.div.Tensor(sub_4, 1.15);  sub_4 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:234 in getNonLinearity, code: positions = torch.arange(start=0, end=self.sequenceLength, step=1, dtype=torch.float32, device=device)
        iota_3: "i64[129]" = torch.ops.prims.iota.default(129, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_19: "i64[129]" = torch.ops.aten.mul.Tensor(iota_3, 1);  iota_3 = None
        add_14: "i64[129]" = torch.ops.aten.add.Tensor(mul_19, 0);  mul_19 = None
        convert_element_type_3: "f32[129]" = torch.ops.prims.convert_element_type.default(add_14, torch.float32);  add_14 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:236 in getNonLinearity, code: return self.amplitude*(positions*2*torch.pi*self.learnableFrequency + self.learnablePhaseShift).sin().pow(2) + 1 - self.amplitude/2
        mul_20: "f32[129]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 2);  convert_element_type_3 = None
        mul_21: "f32[129]" = torch.ops.aten.mul.Tensor(mul_20, 3.141592653589793);  mul_20 = None
        mul_22: "f32[129]" = torch.ops.aten.mul.Tensor(mul_21, primals_20)
        add_15: "f32[129]" = torch.ops.aten.add.Tensor(mul_22, primals_21);  mul_22 = None
        sin_3: "f32[129]" = torch.ops.aten.sin.default(add_15);  add_15 = None
        pow_4: "f32[129]" = torch.ops.aten.pow.Tensor_Scalar(sin_3, 2);  sin_3 = None
        mul_23: "f32[129]" = torch.ops.aten.mul.Tensor(pow_4, 0.1);  pow_4 = None
        add_16: "f32[129]" = torch.ops.aten.add.Tensor(mul_23, 1);  mul_23 = None
        sub_5: "f32[129]" = torch.ops.aten.sub.Tensor(add_16, 0.05);  add_16 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:231 in inversePass, code: return y / nonLinearityTerm
        div_4: "f64[12, 6, 129]" = torch.ops.aten.div.Tensor(div_2, sub_5);  div_2 = sub_5 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:236 in getNonLinearity, code: return self.amplitude*(positions*2*torch.pi*self.learnableFrequency + self.learnablePhaseShift).sin().pow(2) + 1 - self.amplitude/2
        mul_27: "f32[129]" = torch.ops.aten.mul.Tensor(mul_21, primals_22)
        add_18: "f32[129]" = torch.ops.aten.add.Tensor(mul_27, primals_23);  mul_27 = None
        sin_4: "f32[129]" = torch.ops.aten.sin.default(add_18);  add_18 = None
        pow_5: "f32[129]" = torch.ops.aten.pow.Tensor_Scalar(sin_4, 2);  sin_4 = None
        mul_28: "f32[129]" = torch.ops.aten.mul.Tensor(pow_5, 0.1);  pow_5 = None
        add_19: "f32[129]" = torch.ops.aten.add.Tensor(mul_28, 1);  mul_28 = None
        sub_6: "f32[129]" = torch.ops.aten.sub.Tensor(add_19, 0.05);  add_19 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:231 in inversePass, code: return y / nonLinearityTerm
        div_5: "f64[12, 6, 129]" = torch.ops.aten.div.Tensor(div_3, sub_6);  div_3 = sub_6 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleDualLinearLayer.py:91 in applyLayer, code: y2 = (x2 - torch.einsum('bns,nsi->bni', x1, A2)) / self.stabilityFactor
        unsqueeze_10: "f64[12, 6, 129, 1]" = torch.ops.aten.unsqueeze.default(div_4, 3)
        permute_25: "f64[12, 6, 1, 129]" = torch.ops.aten.permute.default(unsqueeze_10, [0, 1, 3, 2]);  unsqueeze_10 = None
        unsqueeze_11: "f64[6, 129, 129, 1]" = torch.ops.aten.unsqueeze.default(primals_25, 3);  primals_25 = None
        permute_26: "f64[1, 6, 129, 129]" = torch.ops.aten.permute.default(unsqueeze_11, [3, 0, 2, 1]);  unsqueeze_11 = None
        permute_27: "f64[6, 12, 129, 1]" = torch.ops.aten.permute.default(permute_25, [1, 0, 3, 2]);  permute_25 = None
        view_20: "f64[6, 12, 129]" = torch.ops.aten.reshape.default(permute_27, [6, 12, 129]);  permute_27 = None
        permute_28: "f64[6, 129, 129, 1]" = torch.ops.aten.permute.default(permute_26, [1, 3, 2, 0]);  permute_26 = None
        view_21: "f64[6, 129, 129]" = torch.ops.aten.reshape.default(permute_28, [6, 129, 129]);  permute_28 = None
        bmm_5: "f64[6, 12, 129]" = torch.ops.aten.bmm.default(view_20, view_21);  view_20 = None
        view_22: "f64[6, 12, 1, 129]" = torch.ops.aten.reshape.default(bmm_5, [6, 12, 1, 129]);  bmm_5 = None
        permute_29: "f64[12, 6, 129, 1]" = torch.ops.aten.permute.default(view_22, [1, 0, 3, 2]);  view_22 = None
        view_23: "f64[12, 6, 129]" = torch.ops.aten.reshape.default(permute_29, [12, 6, 129]);  permute_29 = None
        sub_7: "f64[12, 6, 129]" = torch.ops.aten.sub.Tensor(div_5, view_23);  div_5 = view_23 = None
        div_6: "f64[12, 6, 129]" = torch.ops.aten.div.Tensor(sub_7, 1.15);  sub_7 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleDualLinearLayer.py:92 in applyLayer, code: y1 = (x1 - torch.einsum('bns,nsi->bni', y2, A1)) / self.stabilityFactor
        unsqueeze_12: "f64[12, 6, 129, 1]" = torch.ops.aten.unsqueeze.default(div_6, 3)
        permute_30: "f64[12, 6, 1, 129]" = torch.ops.aten.permute.default(unsqueeze_12, [0, 1, 3, 2]);  unsqueeze_12 = None
        unsqueeze_13: "f64[6, 129, 129, 1]" = torch.ops.aten.unsqueeze.default(primals_24, 3);  primals_24 = None
        permute_31: "f64[1, 6, 129, 129]" = torch.ops.aten.permute.default(unsqueeze_13, [3, 0, 2, 1]);  unsqueeze_13 = None
        permute_32: "f64[6, 12, 129, 1]" = torch.ops.aten.permute.default(permute_30, [1, 0, 3, 2]);  permute_30 = None
        view_24: "f64[6, 12, 129]" = torch.ops.aten.reshape.default(permute_32, [6, 12, 129]);  permute_32 = None
        permute_33: "f64[6, 129, 129, 1]" = torch.ops.aten.permute.default(permute_31, [1, 3, 2, 0]);  permute_31 = None
        view_25: "f64[6, 129, 129]" = torch.ops.aten.reshape.default(permute_33, [6, 129, 129]);  permute_33 = None
        bmm_6: "f64[6, 12, 129]" = torch.ops.aten.bmm.default(view_24, view_25);  view_24 = None
        view_26: "f64[6, 12, 1, 129]" = torch.ops.aten.reshape.default(bmm_6, [6, 12, 1, 129]);  bmm_6 = None
        permute_34: "f64[12, 6, 129, 1]" = torch.ops.aten.permute.default(view_26, [1, 0, 3, 2]);  view_26 = None
        view_27: "f64[12, 6, 129]" = torch.ops.aten.reshape.default(permute_34, [12, 6, 129]);  permute_34 = None
        sub_8: "f64[12, 6, 129]" = torch.ops.aten.sub.Tensor(div_4, view_27);  div_4 = view_27 = None
        div_7: "f64[12, 6, 129]" = torch.ops.aten.div.Tensor(sub_8, 1.15);  sub_8 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:236 in getNonLinearity, code: return self.amplitude*(positions*2*torch.pi*self.learnableFrequency + self.learnablePhaseShift).sin().pow(2) + 1 - self.amplitude/2
        mul_32: "f32[129]" = torch.ops.aten.mul.Tensor(mul_21, primals_26);  primals_26 = None
        add_21: "f32[129]" = torch.ops.aten.add.Tensor(mul_32, primals_27);  mul_32 = primals_27 = None
        sin_5: "f32[129]" = torch.ops.aten.sin.default(add_21);  add_21 = None
        pow_6: "f32[129]" = torch.ops.aten.pow.Tensor_Scalar(sin_5, 2);  sin_5 = None
        mul_33: "f32[129]" = torch.ops.aten.mul.Tensor(pow_6, 0.1);  pow_6 = None
        add_22: "f32[129]" = torch.ops.aten.add.Tensor(mul_33, 1);  mul_33 = None
        sub_9: "f32[129]" = torch.ops.aten.sub.Tensor(add_22, 0.05);  add_22 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:227 in forwardPass, code: return x * nonLinearityTerm
        mul_34: "f64[12, 6, 129]" = torch.ops.aten.mul.Tensor(div_7, sub_9);  div_7 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:236 in getNonLinearity, code: return self.amplitude*(positions*2*torch.pi*self.learnableFrequency + self.learnablePhaseShift).sin().pow(2) + 1 - self.amplitude/2
        mul_38: "f32[129]" = torch.ops.aten.mul.Tensor(mul_21, primals_28);  primals_28 = None
        add_24: "f32[129]" = torch.ops.aten.add.Tensor(mul_38, primals_29);  mul_38 = primals_29 = None
        sin_6: "f32[129]" = torch.ops.aten.sin.default(add_24);  add_24 = None
        pow_7: "f32[129]" = torch.ops.aten.pow.Tensor_Scalar(sin_6, 2);  sin_6 = None
        mul_39: "f32[129]" = torch.ops.aten.mul.Tensor(pow_7, 0.1);  pow_7 = None
        add_25: "f32[129]" = torch.ops.aten.add.Tensor(mul_39, 1);  mul_39 = None
        sub_10: "f32[129]" = torch.ops.aten.sub.Tensor(add_25, 0.05);  add_25 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:227 in forwardPass, code: return x * nonLinearityTerm
        mul_40: "f64[12, 6, 129]" = torch.ops.aten.mul.Tensor(div_6, sub_10);  div_6 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleDualLinearLayer.py:91 in applyLayer, code: y2 = (x2 - torch.einsum('bns,nsi->bni', x1, A2)) / self.stabilityFactor
        unsqueeze_14: "f64[12, 6, 129, 1]" = torch.ops.aten.unsqueeze.default(mul_40, 3)
        permute_35: "f64[12, 6, 1, 129]" = torch.ops.aten.permute.default(unsqueeze_14, [0, 1, 3, 2]);  unsqueeze_14 = None
        unsqueeze_15: "f64[6, 129, 129, 1]" = torch.ops.aten.unsqueeze.default(primals_31, 3);  primals_31 = None
        permute_36: "f64[1, 6, 129, 129]" = torch.ops.aten.permute.default(unsqueeze_15, [3, 0, 2, 1]);  unsqueeze_15 = None
        permute_37: "f64[6, 12, 129, 1]" = torch.ops.aten.permute.default(permute_35, [1, 0, 3, 2]);  permute_35 = None
        view_28: "f64[6, 12, 129]" = torch.ops.aten.reshape.default(permute_37, [6, 12, 129]);  permute_37 = None
        permute_38: "f64[6, 129, 129, 1]" = torch.ops.aten.permute.default(permute_36, [1, 3, 2, 0]);  permute_36 = None
        view_29: "f64[6, 129, 129]" = torch.ops.aten.reshape.default(permute_38, [6, 129, 129]);  permute_38 = None
        bmm_7: "f64[6, 12, 129]" = torch.ops.aten.bmm.default(view_28, view_29);  view_28 = None
        view_30: "f64[6, 12, 1, 129]" = torch.ops.aten.reshape.default(bmm_7, [6, 12, 1, 129]);  bmm_7 = None
        permute_39: "f64[12, 6, 129, 1]" = torch.ops.aten.permute.default(view_30, [1, 0, 3, 2]);  view_30 = None
        view_31: "f64[12, 6, 129]" = torch.ops.aten.reshape.default(permute_39, [12, 6, 129]);  permute_39 = None
        sub_11: "f64[12, 6, 129]" = torch.ops.aten.sub.Tensor(mul_34, view_31);  mul_34 = view_31 = None
        div_8: "f64[12, 6, 129]" = torch.ops.aten.div.Tensor(sub_11, 1.15);  sub_11 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleDualLinearLayer.py:92 in applyLayer, code: y1 = (x1 - torch.einsum('bns,nsi->bni', y2, A1)) / self.stabilityFactor
        unsqueeze_16: "f64[12, 6, 129, 1]" = torch.ops.aten.unsqueeze.default(div_8, 3)
        permute_40: "f64[12, 6, 1, 129]" = torch.ops.aten.permute.default(unsqueeze_16, [0, 1, 3, 2]);  unsqueeze_16 = None
        unsqueeze_17: "f64[6, 129, 129, 1]" = torch.ops.aten.unsqueeze.default(primals_30, 3);  primals_30 = None
        permute_41: "f64[1, 6, 129, 129]" = torch.ops.aten.permute.default(unsqueeze_17, [3, 0, 2, 1]);  unsqueeze_17 = None
        permute_42: "f64[6, 12, 129, 1]" = torch.ops.aten.permute.default(permute_40, [1, 0, 3, 2]);  permute_40 = None
        view_32: "f64[6, 12, 129]" = torch.ops.aten.reshape.default(permute_42, [6, 12, 129]);  permute_42 = None
        permute_43: "f64[6, 129, 129, 1]" = torch.ops.aten.permute.default(permute_41, [1, 3, 2, 0]);  permute_41 = None
        view_33: "f64[6, 129, 129]" = torch.ops.aten.reshape.default(permute_43, [6, 129, 129]);  permute_43 = None
        bmm_8: "f64[6, 12, 129]" = torch.ops.aten.bmm.default(view_32, view_33);  view_32 = None
        view_34: "f64[6, 12, 1, 129]" = torch.ops.aten.reshape.default(bmm_8, [6, 12, 1, 129]);  bmm_8 = None
        permute_44: "f64[12, 6, 129, 1]" = torch.ops.aten.permute.default(view_34, [1, 0, 3, 2]);  view_34 = None
        view_35: "f64[12, 6, 129]" = torch.ops.aten.reshape.default(permute_44, [12, 6, 129]);  permute_44 = None
        sub_12: "f64[12, 6, 129]" = torch.ops.aten.sub.Tensor(mul_40, view_35);  mul_40 = view_35 = None
        div_9: "f64[12, 6, 129]" = torch.ops.aten.div.Tensor(sub_12, 1.15);  sub_12 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:236 in getNonLinearity, code: return self.amplitude*(positions*2*torch.pi*self.learnableFrequency + self.learnablePhaseShift).sin().pow(2) + 1 - self.amplitude/2
        mul_44: "f32[129]" = torch.ops.aten.mul.Tensor(mul_21, primals_32);  primals_32 = None
        add_27: "f32[129]" = torch.ops.aten.add.Tensor(mul_44, primals_33);  mul_44 = primals_33 = None
        sin_7: "f32[129]" = torch.ops.aten.sin.default(add_27);  add_27 = None
        pow_8: "f32[129]" = torch.ops.aten.pow.Tensor_Scalar(sin_7, 2);  sin_7 = None
        mul_45: "f32[129]" = torch.ops.aten.mul.Tensor(pow_8, 0.1);  pow_8 = None
        add_28: "f32[129]" = torch.ops.aten.add.Tensor(mul_45, 1);  mul_45 = None
        sub_13: "f32[129]" = torch.ops.aten.sub.Tensor(add_28, 0.05);  add_28 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:231 in inversePass, code: return y / nonLinearityTerm
        div_10: "f64[12, 6, 129]" = torch.ops.aten.div.Tensor(div_8, sub_13);  div_8 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:236 in getNonLinearity, code: return self.amplitude*(positions*2*torch.pi*self.learnableFrequency + self.learnablePhaseShift).sin().pow(2) + 1 - self.amplitude/2
        mul_49: "f32[129]" = torch.ops.aten.mul.Tensor(mul_21, primals_34);  mul_21 = primals_34 = None
        add_30: "f32[129]" = torch.ops.aten.add.Tensor(mul_49, primals_35);  mul_49 = primals_35 = None
        sin_8: "f32[129]" = torch.ops.aten.sin.default(add_30);  add_30 = None
        pow_9: "f32[129]" = torch.ops.aten.pow.Tensor_Scalar(sin_8, 2);  sin_8 = None
        mul_50: "f32[129]" = torch.ops.aten.mul.Tensor(pow_9, 0.1);  pow_9 = None
        add_31: "f32[129]" = torch.ops.aten.add.Tensor(mul_50, 1);  mul_50 = None
        sub_14: "f32[129]" = torch.ops.aten.sub.Tensor(add_31, 0.05);  add_31 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:231 in inversePass, code: return y / nonLinearityTerm
        div_11: "f64[12, 6, 129]" = torch.ops.aten.div.Tensor(div_9, sub_14);  div_9 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\neuralOperators\fourierOperator\fourierNeuralOperatorWeights.py:80 in backwardFFT, code: fourierData = realFourierData + 1j * imaginaryFourierData
        mul_51: "c128[12, 6, 129]" = torch.ops.aten.mul.Tensor(div_11, 1j);  div_11 = None
        add_32: "c128[12, 6, 129]" = torch.ops.aten.add.Tensor(div_10, mul_51);  div_10 = mul_51 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\neuralOperators\fourierOperator\fourierNeuralOperatorWeights.py:84 in backwardFFT, code: if resampledTimes is None: return torch.fft.irfft(fourierData, n=self.sequenceLength, dim=-1, norm='ortho')
        _fft_c2r: "f64[12, 6, 256]" = torch.ops.aten._fft_c2r.default(add_32, [2], 1, 256);  add_32 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:236 in getNonLinearity, code: return self.amplitude*(positions*2*torch.pi*self.learnableFrequency + self.learnablePhaseShift).sin().pow(2) + 1 - self.amplitude/2
        mul_55: "f32[256]" = torch.ops.aten.mul.Tensor(mul_3, primals_36);  mul_3 = primals_36 = None
        add_34: "f32[256]" = torch.ops.aten.add.Tensor(mul_55, primals_37);  mul_55 = primals_37 = None
        sin_9: "f32[256]" = torch.ops.aten.sin.default(add_34);  add_34 = None
        pow_10: "f32[256]" = torch.ops.aten.pow.Tensor_Scalar(sin_9, 2);  sin_9 = None
        mul_56: "f32[256]" = torch.ops.aten.mul.Tensor(pow_10, 0.1);  pow_10 = None
        add_35: "f32[256]" = torch.ops.aten.add.Tensor(mul_56, 1);  mul_56 = None
        sub_15: "f32[256]" = torch.ops.aten.sub.Tensor(add_35, 0.05);  add_35 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:231 in inversePass, code: return y / nonLinearityTerm
        div_12: "f64[12, 6, 256]" = torch.ops.aten.div.Tensor(_fft_c2r, sub_15);  _fft_c2r = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleDualLinearLayer.py:92 in applyLayer, code: y1 = (x1 - torch.einsum('bns,nsi->bni', y2, A1)) / self.stabilityFactor
        permute_46: "f64[6, 129, 129]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleDualLinearLayer.py:91 in applyLayer, code: y2 = (x2 - torch.einsum('bns,nsi->bni', x1, A2)) / self.stabilityFactor
        permute_50: "f64[6, 129, 129]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleDualLinearLayer.py:92 in applyLayer, code: y1 = (x1 - torch.einsum('bns,nsi->bni', y2, A1)) / self.stabilityFactor
        permute_54: "f64[6, 129, 129]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleDualLinearLayer.py:91 in applyLayer, code: y2 = (x2 - torch.einsum('bns,nsi->bni', x1, A2)) / self.stabilityFactor
        permute_58: "f64[6, 129, 129]" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleDualLinearLayer.py:92 in applyLayer, code: y1 = (x1 - torch.einsum('bns,nsi->bni', y2, A1)) / self.stabilityFactor
        permute_62: "f64[6, 129, 129]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleDualLinearLayer.py:91 in applyLayer, code: y2 = (x2 - torch.einsum('bns,nsi->bni', x1, A2)) / self.stabilityFactor
        permute_66: "f64[6, 129, 129]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:89 in applyLayer, code: outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)
        permute_70: "f64[6, 256, 256]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_74: "f64[6, 256, 256]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
        permute_78: "f64[6, 256, 256]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return (div_12, primals_3, primals_1, primals_17, primals_9, primals_10, primals_20, primals_21, primals_22, primals_23, sub_1, sub_2, sub_9, sub_10, sub_13, sub_14, sub_15, permute_46, permute_50, permute_54, permute_58, permute_62, permute_66, permute_70, permute_74, permute_78)
        