class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[12, 6, 256]"):
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\neuralOperators\fourierOperator\fourierNeuralOperatorWeights.py:72 in forwardFFT, code: fourierData = torch.fft.rfft(inputData, n=self.sequenceLength, dim=-1, norm='ortho')
        _fft_r2c: "c128[12, 6, 129]" = torch.ops.aten._fft_r2c.default(primals_1, [2], 1, True);  primals_1 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\neuralOperators\fourierOperator\fourierNeuralOperatorWeights.py:73 in forwardFFT, code: imaginaryFourierData = fourierData.imag
        view_as_real: "f64[12, 6, 129, 2]" = torch.ops.aten.view_as_real.default(_fft_r2c)
        select: "f64[12, 6, 129]" = torch.ops.aten.select.int(view_as_real, 3, 1);  view_as_real = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\neuralOperators\fourierOperator\fourierNeuralOperatorWeights.py:74 in forwardFFT, code: realFourierData = fourierData.real
        view_as_real_1: "f64[12, 6, 129, 2]" = torch.ops.aten.view_as_real.default(_fft_r2c)
        select_1: "f64[12, 6, 129]" = torch.ops.aten.select.int(view_as_real_1, 3, 0);  view_as_real_1 = None
        return (select_1, select, _fft_r2c)
        