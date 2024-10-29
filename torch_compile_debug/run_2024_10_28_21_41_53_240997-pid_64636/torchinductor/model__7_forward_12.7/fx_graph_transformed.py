class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[6, 256, 256]", primals_2: "f64[6, 3]", primals_3: "i64[4596]", primals_4: "i64[4596]", primals_5: "i64[4596]", primals_6: "i64[4596]", primals_7: "f64[256, 256]", primals_8: "f64[12, 6, 256]"):
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:80 in applyLayer, code: neuralWeights[self.signalInds, self.rowInds, self.colInds] = kernelWeights[self.signalInds, self.kernelInds]
        index: "f64[4596]" = torch.ops.aten.index.Tensor(primals_2, [primals_3, primals_4]);  primals_2 = primals_4 = None
        index_put: "f64[6, 256, 256]" = torch.ops.aten.index_put.default(primals_1, [primals_3, primals_5, primals_6], index);  primals_1 = primals_3 = primals_5 = primals_6 = index = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:83 in applyLayer, code: neuralWeights = neuralWeights + self.stabilityTerm*0.975
        mul: "f64[256, 256]" = torch.ops.aten.mul.Tensor(primals_7, 0.975);  primals_7 = None
        add: "f64[6, 256, 256]" = torch.ops.aten.add.Tensor(index_put, mul);  index_put = mul = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleConvolutionLayer.py:89 in applyLayer, code: outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)
        unsqueeze: "f64[12, 6, 256, 1]" = torch.ops.aten.unsqueeze.default(primals_8, 3);  primals_8 = None
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
        permute_6: "f64[6, 256, 256]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return (view_3, permute_6)
        