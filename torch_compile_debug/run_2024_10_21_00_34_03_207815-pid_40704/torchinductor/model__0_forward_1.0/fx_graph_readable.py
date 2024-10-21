class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[1, 129, 129]", primals_2: "f64[1, 129, 129]", primals_3: "f64[129, 129]", primals_4: "f64[3080, 1, 129]"):
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleLinearLayer.py:57 in applyLayer, code: if self.kernelSize != self.sequenceLength: neuralWeights = self.restrictedWindowMask * neuralWeights + self.stabilityTerm
        mul: "f64[1, 129, 129]" = torch.ops.aten.mul.Tensor(primals_2, primals_1);  primals_1 = None
        add: "f64[1, 129, 129]" = torch.ops.aten.add.Tensor(mul, primals_3);  mul = primals_3 = None
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\modelComponents\reversibleComponents\reversibleLinearLayer.py:64 in applyLayer, code: outputData = torch.einsum('bns,nsi->bni', inputData, neuralWeights)
        unsqueeze: "f64[3080, 1, 129, 1]" = torch.ops.aten.unsqueeze.default(primals_4, 3);  primals_4 = None
        permute: "f64[3080, 1, 1, 129]" = torch.ops.aten.permute.default(unsqueeze, [0, 1, 3, 2]);  unsqueeze = None
        unsqueeze_1: "f64[1, 129, 129, 1]" = torch.ops.aten.unsqueeze.default(add, 3);  add = None
        permute_1: "f64[1, 1, 129, 129]" = torch.ops.aten.permute.default(unsqueeze_1, [3, 0, 2, 1]);  unsqueeze_1 = None
        permute_2: "f64[3080, 129, 1, 1]" = torch.ops.aten.permute.default(permute, [0, 3, 1, 2]);  permute = None
        view: "f64[1, 3080, 129]" = torch.ops.aten.view.default(permute_2, [1, 3080, 129]);  permute_2 = None
        permute_3: "f64[129, 1, 129, 1]" = torch.ops.aten.permute.default(permute_1, [3, 1, 2, 0]);  permute_1 = None
        view_1: "f64[1, 129, 129]" = torch.ops.aten.view.default(permute_3, [1, 129, 129]);  permute_3 = None
        bmm: "f64[1, 3080, 129]" = torch.ops.aten.bmm.default(view, view_1)
        view_2: "f64[3080, 1, 1, 129]" = torch.ops.aten.view.default(bmm, [3080, 1, 1, 129]);  bmm = None
        permute_4: "f64[3080, 1, 129, 1]" = torch.ops.aten.permute.default(view_2, [0, 2, 3, 1]);  view_2 = None
        view_3: "f64[3080, 1, 129]" = torch.ops.aten.view.default(permute_4, [3080, 1, 129]);  permute_4 = None
        permute_6: "f64[1, 129, 3080]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        permute_7: "f64[1, 129, 129]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        return [view_3, primals_2, permute_6, permute_7]
        