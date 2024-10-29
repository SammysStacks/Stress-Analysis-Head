class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[12, 6, 256]", primals_2: "f32[256]"):
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:231 in inversePass, code: return y / nonLinearityTerm
        div: "f64[12, 6, 256]" = torch.ops.aten.div.Tensor(primals_1, primals_2);  primals_1 = None
        return (div, primals_2)
        