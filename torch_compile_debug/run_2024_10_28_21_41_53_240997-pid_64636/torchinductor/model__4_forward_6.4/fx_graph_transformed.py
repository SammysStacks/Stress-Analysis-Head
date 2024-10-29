class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[60, 256]", primals_2: "i32[12]"):
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\specificSignalEncoderModel.py:55 in getCurrentPhysiologicalProfile, code: return self.physiologicalProfileAnsatz[batchInds]
        index: "f64[12, 256]" = torch.ops.aten.index.Tensor(primals_1, [primals_2]);  primals_1 = None
        return (index, primals_2)
        