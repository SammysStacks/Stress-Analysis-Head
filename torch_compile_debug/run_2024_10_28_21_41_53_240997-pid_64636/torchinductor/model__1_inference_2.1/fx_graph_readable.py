class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i32[12, 6, 3]"):
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\emotionDataInterface.py:136 in getSignalIdentifierData, code: return signalIdentifiers[:, :, channelInd]
        select: "i32[12, 6]" = torch.ops.aten.select.int(arg0_1, 2, 2);  arg0_1 = None
        return (select,)
        