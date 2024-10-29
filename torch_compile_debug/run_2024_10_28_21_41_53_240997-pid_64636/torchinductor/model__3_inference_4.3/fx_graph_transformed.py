class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f64[12, 6, 120, 2]"):
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\emotionDataInterface.py:121 in getChannelData_fromInd, code: return signalData[:, :, :, channelInd]
        select: "f64[12, 6, 120]" = torch.ops.aten.select.int(arg0_1, 3, 0);  arg0_1 = None
        return (select,)
        