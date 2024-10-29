class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f64[12, 6, 120, 2]", arg1_1: "i32[12, 6, 3]", arg2_1: "f64[12, 2]"):
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHead.py:156 in forward, code: basicEmotionProfile = torch.zeros((batchSize, self.numBasicEmotions, self.encodedDimension), device=device)
        full_default: "f64[12, 6, 256]" = torch.ops.aten.full.default([12, 6, 256], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHead.py:157 in forward, code: emotionProfile = torch.zeros((batchSize, self.numEmotions, self.encodedDimension), device=device)
        full_default_1: "f64[12, 32, 256]" = torch.ops.aten.full.default([12, 32, 256], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHead.py:158 in forward, code: activityProfile = torch.zeros((batchSize, self.encodedDimension), device=device)
        full_default_2: "f64[12, 256]" = torch.ops.aten.full.default([12, 256], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        return (arg1_1, arg0_1, arg2_1, full_default, full_default_1, full_default_2)
        