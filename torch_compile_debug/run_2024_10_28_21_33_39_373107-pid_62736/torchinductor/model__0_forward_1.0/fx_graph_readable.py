class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[12, 6, 120, 2]", primals_2: "i32[12, 6, 3]", primals_3: "f64[12, 2]", primals_4: "f64[60, 256]"):
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHead.py:156 in forward, code: basicEmotionProfile = torch.zeros((batchSize, self.numBasicEmotions, self.encodedDimension), device=device)
        full_default: "f64[12, 6, 256]" = torch.ops.aten.full.default([12, 6, 256], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHead.py:157 in forward, code: emotionProfile = torch.zeros((batchSize, self.numEmotions, self.encodedDimension), device=device)
        full_default_1: "f64[12, 32, 256]" = torch.ops.aten.full.default([12, 32, 256], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHead.py:158 in forward, code: activityProfile = torch.zeros((batchSize, self.encodedDimension), device=device)
        full_default_2: "f64[12, 256]" = torch.ops.aten.full.default([12, 256], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\emotionDataInterface.py:136 in getSignalIdentifierData, code: return signalIdentifiers[:, :, channelInd]
        select: "i32[12, 6]" = torch.ops.aten.select.int(primals_2, 2, 2);  primals_2 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHead.py:163 in forward, code: batchInds = emotionDataInterface.getSignalIdentifierData(signalIdentifiers, channelName=modelConstants.batchIndexSI)[:, 0]  # Dim: batchSize
        select_1: "i32[12]" = torch.ops.aten.select.int(select, 1, 0);  select = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\emotionDataInterface.py:121 in getChannelData_fromInd, code: return signalData[:, :, :, channelInd]
        select_2: "f64[12, 6, 120]" = torch.ops.aten.select.int(primals_1, 3, 1)
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\emotionDataInterface.py:121 in getChannelData_fromInd, code: return signalData[:, :, :, channelInd]
        select_3: "f64[12, 6, 120]" = torch.ops.aten.select.int(primals_1, 3, 0);  primals_1 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHead.py:170 in forward, code: missingDataMask = torch.as_tensor((datapoints == 0) & (timepoints == 0), device=device, dtype=torch.bool)
        eq: "b8[12, 6, 120]" = torch.ops.aten.eq.Scalar(select_2, 0)
        eq_1: "b8[12, 6, 120]" = torch.ops.aten.eq.Scalar(select_3, 0)
        bitwise_and: "b8[12, 6, 120]" = torch.ops.aten.bitwise_and.Tensor(eq, eq_1);  eq = eq_1 = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHead.py:171 in forward, code: validSignalMask = ~torch.all(missingDataMask, dim=-1).view(batchSize, numSignals, 1)
        logical_not: "b8[12, 6, 120]" = torch.ops.aten.logical_not.default(bitwise_and)
        any_1: "b8[12, 6]" = torch.ops.aten.any.dim(logical_not, -1);  logical_not = None
        logical_not_1: "b8[12, 6]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        view: "b8[12, 6, 1]" = torch.ops.aten.view.default(logical_not_1, [12, 6, 1]);  logical_not_1 = None
        bitwise_not: "b8[12, 6, 1]" = torch.ops.aten.bitwise_not.default(view);  view = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\submodels\specificSignalEncoderModel.py:55 in getCurrentPhysiologicalProfile, code: return self.physiologicalProfileAnsatz[batchInds]
        index: "f64[12, 256]" = torch.ops.aten.index.Tensor(primals_4, [select_1])
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHead.py:181 in forward, code: print(self.datasetName, self.specificSignalEncoderModel.physiologicalProfileAnsatz[0, :5].detach().cpu().numpy())
        select_4: "f64[256]" = torch.ops.aten.select.int(primals_4, 0, 0);  primals_4 = None
        slice_10: "f64[5]" = torch.ops.aten.slice.Tensor(select_4, 0, 0, 5);  select_4 = None
        device_put: "f64[5]" = torch.ops.prims.device_put.default(slice_10, device(type='cpu'));  slice_10 = None
        return (device_put, primals_3, full_default, full_default_1, full_default_2, select_2, select_3, bitwise_and, bitwise_not, index, select_1)
        