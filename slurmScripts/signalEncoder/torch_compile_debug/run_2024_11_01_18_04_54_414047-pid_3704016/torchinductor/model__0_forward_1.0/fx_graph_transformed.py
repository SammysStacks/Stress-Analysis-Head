class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[10, 64, 120, 2]", primals_2: "f32[10, 64, 3]", primals_3: "f32[10, 2]", primals_4: "f64[60, 128]"):
         # File: /central/groups/GaoGroup/ssolomon/Stress-Analysis-Head/helperFiles/machineLearning/modelControl/Models/pyTorch/emotionModelInterface/emotionModel/emotionModelHead.py:155 in forward, code: signalIdentifiers, signalData, metadata = signalIdentifiers.int(), signalData.double(), metadata.int()
        convert_element_type: "i32[10, 64, 3]" = torch.ops.prims.convert_element_type.default(primals_2, torch.int32);  primals_2 = None
        convert_element_type_1: "i32[10, 2]" = torch.ops.prims.convert_element_type.default(primals_3, torch.int32);  primals_3 = None
        
         # File: /central/groups/GaoGroup/ssolomon/Stress-Analysis-Head/helperFiles/machineLearning/modelControl/Models/pyTorch/emotionModelInterface/emotionModel/emotionModelHead.py:160 in forward, code: basicEmotionProfile = torch.zeros((batchSize, self.numBasicEmotions, self.encodedDimension), device=device)
        full_default: "f64[10, 6, 128]" = torch.ops.aten.full.default([10, 6, 128], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /central/groups/GaoGroup/ssolomon/Stress-Analysis-Head/helperFiles/machineLearning/modelControl/Models/pyTorch/emotionModelInterface/emotionModel/emotionModelHead.py:161 in forward, code: emotionProfile = torch.zeros((batchSize, self.numEmotions, self.encodedDimension), device=device)
        full_default_1: "f64[10, 32, 128]" = torch.ops.aten.full.default([10, 32, 128], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /central/groups/GaoGroup/ssolomon/Stress-Analysis-Head/helperFiles/machineLearning/modelControl/Models/pyTorch/emotionModelInterface/emotionModel/emotionModelHead.py:162 in forward, code: activityProfile = torch.zeros((batchSize, self.encodedDimension), device=device)
        full_default_2: "f64[10, 128]" = torch.ops.aten.full.default([10, 128], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /central/groups/GaoGroup/ssolomon/Stress-Analysis-Head/helperFiles/machineLearning/modelControl/Models/pyTorch/emotionModelInterface/emotionModel/emotionModelHelpers/emotionDataInterface.py:151 in getSignalIdentifierData, code: return signalIdentifiers[:, :, channelInd]
        select: "i32[10, 64]" = torch.ops.aten.select.int(convert_element_type, 2, 0)
        
         # File: /central/groups/GaoGroup/ssolomon/Stress-Analysis-Head/helperFiles/machineLearning/modelControl/Models/pyTorch/emotionModelInterface/emotionModel/emotionModelHelpers/emotionDataInterface.py:122 in getValidDataMask, code: positionTensor = torch.arange(start=0, end=maxSequenceLength, step=1,  device=allSignalData.device).expand(batchSize, numSignals, maxSequenceLength)
        iota: "i64[120]" = torch.ops.prims.iota.default(120, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        expand: "i64[10, 64, 120]" = torch.ops.aten.expand.default(iota, [10, 64, 120]);  iota = None
        
         # File: /central/groups/GaoGroup/ssolomon/Stress-Analysis-Head/helperFiles/machineLearning/modelControl/Models/pyTorch/emotionModelInterface/emotionModel/emotionModelHelpers/emotionDataInterface.py:125 in getValidDataMask, code: validDataMask = positionTensor < allNumSignalPoints.unsqueeze(-1)
        unsqueeze: "i32[10, 64, 1]" = torch.ops.aten.unsqueeze.default(select, -1);  select = None
        lt: "b8[10, 64, 120]" = torch.ops.aten.lt.Tensor(expand, unsqueeze);  expand = unsqueeze = None
        
         # File: /central/groups/GaoGroup/ssolomon/Stress-Analysis-Head/helperFiles/machineLearning/modelControl/Models/pyTorch/emotionModelInterface/emotionModel/emotionModelHelpers/emotionDataInterface.py:151 in getSignalIdentifierData, code: return signalIdentifiers[:, :, channelInd]
        select_1: "i32[10, 64]" = torch.ops.aten.select.int(convert_element_type, 2, 2);  convert_element_type = None
        
         # File: /central/groups/GaoGroup/ssolomon/Stress-Analysis-Head/helperFiles/machineLearning/modelControl/Models/pyTorch/emotionModelInterface/emotionModel/emotionModelHead.py:174 in forward, code: batchInds = emotionDataInterface.getSignalIdentifierData(signalIdentifiers, channelName=modelConstants.batchIndexSI)[:, 0]  # Dim: batchSize
        select_2: "i32[10]" = torch.ops.aten.select.int(select_1, 1, 0);  select_1 = None
        
         # File: /central/groups/GaoGroup/ssolomon/Stress-Analysis-Head/helperFiles/machineLearning/modelControl/Models/pyTorch/emotionModelInterface/emotionModel/emotionModelHelpers/submodels/specificSignalEncoderModel.py:56 in getCurrentPhysiologicalProfile, code: return self.physiologicalProfileAnsatz[batchInds]
        index: "f64[10, 128]" = torch.ops.aten.index.Tensor(primals_4, [select_2]);  primals_4 = None
        return (primals_1, convert_element_type_1, full_default, full_default_1, full_default_2, lt, index, select_2)
        