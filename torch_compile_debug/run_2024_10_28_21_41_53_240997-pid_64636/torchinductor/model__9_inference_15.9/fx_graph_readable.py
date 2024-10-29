class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f64[]", arg1_1: "f64[]"):
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:234 in getNonLinearity, code: positions = torch.arange(start=0, end=self.sequenceLength, step=1, dtype=torch.float32, device=device)
        iota: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul: "i64[256]" = torch.ops.aten.mul.Tensor(iota, 1);  iota = None
        add: "i64[256]" = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        convert_element_type: "f32[256]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        
         # File: C:\Users\sasol.SQUIRTLE\Desktop\userData\Sam\projects\Stress-Analysis-Head\helperFiles\machineLearning\modelControl\Models\pyTorch\emotionModelInterface\emotionModel\emotionModelHelpers\optimizerMethods\activationFunctions.py:236 in getNonLinearity, code: return self.amplitude*(positions*2*torch.pi*self.learnableFrequency + self.learnablePhaseShift).sin().pow(2) + 1 - self.amplitude/2
        mul_1: "f32[256]" = torch.ops.aten.mul.Tensor(convert_element_type, 2);  convert_element_type = None
        mul_2: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1, 3.141592653589793);  mul_1 = None
        mul_3: "f32[256]" = torch.ops.aten.mul.Tensor(mul_2, arg0_1);  mul_2 = arg0_1 = None
        add_1: "f32[256]" = torch.ops.aten.add.Tensor(mul_3, arg1_1);  mul_3 = arg1_1 = None
        sin: "f32[256]" = torch.ops.aten.sin.default(add_1);  add_1 = None
        pow_1: "f32[256]" = torch.ops.aten.pow.Tensor_Scalar(sin, 2);  sin = None
        mul_4: "f32[256]" = torch.ops.aten.mul.Tensor(pow_1, 0.1);  pow_1 = None
        add_2: "f32[256]" = torch.ops.aten.add.Tensor(mul_4, 1);  mul_4 = None
        sub: "f32[256]" = torch.ops.aten.sub.Tensor(add_2, 0.05);  add_2 = None
        return (sub,)
        