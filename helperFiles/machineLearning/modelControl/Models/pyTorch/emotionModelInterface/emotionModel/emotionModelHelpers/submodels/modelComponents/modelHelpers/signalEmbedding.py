# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import time

# PyTorch
import torch.nn as nn
from torchsummary import summary

# Import helper models
import convolutionalHelpers

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #
 
class embeddingLayer(_convolutionalHelpers.convolutionalHelpers):
    def __init__(self, sequenceLength, embeddedLength):
        super(embeddingLayer, self).__init__()   
        # General shape parameters.
        self.embeddedLength = embeddedLength
        self.sequenceLength = sequenceLength
        
        # embeddingLayer notes:
        #   padding: the number of added values around the image borders. padding = dilation * (kernel_size - 1) // 2
        #   dilation: the number of indices skipped between kernel points.
        #   kernel_size: the number of indices within the sliding filter.
        #   stride: the number of indices skipped when sliding the filter.
        
        # ------------------------- Pooling Layers ------------------------- # 
        
        # Pooling layers.
        self.maxPooling = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, return_indices=False, ceil_mode=False)
        self.avgPooling = nn.AvgPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=False, count_include_pad=True)
        self.minPooling = self.minPooling(self.maxPooling)
        # Compile the pooling layers
        self.poolingLayers = [self.maxPooling, self.avgPooling, self.minPooling]
        
        # ------------------------ CNN Architecture ------------------------ # 

        # Embed and encode spatial features.
        self.embedSignalsCNN_1 = nn.Sequential(
            # Convolution architecture
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Embed and encode spatial features.
        self.embedSignalsCNN_2 = nn.Sequential(
            # Convolution architecture
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, dilation = 2, padding=2, padding_mode='reflect', groups=1, bias=True),
            nn.SELU(),
        )
        
        # Embed and encode spatial features.
        self.embedSignalsCNN_3 = nn.Sequential(
            # Convolution architecture
            nn.Conv1d(in_channels=6, out_channels=3, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, stride=1, dilation = 1, padding=1, padding_mode='reflect', groups=1, bias=True),
            nn.LayerNorm(int(self.sequenceLength/2), eps=1E-10, elementwise_affine=True, bias=True),
            nn.SELU(),
        )
        
        # ------------------------ CNN Architecture ------------------------ # 
        
        # Embed the number of data points.
        self.embedSignalsANN_1 = nn.Sequential( 
            # Neural architecture: Layer 1.
            nn.Linear(int(self.sequenceLength/2), int(self.sequenceLength/2)*2, bias = True),
            nn.SELU(),
            
            # Neural architecture: Layer 1.
            nn.Linear(int(self.sequenceLength/2)*2, self.embeddedLength, bias = True),
        )
        
        # ------------------------------------------------------------------ # 

    def forward(self, inputData):
        """ The shape of inputData: (batchSize, numSignals, finalDistributionLength) """
        # Specify the current input shape of the data.
        batchSize, numSignals, sequenceLength = inputData.size()
        assert self.sequenceLength == sequenceLength
        
        # Reshape the data to the expected input into the CNN architecture.
        signalData = inputData.view(batchSize * numSignals, 1, sequenceLength) # Seperate out indivisual signals.
        # signalChannel dimension: batchSize*numSignals, 1, finalDistributionLength
        
        # ------------------------ CNN Architecture ------------------------ # 
        
        # Increase the number of channels in the encocder.
        embeddedSignals_0 = self.embedSignalsCNN_1(signalData)
        # Add a residual connection to prevent loss of information.
        embeddedSignals_0 = embeddedSignals_0 + signalData
        # Apply a pooling layer to reduce the signal's dimension.
        embeddedSignals_1 = self.splitPooling(embeddedSignals_0, self.poolingLayers)
        # embeddedSignals_1 dimension: batchSize*numSignals, 6, finalDistributionLength/2
        
        # Apply the second CNN block to reduce spatial dimension.
        embeddedSignals_2 = self.embedSignalsCNN_2(embeddedSignals_1)
        embeddedSignals_2 = self.embedSignalsCNN_2(embeddedSignals_2)
        # Add a residual connection to prevent loss of information.
        embeddedSignals_2 = embeddedSignals_2 + embeddedSignals_1
        # Apply a pooling layer to reduce the signal's dimension.
        # embeddedSignals_2 = self.splitPooling(embeddedSignals_2, self.poolingLayers)
        # embeddedSignals_2 dimension: batchSize*numSignals, 6, finalDistributionLength/4
        
        # Apply the third CNN block to reduce spatial dimension.
        embeddedSignals_3 = self.embedSignalsCNN_3(embeddedSignals_2)
        # embeddedSignals_3 dimension: batchSize*numSignals, 1, finalDistributionLength/4

        # ------------------------ ANN Architecture ------------------------ # 
        
        # Synthesize the information in the embedded space.
        embeddedSignals = self.embedSignalsANN_1(embeddedSignals_3)
        # EmbeddedData dimension: batchSize*numSignals, 1, self.embeddedLength

        # Seperate put each signal into its respective batch.
        embeddedData = embeddedSignals.view(batchSize, numSignals, self.embeddedLength)
        # EmbeddedData dimension: batchSize, numSignals, self.embeddedLength

        # ------------------------------------------------------------------ # 
        
        return embeddedData
    
    def printParams(self, numSignals = 2):
        #embeddingLayer(finalDistributionLength = 64, embeddedLength = 16).printParams(numSignals = 2)
        t1 = time.time()
        summary(self, (numSignals, self.sequenceLength,))
        t2 = time.time()
        
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')
        print("Total time:", t2 - t1)
    
    