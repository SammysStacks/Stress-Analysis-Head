
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import files
import pytorchGRU

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class compiledGRUs(pytorchGRU.GRU):
    def __init__(self, output_size, numSignals, metaTraining = True):
        super(compiledGRUs, self).__init__(output_size, numSignals = 1)
        # Specify general model parameters
        self.metaTraining = metaTraining
        self.output_size = output_size   # For classification this is numClasses. For regression this is 1.

        numFeatures_perSignal = 4
        compiledFeatureSize = numFeatures_perSignal*numSignals
        # Calculate the final number of features.        
        if self.useAllHiddenLayers:
            outputSizeRNN = self.hidden_size*self.num_layers
        else:
            outputSizeRNN = self.hidden_size
            
        # Create the fully connected layers.
        self.fc1 = nn.Linear(outputSizeRNN, 64, bias = True)
        self.bn1 = nn.BatchNorm1d(64)  # Batch normalization after the first linear layer
        self.fc2 = nn.Linear(64, numFeatures_perSignal, bias = True)
        self.bn2 = nn.BatchNorm1d(numFeatures_perSignal)  # Batch normalization after the second linear layer
        # Model Specific Architecture.
        self.fc3 = nn.Linear(compiledFeatureSize, self.output_size, bias = True)

    def forward(self, inputData):
        """ If batch_first: the shape of inputData = (numBatches, numTimePoints, numSignals)"""   
        if self.metaTraining:
            compiledFeatures = self.featureExtraction(inputData)
        else:
            compiledFeatures = inputData
        
        # Process the features into a prediction
        finalPrediction = self.modelSpecificPrediction(compiledFeatures)
        return finalPrediction    
    
    def featureExtraction(self, inputData):
        # Seperate out indivisual signals.
        batchSize, sequenceLength, numSignals = inputData.size()
        signalData = inputData.transpose(1, 2).reshape(batchSize * numSignals, sequenceLength, -1)
        # The new dimension: batchSize*numSignals, sequenceLength, 1
        
        # Pass the signal into the trained GRU
        compiledFeatures = self.compileFeatures(signalData)
        # Reshape the features into the correct batches.
        compiledFeatures = compiledFeatures.view(batchSize, -1)
        
        return compiledFeatures
    
    def modelSpecificPrediction(self, signalFeatures):
        # Pass the data through the neural architecture: Layer 3.
        output = self.fc3(signalFeatures.to(torch.float32))
        
        # For classification problems
        if self.applyLogSoftmax:
            output = F.log_softmax(output, dim=1)  # Apply log-softmax activation to get class probabilities.
        elif self.applySoftmax:
            output = F.softmax(output, dim=1)  # Apply log-softmax activation to get class probabilities.

        return output
    