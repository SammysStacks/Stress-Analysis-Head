
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import _globalPytorchModel

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class GRU(_globalPytorchModel.globalModel):
    def __init__(self, output_size, numSignals):
        super(GRU, self).__init__()
        # Specify GRU model parameters
        self.input_size = numSignals # The number of signals going into the RNN.
        self.bidirectional = False   # Go forward and backward through the time-series data.
        self.batch_first = True      # If True: the first dimension is the batch index: (batchInd, pointInd, signalInd)
        self.hidden_size = 8         # The number of outputs from one RNN node.
        self.num_layers = 5          # The number of sequential RNN nodes.
        self.dropout = 0.4           # Probability of dropping weights; On the outputs of each GRU layer except the last.
        self.bias = True             # Introduce a shift or offset in the activation of each unit or hidden state.
        # Specify general model parameters
        self.output_size = output_size   # For classification this is numClasses. For regression this is 1.
        self.numDirections = self.bidirectional + 1
        self.useAllHiddenLayers = False
        
        if self.useAllHiddenLayers:
            inputANN = self.hidden_size*self.num_layers
        else:
            inputANN = self.hidden_size
            
        # Apply batch normalization to the input.
        self.bn0 = nn.BatchNorm1d(num_features = numSignals)
        # Create an instance of the RNN model
        self.rnn = nn.GRU(
            bidirectional=self.bidirectional,
            hidden_size=self.hidden_size,
            batch_first=self.batch_first,
            num_layers=self.num_layers,
            input_size=self.input_size,
            dropout=self.dropout,
            bias=self.bias,
        )
        # Create the fully connected layers.
        self.fc1 = nn.Linear(inputANN, 32, bias = True)
        self.bn1 = nn.BatchNorm1d(32)  # Batch normalization after the first linear layer
        self.fc2 = nn.Linear(inputANN, 32, bias = True)
        self.bn2 = nn.BatchNorm1d(32)  # Batch normalization after the first linear layer
        self.fc3 = nn.Linear(32, self.output_size, bias = True)

    def forward(self, inputData):
        # Compile the model's features from the RNN
        signalFeatures = self.compileFeatures(inputData)
        
        # Process the features into a prediction
        finalPrediction = self.modelSpecificPrediction(signalFeatures)
        return finalPrediction
    
    def compileFeatures(self, inputData):
        """ If batch_first: the shape of inputData = (numBatches, numTimePoints, numSignals)"""   
        # # Apply batch normalization
        inputData = inputData.permute(0, 2, 1) # Reshape to (numBatches, numSignals, numTimePoints)
        inputData = self.bn0(inputData.float())
        inputData = inputData.permute(0, 2, 1) # Reshape to (numBatches, numSignals, numTimePoints)
        
        # Initialize the hidden state and cell state
        batch_size = inputData.shape[0] if self.batch_first else inputData.shape[1] # Get the batch size of the input data.
        initialHiddenState = torch.zeros(self.num_layers*self.numDirections, batch_size, self.hidden_size)  # Shape: (num_layers*num_directions, batch_size, hidden_size)
        # Put the data through the GRU network.  
        outputSequence, finalHiddenStates = self.rnn(inputData.to(torch.float32), initialHiddenState)     
        # outputSequence: (batch_size, seqlength, hidden_size * numDirections)
        # finalHiddenStates: (num_layers * numDirections, batch_size, hidden_size or proj_size)
                        
        # Get the last hidden state of the GRU.
        if self.useAllHiddenLayers:
            lastHiddenState = finalHiddenStates.transpose(0, 1).reshape(batch_size, -1) # Shape: (batch_size, (hidden_size or proj_size) * num_layers * numDirections)
        else:
            lastHiddenState = finalHiddenStates[-1] # Shape: (batch_size, hidden_size or proj_size)
        
        # Pass the data through the neural architecture: Layer 1.
        output = self.fc1(lastHiddenState)
        output = self.bn1(output)  # Apply batch normalization
        output = F.relu(output)
        # Pass the data through the neural architecture: Layer 2.
        output = self.fc2(output)
        output = self.bn2(output)  # Apply batch normalization
        output = F.relu(output)
            
        return output
    
    def modelSpecificPrediction(self, signalFeatures):
        assert False
        # Pass the data through the neural architecture: Layer 1.
        output = self.fc1(signalFeatures)
        output = self.bn1(output)  # Apply batch normalization
        output = F.relu(output)
        # Pass the data through the neural architecture: Layer 2.
        output = self.fc2(output)
        output = self.bn2(output)  # Apply batch normalization
        output = F.relu(output)
        # Pass the data through the neural architecture: Layer 3.
        output = self.fc3(output)
        
        # For classification problems
        if self.applyLogSoftmax:
            output = F.log_softmax(output, dim=1)  # Apply log-softmax activation to get class probabilities.
        elif self.applySoftmax:
            output = F.softmax(output, dim=1)  # Apply log-softmax activation to get class probabilities.

        return output
        
    
    
    