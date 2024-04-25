
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class ANN(nn.Module):
    def __init__(self, output_size, input_size):
        super(ANN, self).__init__()
        # Specify general model parameters
        self.output_size = output_size  # For classification this is numClasses. For regression this is 1.
        self.input_size = input_size    # The number of signals going into the RNN.
        self.bias = False               # Introduce a shift or offset in the activation of each unit or hidden state.
        
        self.fc1 = nn.Linear(self.input_size, 64, bias = False)
        self.bn1 = nn.BatchNorm1d(64)  # Batch normalization after the first linear layer
        self.fc2 = nn.Linear(64, 64, bias = False)
        self.bn2 = nn.BatchNorm1d(64)  # Batch normalization after the second linear layer
        self.fc3 = nn.Linear(64, self.output_size, bias = True)

    def forward(self, inputData):
        """ If batch_first: the shape of inputData = (numBatches, numTimesteps, numSignals)"""
        # If only ANN
        lastHiddenState = inputData.float().squeeze()
                
        # Pass the data through the neural architecture: Layer 1.
        output = self.fc1(lastHiddenState)
        output = self.bn1(output)  # Apply batch normalization
        output = F.relu(output)
        # Pass the data through the neural architecture: Layer 2.
        output = self.fc2(output)
        output = self.bn2(output)  # Apply batch normalization
        output = F.relu(output)
        # Pass the data through the neural architecture: Layer 4.
        output = self.fc3(output)
        
        # For classification problems
        output = F.log_softmax(output, dim=1)  # Apply log-softmax activation to get class probabilities.

        return output
    
    
    