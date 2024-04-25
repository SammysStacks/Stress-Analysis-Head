
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------- #
# --------------------------- Model Architecture --------------------------- #

class LSTM(nn.Module):
    def __init__(self, output_size, numSignals):
        super(LSTM, self).__init__()
        # Specify LSTM model parameters
        self.input_size = numSignals # The number of signals going into the RNN.
        self.bidirectional = False   # Go forward and backward through the time-series data.
        self.batch_first = True      # If True: the first dimension is the batch index: (batchInd, pointInd, signalInd)
        self.hidden_size = 20        # The number of outputs from one RNN node.
        self.num_layers = 10         # The number of sequential RNN nodes.
        self.proj_size = 0           # If > 0, will use LSTM with projections of corresponding size. Reduces the dimensionality of the hidden state.
        self.dropout = 0.1           # Probability of dropping weights; On the outputs of each LSTM layer except the last.
        self.bias = True             # Introduce a shift or offset in the activation of each unit or hidden state.
        # Specify general model parameters
        self.output_size = output_size   # For classification this is numClasses. For regression this is 1.
        
        # Create an instance of the RNN model
        self.rnn = nn.LSTM(
            bidirectional=self.bidirectional,
            hidden_size=self.hidden_size,
            batch_first=self.batch_first,
            num_layers=self.num_layers,
            input_size=self.input_size,
            proj_size=self.proj_size,
            dropout=self.dropout,
            bias=self.bias,
        )
        
        # Additional fully connected layer
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.bn1 = nn.BatchNorm1d(64)  # Batch normalization after the first linear layer
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)  # Batch normalization after the second linear layer
        self.fc3 = nn.Linear(32, self.output_size)

    def forward(self, inputData):
        """ If batch_first: the shape of inputData = (numBatches, numTimesteps, numSignals)"""
        # Initialize the hidden state and cell state
        batch_size = inputData.shape[0] if self.batch_first else inputData.shape[1] # Get the batch size of the input data.
        initialCellState = torch.zeros(self.num_layers*(self.bidirectional+1), batch_size, self.hidden_size)    # Shape: (num_layers*num_directions, batch_size, hidden_size)
        initialHiddenState = torch.zeros(self.num_layers*(self.bidirectional+1), batch_size, self.hidden_size)  # Shape: (num_layers*num_directions, batch_size, hidden_size)
        # Put the data through the LSTM network.        
        outputSequence, (finalHiddenStates, finalCellStates) = self.rnn(inputData.to(torch.float32), (initialHiddenState, initialCellState))     
        # outputSequence: (batch_size, seqlength, hidden_size (*2 if bidirectional)) 
        # finalCellStates: (num_layers (*2 if bidirectional), batch_size, hidden_size or proj_size)
        # finalHiddenStates: (num_layers (*2 if bidirectional), batch_size, hidden_size or proj_size)
        
        # Get the last hidden state of the LSTM.
        lastHiddenState = finalHiddenStates[-1] # Shape: (batch_size, hidden_size or proj_size)       
        # Pass the data through the neural architecture: Layer 1.
        output = self.fc1(lastHiddenState)
        output = self.bn1(output)  # Apply batch normalization
        output = F.relu(output)
        # Pass the data through the neural architecture: Layer 2.
        output = self.fc2(output)
        output = self.bn2(output)  # Apply batch normalization
        output = F.relu(output)
        # Pass the data through the neural architecture: Layer 3.
        output = self.fc3(output)
        output = F.relu(output)
        
        # For classification problems
        output = F.softmax(output, dim=1)  # Apply softmax activation to get class probabilities.

        return output

