# PyTorch
import timeit

import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
# Import files for machine learning
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.signalEncoderHelpers.signalEncoderModules import signalEncoderModules


class resamplingBridge(signalEncoderModules):

    def __init__(self, maxSequenceLength, finalSequenceLength, debuggingResults):
        super(resamplingBridge, self).__init__()
        # General parameters.
        self.finalSequenceLength = finalSequenceLength  # The number of final times. Type: int
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool

        # Initialize the model parameters.
        self.resamplingModel = self.resamplingOperator(inChannel=maxSequenceLength, outChannel=finalSequenceLength)  # The resampling model.
        self.finalTimeSeries = torch.linspace(start=0, end=1, steps=finalSequenceLength)  # The final time series.

    # ----------------------- Signal Pairing Methods ----------------------- #

    def forwardDirection(self, signalData):
        # signalData dimension: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
        batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()
        signalData = signalData.transpose(2, 3)  # Swaps dimensions at positions 2 and 3

        # Apply a resampling operation.
        transformedSignals = self.resamplingModel(signalData).transpose(2, 3)
        # transformedSignals dimension: batchSize, numSignals, finalSequenceLength, numChannels

        # Get the time series data.
        timeChannelData = emotionDataInterface.getChannelData(transformedSignals, modelConstants.timeChannel)
        finalTimeSeries = self.finalTimeSeries.expand(batchSize, numSignals, self.finalSequenceLength)
        # timeChannelData and finalTimeSeries dimension: batchSize, numSignals, finalSequenceLength

        # Add a correction term.
        correctionTerm = finalTimeSeries / timeChannelData
        transformedSignals = transformedSignals * correctionTerm.unsqueeze(-1)

        return transformedSignals

    def printParams(self):
        # Count the trainable parameters.
        numParams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {numParams} trainable parameters.')


if __name__ == '__main__':
    # Define the input parameters.
    debuggingResultsMain = False
    maxSequenceLengthMain = 1024
    numberIterations = 10

    # Initialize the signal data.
    signalDataMain = torch.randn(16, 128, maxSequenceLengthMain, 2)
    finalSequenceLengthMain = 64*modelConstants.maxTimeWindow

    # Initialize the class.
    resamplingBridgeClass = resamplingBridge(maxSequenceLength=maxSequenceLengthMain, finalSequenceLength=finalSequenceLengthMain, debuggingResults=debuggingResultsMain).to('cpu')

    # Perform a forward pass.
    def run_forward(): resamplingBridgeClass.forwardDirection(signalDataMain)
    execution_time = timeit.timeit(run_forward, number=numberIterations) / numberIterations
    print(f'Execution time: {execution_time:.6f} seconds')

    # Print the model parameters.
    resamplingBridgeClass.printParams()
