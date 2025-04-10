import torch
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.fourierOperator.fourierNeuralOperatorWeights import fourierNeuralOperatorWeights
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleLieLayer import reversibleLieLayer


class fourierNeuralOperatorLayer(fourierNeuralOperatorWeights):

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, numLayers, addBiasTerm, activationMethod, skipConnectionProtocol, encodeRealFrequencies, encodeImaginaryFrequencies, learningProtocol='rFC'):
        super(fourierNeuralOperatorLayer, self).__init__(sequenceLength, numInputSignals, numOutputSignals, numLayers, addBiasTerm, activationMethod, skipConnectionProtocol, encodeRealFrequencies, encodeImaginaryFrequencies, learningProtocol)

    def forward(self, neuralOperatorData, realFrequencyTerms=None, imaginaryFrequencyTerms=None):
        # Apply the wavelet neural operator and the skip connection.
        neuralOperatorData = self.fourierNeuralOperator(neuralOperatorData, realFrequencyTerms, imaginaryFrequencyTerms)
        if self.skipConnectionModel is not None: neuralOperatorData = neuralOperatorData + self.skipConnectionModel(neuralOperatorData)  # Not reversible.
        # neuralOperatorData dimension: batchSize, numOutputSignals, finalLength

        # Apply the activation function.
        if self.activationMethod != 'none': neuralOperatorData = self.activationFunction(neuralOperatorData)
        # neuralOperatorData dimension: batchSize, numOutputSignals, finalLength

        return neuralOperatorData

    def fourierNeuralOperator(self, inputData, realFrequencyTerms=None, imaginaryFrequencyTerms=None):
        # Extract the input data parameters.
        batchSize, numInputSignals, sequenceLength = inputData.size()
        left_pad = (self.sequenceLength - sequenceLength) // 2
        right_pad = self.sequenceLength - sequenceLength - left_pad

        # Apply padding to the input data
        inputData = torch.nn.functional.pad(inputData, pad=(left_pad, right_pad), mode='constant', value=0)
        # inputData dimension: batchSize, numInputSignals, paddedSequenceLength

        # Project the data into the Fourier domain.
        realFourierData, imaginaryFourierData = self.forwardFFT(inputData)
        # imaginaryFourierData: batchSize, numInputSignals, fourierDimension
        # realFourierData: batchSize, numInputSignals, fourierDimension

        # For each layer in the neural operator.
        for layerInd in range(self.numLayers):
            # Learn a new set of coefficients using both of the frequency data.
            if reversibleInterface.forwardDirection: layerInd = self.numLayers - layerInd - 1
            if self.encodeImaginaryFrequencies: imaginaryFourierData = self.applyEncoding(equationString='oin,bin->bon', layerInd=layerInd, frequencies=imaginaryFourierData, weights=self.imaginaryFourierWeights, frequencyTerms=imaginaryFrequencyTerms)
            if self.encodeRealFrequencies: realFourierData = self.applyEncoding(equationString='oin,bin->bon', layerInd=layerInd, frequencies=realFourierData, weights=self.realFourierWeights, frequencyTerms=realFrequencyTerms)
            # imaginaryFourierData: batchSize, numOutputChannels, fourierDimension
            # realFourierData: batchSize, numOutputChannels, fourierDimension

        # Return to physical space
        reconstructedData = self.backwardFFT(realFourierData, imaginaryFourierData, resampledTimes=None)[:, :, left_pad:left_pad + sequenceLength]
        # reconstructedData dimension: batchSize, numOutputChannels, signalDimension

        # Add a bias term if needed.
        if self.addBiasTerm: reconstructedData = reconstructedData + self.operatorBiases
        # reconstructedData dimension: batchSize, numOutputChannels, signalDimension

        return reconstructedData

    def applyEncoding(self, equationString, layerInd, frequencies, weights, frequencyTerms=None):
        # Add in the learned wavelet coefficients.
        if frequencyTerms is not None: frequencies = frequencies + frequencyTerms
        # frequencies dimension: batchSize, numLiftedChannels, frequencyDimension

        if weights is not None:
            # Learn a new set of wavelet coefficients to transform the data.
            if isinstance(weights, reversibleLieLayer): frequencies = weights.applySingleLayer(frequencies, layerInd=layerInd)
            elif isinstance(weights, nn.Identity): frequencies = weights(frequencies)
            elif isinstance(weights, nn.Module): frequencies = weights(frequencies)
            elif 'FC' in self.learningProtocol or 'CNN' in self.learningProtocol:  frequencies = weights(frequencies)
            else: frequencies = torch.einsum(equationString, weights, frequencies)
        # frequencies dimension: batchSize, numOutputSignals, frequencyDimension

        return frequencies
