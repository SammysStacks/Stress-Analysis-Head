import torch
from torch import nn

from .waveletNeuralOperatorWeights import waveletNeuralOperatorWeights


class waveletNeuralOperatorLayer(waveletNeuralOperatorWeights):

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, numLayers, numDecompositions, waveletType, mode, addBiasTerm, activationMethod,
                 skipConnectionProtocol, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', learningProtocol='CNN'):
        super(waveletNeuralOperatorLayer, self).__init__(sequenceLength=sequenceLength, numInputSignals=numInputSignals, numOutputSignals=numOutputSignals, numLayers=numLayers, numDecompositions=numDecompositions,
                                                         waveletType=waveletType, mode=mode, addBiasTerm=addBiasTerm, activationMethod=activationMethod, skipConnectionProtocol=skipConnectionProtocol,
                                                         encodeLowFrequencyProtocol=encodeLowFrequencyProtocol, encodeHighFrequencyProtocol=encodeHighFrequencyProtocol, learningProtocol=learningProtocol)

    def forward(self, neuralOperatorData, residualLowFrequencyTerms=None, residualHighFrequencyTerms=None):
        # Apply the wavelet neural operator and the skip connection.
        neuralOperatorData = self.waveletNeuralOperator(neuralOperatorData, residualLowFrequencyTerms, residualHighFrequencyTerms)
        if self.skipConnectionModel is not None: neuralOperatorData = neuralOperatorData + self.skipConnectionModel(neuralOperatorData)  # Not reversible.
        # neuralOperatorData dimension: batchSize, numOutputSignals, finalLength

        # Apply the activation function.
        if self.activationMethod != 'none': neuralOperatorData = self.activationFunction(neuralOperatorData)
        # neuralOperatorData dimension: batchSize, numOutputSignals, finalLength

        return neuralOperatorData

    def waveletNeuralOperator(self, inputData, residualLowFrequencyTerms=None, residualHighFrequencyTerms=None):
        # Perform wavelet decomposition.
        lowFrequency, highFrequencies = self.dwt(inputData)  # Note: each channel is treated independently here.
        # highFrequencies[decompositionLayer] dimension: batchSize, numLiftedChannels, highFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numLiftedChannels, lowFrequencyShape

        # Encode each frequency decomposition, separating high and low frequencies.
        lowFrequency, highFrequencies = self.independentFrequencyAnalysis(lowFrequency, highFrequencies, residualLowFrequencyTerms, residualHighFrequencyTerms)
        # highFrequencies[highFrequencyInd] dimension: batchSize, numOutputSignals, highFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numOutputSignals, lowFrequencyShape

        # Perform wavelet reconstruction.
        reconstructedData = self.idwt((lowFrequency, highFrequencies))
        # reconstructedData dimension: batchSize, numOutputSignals, sequenceLength

        # Add a bias term if needed.
        if self.addBiasTerm: reconstructedData = reconstructedData + self.operatorBiases
        # reconstructedData dimension: batchSize, numOutputSignals, sequenceLength

        return reconstructedData

    def independentFrequencyAnalysis(self, lowFrequency, highFrequencies, residualLowFrequencyTerms=None, residualHighFrequencyTerms=None):
        # Set up the equation to apply the weights.
        equationString = 'oin,bin->bon'  # The equation to apply the weights.
        # b = batchSize, i = numLiftedChannels, o = numOutputSignals, n = signalDimension
        # 'oin,bin->bon' = weights.size(), frequencies.size() -> frequencies.size()

        # For each set of high-frequency coefficients.
        for highFrequencyInd in range(len(highFrequencies)):
            # Learn a new set of wavelet coefficients to transform the data.
            highFrequencies[highFrequencyInd] = self.applyEncoding(equationString, highFrequencies[highFrequencyInd], self.highFrequenciesWeights[highFrequencyInd], residualHighFrequencyTerms)
            # highFrequencies[highFrequencyInd] dimension: batchSize, numOutputSignals, highFrequenciesShapes[decompositionLayer]

        # Learn a new set of wavelet coefficients to transform the data.
        lowFrequency = self.applyEncoding(equationString, lowFrequency, self.lowFrequencyWeights, residualLowFrequencyTerms)
        # lowFrequency dimension: batchSize, numOutputSignals, lowFrequencyShape

        return lowFrequency, highFrequencies

    def applyEncoding(self, equationString, frequencies, weights, frequencyTerms=None):
        # Add in the learned wavelet coefficients.
        if frequencyTerms is not None: frequencies = frequencies + frequencyTerms
        # frequencies dimension: batchSize, numLiftedChannels, frequencyDimension

        if weights is not None:
            if isinstance(weights, nn.Module):
                frequencies = weights(frequencies)  # Learn a new set of wavelet coefficients to transform the data.
            if 'FC' in self.learningProtocol or 'CNN' in self.learningProtocol:
                frequencies = weights(frequencies)  # Learn a new set of wavelet coefficients to transform the data.
                # frequencies dimension: batchSize, numOutputSignals, frequencyDimension
            else:
                # Learn a new set of wavelet coefficients to transform the data.
                frequencies = torch.einsum(equationString, weights, frequencies)
                # frequencies dimension: batchSize, numOutputSignals, frequencyDimension

        return frequencies
