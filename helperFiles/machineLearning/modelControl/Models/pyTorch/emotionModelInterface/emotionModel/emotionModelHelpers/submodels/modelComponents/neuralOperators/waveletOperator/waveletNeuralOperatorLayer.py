# General
import torch

# Import machine learning files
from .waveletNeuralOperatorWeights import waveletNeuralOperatorWeights


class waveletNeuralOperatorLayer(waveletNeuralOperatorWeights):

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, numDecompositions, waveletType, mode, addBiasTerm, activationMethod,
                 skipConnectionProtocol, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', learningProtocol='CNN'):
        super(waveletNeuralOperatorLayer, self).__init__(sequenceLength, numInputSignals, numOutputSignals, numDecompositions, waveletType, mode, addBiasTerm, activationMethod,
                                                         skipConnectionProtocol, encodeLowFrequencyProtocol, encodeHighFrequencyProtocol, learningProtocol)

    def forward(self, inputData, lowFrequencyTerms=None, highFrequencyTerms=None):
        # Apply the wavelet neural operator and the skip connection.
        neuralOperatorOutput = self.waveletNeuralOperator(inputData, lowFrequencyTerms, highFrequencyTerms)
        neuralOperatorOutput = neuralOperatorOutput + self.skipConnectionModel(inputData)
        # neuralOperatorOutput dimension: batchSize, numOutputSignals, finalLength

        # Apply the activation function.
        neuralOperatorOutput = self.activationFunction(neuralOperatorOutput)
        # neuralOperatorOutput dimension: batchSize, numOutputSignals, finalLength

        return neuralOperatorOutput

    def waveletNeuralOperator(self, inputData, lowFrequencyTerms=None, highFrequencyTerms=None):
        # Perform wavelet decomposition.
        lowFrequency, highFrequencies = self.dwt(inputData)  # Note: each channel is treated independently here.
        # highFrequencies[decompositionLayer] dimension: batchSize, numLiftedChannels, highFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numLiftedChannels, lowFrequencyShape

        # Encode each frequency decomposition, separating high and low frequencies.
        lowFrequency, highFrequencies = self.independentFrequencyAnalysis(lowFrequency, highFrequencies, lowFrequencyTerms, highFrequencyTerms)
        # highFrequencies[highFrequencyInd] dimension: batchSize, numOutputSignals, highFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numOutputSignals, lowFrequencyShape

        # Perform wavelet reconstruction.
        reconstructedData = self.idwt((lowFrequency, highFrequencies))
        # reconstructedData dimension: batchSize, numOutputSignals, finalSequenceLength

        # Add a bias term if needed.
        if self.addBiasTerm: reconstructedData = reconstructedData + self.operatorBiases

        return reconstructedData

    def independentFrequencyAnalysis(self, lowFrequency, highFrequencies, lowFrequencyTerms=None, highFrequencyTerms=None):
        # Set up the equation to apply the weights.
        equationString = 'oin,bin->bon'  # The equation to apply the weights.
        # b = batchSize, i = numLiftedChannels, o = numOutputSignals, n = signalDimension
        # 'oin,bin->bon' = weights.size(), frequencies.size() -> frequencies.size()

        # For each set of high-frequency coefficients.
        for highFrequencyInd in range(len(highFrequencies)):
            # Learn a new set of wavelet coefficients to transform the data.
            highFrequencies[highFrequencyInd] = self.applyEncoding(equationString, highFrequencies[highFrequencyInd], self.highFrequenciesWeights[highFrequencyInd], highFrequencyTerms)
            # highFrequencies[highFrequencyInd] dimension: batchSize, numOutputSignals, highFrequenciesShapes[decompositionLayer]

        # Learn a new set of wavelet coefficients to transform the data.
        lowFrequency = self.applyEncoding(equationString, lowFrequency, self.lowFrequencyWeights, lowFrequencyTerms)
        # lowFrequency dimension: batchSize, numOutputSignals, lowFrequencyShape

        return lowFrequency, highFrequencies

    def applyEncoding(self, equationString, frequencies, weights, frequencyTerms=None):
        # Add in the learned wavelet coefficients.
        if frequencyTerms is not None: frequencies = frequencies + frequencyTerms
        # frequencies dimension: batchSize, numLiftedChannels, frequencyDimension

        if weights is not None:
            if self.learningProtocol in ['FC', 'CNN', "iCNN"]:
                frequencies = weights(frequencies)  # Learn a new set of wavelet coefficients to transform the data.
                # frequencies dimension: batchSize, numOutputSignals, frequencyDimension
            else:
                # Learn a new set of wavelet coefficients to transform the data.
                frequencies = torch.einsum(equationString, weights, frequencies)
                # frequencies dimension: batchSize, numOutputSignals, frequencyDimension

        return frequencies
