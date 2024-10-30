import torch

from .waveletNeuralOperatorWeights import waveletNeuralOperatorWeights
from ...reversibleComponents.reversibleInterface import reversibleInterface


class waveletNeuralOperatorLayer(waveletNeuralOperatorWeights):

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, numDecompositions, waveletType, mode, addBiasTerm, activationMethod,
                 skipConnectionProtocol, encodeLowFrequencyProtocol='lowFreq', encodeHighFrequencyProtocol='highFreq', learningProtocol='CNN', extraOperators=()):
        super(waveletNeuralOperatorLayer, self).__init__(sequenceLength, numInputSignals, numOutputSignals, numDecompositions, waveletType, mode, addBiasTerm, activationMethod,
                                                         skipConnectionProtocol, encodeLowFrequencyProtocol, encodeHighFrequencyProtocol, learningProtocol)
        self.extraOperators = extraOperators  # Extra operators to apply to the wavelet data.
        
    def forward(self, neuralOperatorData, residualLowFrequencyTerms=None, residualHighFrequencyTerms=None):
        # Apply the wavelet neural operator and the skip connection.
        neuralOperatorData = self.waveletNeuralOperator(neuralOperatorData, residualLowFrequencyTerms, residualHighFrequencyTerms)
        if self.skipConnectionModel is not None: neuralOperatorData = neuralOperatorData + self.skipConnectionModel(neuralOperatorData)  # Not reversible.
        # neuralOperatorData dimension: batchSize, numOutputSignals, finalLength

        # Apply the activation function.
        neuralOperatorData = self.activationFunction(neuralOperatorData)
        # neuralOperatorData dimension: batchSize, numOutputSignals, finalLength

        return neuralOperatorData
    
    def backwardPass(self, neuralOperatorData, residualLowFrequencyTerms=None, residualHighFrequencyTerms=None):
        # Assert that the skip connection model is None.
        assert self.skipConnectionModel is None, "The skip connection model must be None for the reversible interface."

        # Apply the activation function and the wavelet neural operator.
        neuralOperatorData = self.activationFunction(neuralOperatorData)
        neuralOperatorData = self.waveletNeuralOperator(neuralOperatorData, residualLowFrequencyTerms, residualHighFrequencyTerms)
        # neuralOperatorData dimension: batchSize, numOutputSignals, finalLength

        return neuralOperatorData

    def reversibleInterface(self, neuralOperatorData):
        assert self.skipConnectionModel is None, "The skip connection model must be None for the reversible interface."
        if not reversibleInterface.forwardDirection: return self.forward(neuralOperatorData, residualLowFrequencyTerms=None, residualHighFrequencyTerms=None)
        else: return self.backwardPass(neuralOperatorData, residualLowFrequencyTerms=None, residualHighFrequencyTerms=None)

    def fourierInterface(self, realFourierData, imaginaryFourierData):
        # Assert that the dimensions are correct.
        assert self.expectedSequenceLength == realFourierData.size(-1), f"The expected sequence length must equal the real Fourier data length: {self.expectedSequenceLength}, {realFourierData.size(-1)}"
        assert realFourierData.size() == imaginaryFourierData.size(), f"The real and imaginary Fourier data must have the same dimensions: {realFourierData.size()}, {imaginaryFourierData.size()}"
        assert self.numInputSignals == self.numOutputSignals, f"The number of input signals must equal the output signals for now: {self.numInputSignals}, {self.numOutputSignals}"
        assert self.numInputSignals == 2*realFourierData.size(1), f"The number of input signals must equal twice the incoming signals: {self.numInputSignals}, {realFourierData.size(1)}"
        initialNumSignals = realFourierData.size(1)

        # Combine the real and imaginary Fourier data.
        fourierData = torch.cat(tensors=(realFourierData, imaginaryFourierData), dim=1)
        # fourierData dimension: batchSize, 2*numInputSignals, fourierDimension
        
        if not reversibleInterface.forwardDirection:
            # Apply the neural operator and the activation method.
            fourierData = self.waveletNeuralOperator(fourierData, residualLowFrequencyTerms=None, residualHighFrequencyTerms=None)
            fourierData = self.activationFunction(fourierData)  # Apply the activation function.
            # fourierData dimension: batchSize, 2*numOutputSignals, fourierDimension
        else:
            # Apply the neural operator and the activation method.
            fourierData = self.activationFunction(fourierData)  # Apply the activation function.
            fourierData = self.waveletNeuralOperator(fourierData, residualLowFrequencyTerms=None, residualHighFrequencyTerms=None)
            # fourierData dimension: batchSize, 2*numOutputSignals, fourierDimension

        # Split the real and imaginary Fourier data.
        realFourierData, imaginaryFourierData = fourierData[:, 0:initialNumSignals], fourierData[:, initialNumSignals:]
        assert realFourierData.size() == imaginaryFourierData.size(), f"The real and imaginary Fourier data must have the same dimensions: {realFourierData.size()}, {imaginaryFourierData.size()}"

        return realFourierData, imaginaryFourierData

    def waveletNeuralOperator(self, inputData, residualLowFrequencyTerms=None, residualHighFrequencyTerms=None):
        # Extract the input data parameters.
        batchSize, numInputSignals, sequenceLength = inputData.size()
        left_pad = (self.sequenceLength - sequenceLength) // 2
        right_pad = self.sequenceLength - sequenceLength - left_pad

        # Apply padding to the input data
        inputData = torch.nn.functional.pad(inputData, pad=(left_pad, right_pad), mode='constant', value=0)
        # inputData dimension: batchSize, numInputSignals, paddedSequenceLength

        # Perform wavelet decomposition.
        lowFrequency, highFrequencies = self.dwt(inputData)  # Note: each channel is treated independently here.
        # highFrequencies[decompositionLayer] dimension: batchSize, numLiftedChannels, highFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numLiftedChannels, lowFrequencyShape

        # Apply the extra operators.
        if not reversibleInterface.forwardDirection: lowFrequency, highFrequencies = self.applyExtraOperators(lowFrequency, highFrequencies)

        # Encode each frequency decomposition, separating high and low frequencies.
        lowFrequency, highFrequencies = self.independentFrequencyAnalysis(lowFrequency, highFrequencies, residualLowFrequencyTerms, residualHighFrequencyTerms)
        # highFrequencies[highFrequencyInd] dimension: batchSize, numOutputSignals, highFrequenciesShapes[decompositionLayer]
        # lowFrequency dimension: batchSize, numOutputSignals, lowFrequencyShape

        # Apply the extra operators.
        if reversibleInterface.forwardDirection: lowFrequency, highFrequencies = self.applyExtraOperators(lowFrequency, highFrequencies)

        # Perform wavelet reconstruction.
        reconstructedData = self.idwt((lowFrequency, highFrequencies))
        reconstructedData = reconstructedData[:, :, left_pad:left_pad + sequenceLength]
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

        if self.learningProtocol in ['rFC', 'rCNN']:
            # Learn a new set of wavelet coefficients using both of the frequency data.
            lowFrequency, highFrequencies[0] = self.dualFrequencyWeights(lowFrequency + (residualLowFrequencyTerms or 0), highFrequencies[0] + (residualHighFrequencyTerms or 0))
            return lowFrequency, highFrequencies

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
            if self.learningProtocol in ['FC', 'CNN']:
                frequencies = weights(frequencies)  # Learn a new set of wavelet coefficients to transform the data.
                # frequencies dimension: batchSize, numOutputSignals, frequencyDimension
            else:
                # Learn a new set of wavelet coefficients to transform the data.
                frequencies = torch.einsum(equationString, weights, frequencies)
                # frequencies dimension: batchSize, numOutputSignals, frequencyDimension

        return frequencies

    def applyExtraOperators(self, lowFrequency, highFrequencies):
        # Apply the extra operators.
        for extraOperatorInd in range(len(self.extraOperators)):
            if reversibleInterface.forwardDirection: extraOperatorInd = len(self.extraOperators) - 1 - extraOperatorInd
            extraOperator = self.extraOperators[extraOperatorInd]

            # Apply the extra operator.
            lowFrequency, highFrequencies = extraOperator.waveletInterface(lowFrequency, highFrequencies)
            # highFrequencies[decompositionLayer] dimension: batchSize, numLiftedChannels, highFrequenciesShapes[decompositionLayer]
            # lowFrequency dimension: batchSize, numLiftedChannels, lowFrequencyShape

        return lowFrequency, highFrequencies
