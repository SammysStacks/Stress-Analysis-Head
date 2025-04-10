import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.fourierOperator.fourierNeuralOperatorWeights import fourierNeuralOperatorWeights


class fourierNeuralOperatorLayer(fourierNeuralOperatorWeights):

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, addBiasTerm, activationMethod, skipConnectionProtocol, encodeRealFrequencies, encodeImaginaryFrequencies, learningProtocol='rFC'):
        super(fourierNeuralOperatorLayer, self).__init__(sequenceLength, numInputSignals, numOutputSignals, addBiasTerm, activationMethod, skipConnectionProtocol, encodeRealFrequencies, encodeImaginaryFrequencies, learningProtocol)

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

        # Learn a new set of wavelet coefficients using both of the frequency data.
        if self.encodeImaginaryFrequencies: imaginaryFourierData = self.applyEncoding(equationString='oin,bin->bon', frequencies=imaginaryFourierData, weights=self.imaginaryFourierWeights, frequencyTerms=imaginaryFrequencyTerms)
        if self.encodeRealFrequencies: realFourierData = self.applyEncoding(equationString='oin,bin->bon', frequencies=realFourierData, weights=self.realFourierWeights, frequencyTerms=realFrequencyTerms)
        # imaginaryFourierData: batchSize, numOutputChannels, fourierDimension
        # realFourierData: batchSize, numOutputChannels, fourierDimension

        # Return to physical space
        reconstructedData = self.backwardFFT(realFourierData, imaginaryFourierData, resampledTimes=None)[:, :, left_pad:left_pad + sequenceLength]
        # reconstructedData dimension: batchSize, numOutputChannels, signalDimension

        # Add a bias term if needed.
        if self.addBiasTerm: reconstructedData = reconstructedData + self.operatorBiases
        # reconstructedData dimension: batchSize, numOutputChannels, signalDimension

        return reconstructedData

    def applyEncoding(self, equationString, frequencies, weights, frequencyTerms=None):
        # Add in the learned wavelet coefficients.
        if frequencyTerms is not None: frequencies = frequencies + frequencyTerms
        # frequencies dimension: batchSize, numLiftedChannels, frequencyDimension

        if weights is not None:
            if 'FC' in self.learningProtocol or 'CNN' in self.learningProtocol:
                frequencies = weights(frequencies)  # Learn a new set of wavelet coefficients to transform the data.
                # frequencies dimension: batchSize, numOutputSignals, frequencyDimension
            else:
                # Learn a new set of wavelet coefficients to transform the data.
                frequencies = torch.einsum(equationString, weights, frequencies)
                # frequencies dimension: batchSize, numOutputSignals, frequencyDimension

        return frequencies
