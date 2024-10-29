import torch

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.neuralOperators.fourierOperator.fourierNeuralOperatorWeights import fourierNeuralOperatorWeights
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface


class fourierNeuralOperatorLayer(fourierNeuralOperatorWeights):

    def __init__(self, sequenceLength, numInputSignals, numOutputSignals, addBiasTerm, activationMethod, skipConnectionProtocol, encodeRealFrequencies, encodeImaginaryFrequencies, learningProtocol='rFC', extraOperators=()):
        super(fourierNeuralOperatorLayer, self).__init__(sequenceLength, numInputSignals, numOutputSignals, addBiasTerm, activationMethod, skipConnectionProtocol, encodeRealFrequencies, encodeImaginaryFrequencies, learningProtocol)
        self.extraOperators = extraOperators  # Extra operators to apply to the Fourier data.

    def forward(self, inputData, realFrequencyTerms=None, imaginaryFrequencyTerms=None):
        # Apply the activation function.
        if reversibleInterface.forwardDirection: inputData = self.activationFunction(inputData)
        # neuralOperatorOutput dimension: batchSize, numOutputChannels, signalDimension

        # Apply the Fourier neural operator and the skip connection.
        neuralOperatorOutput = self.fourierNeuralOperator(inputData, realFrequencyTerms, imaginaryFrequencyTerms)
        if self.skipConnectionModel is not None: neuralOperatorOutput = neuralOperatorOutput + self.skipConnectionModel(inputData)

        # Apply the activation function.
        if not reversibleInterface.forwardDirection: neuralOperatorOutput = self.activationFunction(neuralOperatorOutput)
        # neuralOperatorOutput dimension: batchSize, numOutputChannels, signalDimension

        return neuralOperatorOutput

    def waveletInterface(self, lowFrequency, highFrequencies):
        # Assert that the dimensions are correct.
        assert self.expectedSequenceLength == lowFrequency.size(-1), f"The expected sequence length must equal the wavelet length: {self.expectedSequenceLength}, {lowFrequency.size()}"
        assert self.numInputSignals == self.numOutputSignals, f"The number of input signals must equal the output signals for now: {self.numInputSignals}, {self.numOutputSignals}"
        assert len(highFrequencies) == 1, f"The high frequency terms must have only one decomposition: {len(highFrequencies)} {highFrequencies}"
        assert self.numInputSignals == 2*lowFrequency.size(1), f"The number of input signals must equal twice the incoming signals: {self.numInputSignals}, {lowFrequency.size(1)}"
        initialNumSignals = lowFrequency.size(1)

        # Combine the real and imaginary Fourier data.
        frequencyCoeffs = torch.cat(tensors=(lowFrequency, highFrequencies[0]), dim=1)
        # frequencyCoeffs dimension: batchSize, 2*numInputSignals, waveletDimension

        if not reversibleInterface.forwardDirection:
            # Apply the neural operator and the activation method.
            frequencyCoeffs = self.fourierNeuralOperator(frequencyCoeffs, realFrequencyTerms=None, imaginaryFrequencyTerms=None)
            frequencyCoeffs = self.activationFunction(frequencyCoeffs)  # Apply the activation function.
            # frequencyCoeffs dimension: batchSize, 2*numOutputSignals, waveletDimension
        else:
            # Apply the neural operator and the activation method.
            frequencyCoeffs = self.activationFunction(frequencyCoeffs)  # Apply the activation function.
            frequencyCoeffs = self.fourierNeuralOperator(frequencyCoeffs, realFrequencyTerms=None, imaginaryFrequencyTerms=None)
            # frequencyCoeffs dimension: batchSize, 2*numOutputSignals, waveletDimension

        # Split the real and imaginary Fourier data.
        lowFrequency, highFrequencies = frequencyCoeffs[:, 0:initialNumSignals], frequencyCoeffs[:, initialNumSignals:]
        assert lowFrequency.size() == highFrequencies.size(), f"The low and high frequency terms must have the same dimensions: {lowFrequency.size()}, {highFrequencies.size()}"

        return lowFrequency, [highFrequencies]

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

        # Apply the extra operators.
        if not reversibleInterface.forwardDirection: realFourierData, imaginaryFourierData = self.applyExtraOperators(realFourierData, imaginaryFourierData)
        # imaginaryFourierData: batchSize, numInputSignals, fourierDimension
        # realFourierData: batchSize, numInputSignals, fourierDimension

        # Learn a new set of wavelet coefficients using both of the frequency data.
        if self.learningProtocol in ['rFC', 'rCNN']: realFourierData, imaginaryFourierData = self.dualFrequencyWeights(realFourierData, imaginaryFourierData)
        else:  # Multiply relevant Fourier modes (Sampling low-frequency spectrum).
            if self.encodeImaginaryFrequencies: imaginaryFourierData = self.applyEncoding(equationString='oin,bin->bon', frequencies=imaginaryFourierData, weights=self.imaginaryFourierWeights, frequencyTerms=imaginaryFrequencyTerms)
            if self.encodeRealFrequencies: realFourierData = self.applyEncoding(equationString='oin,bin->bon', frequencies=realFourierData, weights=self.realFourierWeights, frequencyTerms=realFrequencyTerms)
            # b = batchSize, i = numInputChannels, o = numInputChannels, n = fourierDimension
        # imaginaryFourierData: batchSize, numOutputChannels, fourierDimension
        # realFourierData: batchSize, numOutputChannels, fourierDimension

        # Apply the extra operators.
        if reversibleInterface.forwardDirection: realFourierData, imaginaryFourierData = self.applyExtraOperators(realFourierData, imaginaryFourierData)
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
            if self.learningProtocol in ['FC', 'CNN']:
                frequencies = weights(frequencies)  # Learn a new set of wavelet coefficients to transform the data.
                # frequencies dimension: batchSize, numOutputSignals, frequencyDimension
            else:
                # Learn a new set of wavelet coefficients to transform the data.
                frequencies = torch.einsum(equationString, weights, frequencies)
                # frequencies dimension: batchSize, numOutputSignals, frequencyDimension

        return frequencies

    def applyExtraOperators(self, realFourierData, imaginaryFourierData):
        # Apply the extra operators.
        for extraOperatorInd in range(len(self.extraOperators)):
            if reversibleInterface.forwardDirection: extraOperatorInd = len(self.extraOperators) - 1 - extraOperatorInd
            extraOperator = self.extraOperators[extraOperatorInd]

            # Apply the extra operator.
            realFourierData, imaginaryFourierData = extraOperator.fourierInterface(realFourierData, imaginaryFourierData)
            # imaginaryFourierData: batchSize, numOutputChannels, fourierDimension
            # realFourierData: batchSize, numOutputChannels, fourierDimension

        return realFourierData, imaginaryFourierData
