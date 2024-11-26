from hmmlearn import hmm
import numpy as np
import torch
# Import files.
from .generalTherapyProtocol import generalTherapyProtocol


class hmmTherapyProtocol(generalTherapyProtocol):
    def __init__(self, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod):
        super().__init__(initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod)
        # Resampled bins for the parameter and prediction bins
        self.allParameterBins_resampled, self.allPredictionBins_resampled = self.generalMethods.resampleBins(self.allParameterBins, self.allPredictionBins, eventlySpacedBins=False)
        self.numStates = len(self.allParameterBins_resampled[0])  # 11 in this case; len(self.allParameterBins_resampled[0]) * len(self.allPredictionBins_resampled[0])

        # Parameters for observation sequence
        self.sequenceLength = self.numStates * 500
        self.numSequence = 19
        self.predictionSequenceLength = 2

        # HMM models
        self.HMMmodels = 'categorical'
        self.hmmModel = None

        self.emissionMatrix = self.initialEmissionMatrix()  # Emission matrix for the HMM
        self.transitionMatrix = self.initialTransitionMatrix()  # Transition matrix for the HMM
        self.startProb = np.full(self.numStates, 1.0 / self.numStates)
        self.observationSequence_1D = self.generate_observation_sequences_1D()  # for multinomial HMM

    def initialEmissionMatrix(self):
        # Initialize the emission matrix.
        if self.simulateTherapy:
            normalized_initial_matrix = self.simulationProtocols.simulatedMapCompiledLossUnNormalized / self.simulationProtocols.simulatedMapCompiledLossUnNormalized.sum(axis=1, keepdims=True)
            print("Initial emission matrix shape:", normalized_initial_matrix.shape)
            return normalized_initial_matrix.numpy()
        else:
            # TODO: get the emission matrix from actual data
            normalized_initial_matrix = self.simulationProtocols.realSimMapCompiledLoss / self.simulationProtocols.realSimMapCompiledLoss.sum(axis=1, keepdims=True)
            print("Initial emission matrix shape:", normalized_initial_matrix.shape)
            return normalized_initial_matrix.numpy()

    def initialTransitionMatrix(self):
        # Initialize the transition matrix.
        # For start, assume uniform distribution for the transition matrix
        transition_matrix = np.full((self.numStates, self.numStates), 1.0 / self.numStates)
        return transition_matrix

    def generate_observation_sequences_1D(self):
        if self.simulateTherapy:
            observation_sequences = []
            current_state = np.random.choice(self.numStates, p=self.startProb)

            for _ in range(self.sequenceLength):
                observed_bin = np.random.choice(len(self.allPredictionBins_resampled[0]), p=self.emissionMatrix[current_state])
                observation_sequences.append(observed_bin)
                current_state = np.random.choice(self.numStates, p=self.transitionMatrix[current_state])

            observation_sequences = np.asarray(observation_sequences).reshape(-1, 1)
            return observation_sequences
        else:
            # TODO: think about how we are gonna use the actual data to train the HMM
            # right now this is just the same as generating random sequence from real data
            observation_sequences = []
            current_state = np.random.choice(self.numStates, p=self.startProb)

            for _ in range(self.sequenceLength):
                observed_bin = np.random.choice(len(self.allPredictionBins_resampled[0]), p=self.emissionMatrix[current_state])
                observation_sequences.append(observed_bin)
                current_state = np.random.choice(self.numStates, p=self.transitionMatrix[current_state])

            observation_sequences = np.asarray(observation_sequences).reshape(-1, 1)
            return observation_sequences
    def trainHMM(self):
        if self.HMMmodels == 'categorical':
            model = hmm.CategoricalHMM(n_components=self.numStates, n_iter=1000, init_params="")
            model.startprob_ = self.startProb
            model.transmat_ = self.transitionMatrix
            model.emissionprob_ = np.asarray(self.emissionMatrix)

            print('self.emissionMatrix:', self.emissionMatrix.shape)
            print('length of observation sequence:', len(self.observationSequence_1D))
            print('unique observation symbols:', np.unique(self.observationSequence_1D))

            # Check if the observation symbols are within the correct range
            assert np.all(self.observationSequence_1D >= 0) and np.all(self.observationSequence_1D < self.emissionMatrix.shape[1]), "Observation symbols out of range"

            # Prepare the observation sequences for training
            train_data = self.observationSequence_1D
            print("Train data shape:", train_data.shape)
            model.fit(train_data)
            self.hmmModel = model

    def predict_optimal_sequence(self, currentParamIndex, sequence_length):
        predicted_sequence = [currentParamIndex]
        for _ in range(sequence_length - 1):
            # Sample the next state based on the transition probabilities
            nextStateIndex = np.random.choice(self.numStates, p=self.hmmModel.transmat_[currentParamIndex])
            predicted_sequence.append(nextStateIndex)
            currentParamIndex = nextStateIndex
        return predicted_sequence

    # def predict_optimal_sequence(self, currentParamIndex, sequence_length):
    #     predicted_sequence = [currentParamIndex]
    #     loss_bin_sequence = [np.argmax(self.simulationProtocols.simulatedMapCompiledLoss[currentParamIndex])]
    #     for _ in range(sequence_length - 1):
    #         # Sample the next state based on the transition probabilities
    #         nextStateIndex = np.random.choice(self.numStates, p=self.hmmModel.transmat_[currentParamIndex])
    #         predicted_sequence.append(nextStateIndex)
    #         currentParamIndex = nextStateIndex
    #         loss_bin_sequence.append(np.argmax(self.simulationProtocols.simulatedMapCompiledLoss[currentParamIndex]))
    #         if np.argmax(self.simulationProtocols.simulatedMapCompiledLoss[currentParamIndex]) < 0.2:
    #             print('argmax ')
    #             print('self.simulationProtocols.simulatedMapCompiledLoss[currentParamIndex]', self.simulationProtocols.simulatedMapCompiledLoss[currentParamIndex])
    #             break
    #
    #     loss_bin_list = [tensor.item() for tensor in loss_bin_sequence]
    #     return predicted_sequence, loss_bin_list

    def updateTherapyState(self):
        # from IPython import embed
        # embed()
        currentParam = self.paramStatePath[-1].item()
        currentLoss = self.userMentalStateCompiledLoss[-1].item()
        paramBin = self.dataInterface.getBinIndex(self.allParameterBins_resampled[0], currentParam)
        lossBin = self.dataInterface.getBinIndex(self.allPredictionBins_resampled[0], currentLoss)
        sequence_length = 2
        predicted_sequence = self.predict_optimal_sequence(currentParamIndex=paramBin, sequence_length=sequence_length)
        print('initialparameter:', currentParam)
        print('initial Parameter Bin:', paramBin)
        print('predicted sequence:', predicted_sequence)
        # this is a bin index
        updatedParamIndex = predicted_sequence[-1] # + np.random.normal(loc=0, scale=0.5) # add noise to the parameter (subject would not feel the temperature difference within bin)
        print('updatedParamIndex:', updatedParamIndex)
        # unbound parameter
        updatedParam = torch.tensor(self.allParameterBins_resampled[0][updatedParamIndex])
        print('updatedParam:', updatedParam)
        if self.simulateTherapy:
            return updatedParam, self.simulationProtocols.simulatedMapCompiledLoss
        else:
            return updatedParam, self.simulationProtocols.realSimMapCompiledLoss



