from hmmlearn import hmm
import numpy as np

# Import files.
from .generalTherapyProtocol import generalTherapyProtocol


class hmmTherapyProtocol(generalTherapyProtocol):
    def __init__(self, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod):
        super().__init__(initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod)
        # Resampled bins for the parameter and prediction bins
        self.allParameterBins_resampled, self.allPredictionBins_resampled = self.generalMethods.resampleBins(self.allParameterBins, self.allPredictionBins, eventlySpacedBins=False)
        self.numStates = 11 # len(self.allParameterBins_resampled[0]) * len(self.allPredictionBins_resampled[0])

        # Parameters for observation sequence
        self.sequenceLength = self.numStates * 10
        self.numSequence = 30
        self.predictionSequenceLength = 2

        # HMM models
        self.HMMmodels = 'Gaussian'
        self.model = None

        self.emissionMatrix = self.initialEmissionMatrix()  # Emission matrix for the HMM
        self.transitionMatrix = self.initialTransitionMatrix()  # Transition matrix for the HMM
        self.observationSequence = self.generate_observation_sequences()  # Observation sequence for the HMM

    def initialEmissionMatrix(self):
        # Initialize the emission matrix.
        if self.simulateTherapy:
            normalized_emission_matrix = self.simulationProtocols.simulatedMapCompiledLossUnNormalized / self.simulationProtocols.simulatedMapCompiledLossUnNormalized.sum(axis=1, keepdims=True)
            return normalized_emission_matrix.numpy()
        else:
            # TODO: get the emission matrix from actual data
            pass

    def initialTransitionMatrix(self):
        # Initialize the transition matrix.
        # For start, assume uniform distribution for the transition matrix
        transition_matrix = np.full((self.numStates, self.numStates), 1.0 / self.numStates)
        return transition_matrix

    # Generate observation sequences (for simulation purposes)
    def generate_observation_sequences(self):
        observation_sequences = []  # Note this contains bin index
        # 10 times the number of sequences to ensure enough data
        for _ in range(self.numSequence):
            paramLossPairs = np.random.rand(self.sequenceLength, 2)
            paramLossPairsList = paramLossPairs.tolist()
            observation_sequences.append(paramLossPairsList)

        return observation_sequences

    def trainHMM(self):
        if self.HMMmodels == 'Gaussian':
            model = hmm.GaussianHMM(n_components=self.numStates, n_iter=100, covariance_type="diag", init_params="")
            print('lengh of emission matrix:', len(self.emissionMatrix))
            model.startprob_ = self.emissionMatrix #np.full(self.numStates, 1.0 / self.numStates)
            model.transmat_ = self.transitionMatrix

            # Prepare the observation sequences for training
            train_data = np.concatenate(self.observationSequence).reshape(-1, 2)
            lengths = [self.sequenceLength] * self.numSequence

            assert len(train_data) == sum(lengths), (
                f"Sum of lengths {sum(lengths)} does not match number of samples {len(train_data)}")

            # Fit the model
            model.fit(train_data, lengths)
            self.model = model

    def predict_optimal_sequence_transition(self, initial_state, sequence_length):
        if self.model is None:
            raise ValueError("The HMM model is not trained yet.")

        # Initialize the sequence with the initial state
        sequence = [initial_state]
        for _ in range(sequence_length - 1):
            current_state = sequence[-1]
            # Ensure current_state is an integer index
            current_state_index = np.argmax(current_state)
            next_state = np.argmax(self.model.transmat_[current_state_index])
            sequence.append(next_state)
        return sequence

    def updateTherapyState(self):
        initialParam = self.paramStatePath[-1].numpy()
        initialLoss = self.userMentalStateCompiledLoss[-1].numpy()
        # Flatten the arrays
        initialParam_flat = initialParam.flatten()
        initialLoss_flat = initialLoss.flatten()
        # Reshape the flattened arrays to 2D
        initialParam_flat_reshaped = initialParam_flat.reshape(-1, 1)
        initialLoss_flat_reshaped = initialLoss_flat.reshape(-1, 1)
        # Concatenate the reshaped arrays
        initialStates = np.concatenate((initialParam_flat_reshaped, initialLoss_flat_reshaped), axis=1)
        # Ensure initialStates is a valid state index
        initial_state_index = np.argmax(initialStates.flatten())
        sequence = self.predict_optimal_sequence_transition(initial_state=initial_state_index, sequence_length=self.predictionSequenceLength)
        return sequence
