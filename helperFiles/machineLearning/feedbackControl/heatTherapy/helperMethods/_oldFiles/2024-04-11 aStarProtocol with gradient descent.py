import numpy as np

from .generalProtocol import generalProtocol


class aStarProtocol(generalProtocol):
    def __init__(self, temperatureBounds, tempBinWidth, simulationParameters, learning_Rate=2.5):
        super().__init__(temperatureBounds, tempBinWidth, simulationParameters)
        # Define update parameters.
        self.learning_Rate = learning_Rate
        self.percentHeuristic = 1

        # Specific A Star Protocol parameters.
        self.decayConstant = 0.001  # The decay constant for the personalized map.
        self.tempBinsVisited = np.full(len(self.temp_bins), False)

        # Initialize the heuristic and personalized maps.
        self.heuristicMap = self.initializeHeuristicMap(self.simulationProtocols.numSimulationHeuristicSamples)  # Estimate on what temperatures you like. Based on population average.
        self.discretePersonalizedMap = []  # list of Gaussians

        if self.simulateTherapy:
            self.simulationProtocols.simulatedMap = self.initializeHeuristicMap(self.simulationProtocols.numSimulationTrueSamples)  # Estimate on what temperatures you like. Based on population average.

    def updateTherapyState(self):
        # Unpack the current user state.
        currentUserState = self.userStatePath[-1]
        currentUserTemp, currentUserLoss = currentUserState

        # Update the temperatures visited.
        tempBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)
        self.tempBinsVisited[tempBinIndex] = True

        # Update the personalized user map.
        self.trackCurrentState(currentUserState)  # Keep track of each discrete temperature-loss pair.
        personalizedMap = self.getUpdatedPersonalizedMap()  # Convert the discrete map to a probability distribution.

        # Combine the heuristic and personalized maps together and update the weighting.
        finalMap = self.percentHeuristic * self.heuristicMap + (1 - self.percentHeuristic) * personalizedMap
        finalMap = finalMap / (1E-10 + np.sum(finalMap, axis=1)[:, np.newaxis])
        self.updateAlpha()

        # Calculate the gradient of the final map.
        gradientT, gradientL = self.calculateGradient(finalMap, currentUserLoss)

        # Determine direction based on temperature gradient at current loss
        lossBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)
        tempGradient = gradientT[tempBinIndex, lossBinIndex]
        deltaTemp = self.learning_Rate * tempGradient

        # Calculate the new temperature.
        newUserTemp = self.boundNewTemperature(currentUserTemp - deltaTemp)

        return newUserTemp, finalMap, personalizedMap

    # ------------------------ Update Parameters ------------------------ #

    def calculateGradient(self, currentMap, lossValue):
        # Calculate the loss difference map.
        deltaLoss = self.loss_bins - lossValue  # Assuming we want to minimize the loss

        # Compute the gradient using numpy.gradient()
        gradients = np.gradient(currentMap * deltaLoss)  # Dimension: 2, allNumParameterBins, numPredictionBins

        # Assert that we calculated the correct gradient.
        assert len(gradients) == 2, "The gradient calculation is incorrect."

        return gradients

    def updateAlpha(self):
        # Calculate filled ratio
        percentConfidence = self.tempBinsVisited.sum() / len(self.tempBinsVisited)
        self.percentHeuristic = 1 - percentConfidence

    # ------------------------ Personalization Interface ------------------------ #

    def trackCurrentState(self, currentUserState):
        currentUserTemp, currentUserLoss = currentUserState

        # Smoothen out the discrete map into a probability distribution.
        gaussMatrix = self.createGaussianMap(gausMean=(currentUserLoss, currentUserTemp), gausSTD=(0.2, 5))
        self.discretePersonalizedMap.append(gaussMatrix)

    @staticmethod
    def personalizedMapWeightingFunc(timeDelays, decay_constant):
        return np.exp(-decay_constant * np.asarray(timeDelays))

    def getUpdatedPersonalizedMap(self):
        # Assert the integrity of the state tracking.
        assert len(self.timeDelays) == len(self.discretePersonalizedMap), \
            f"The time delays and discrete maps are not the same length. {self.timeDelays} {self.discretePersonalizedMap}"

        # Get the weighting for each discrete temperature-loss pair.
        currentTimeDelays = np.abs(np.asarray(self.timeDelays) - self.timeDelays[-1])
        personalizedMapWeights = self.personalizedMapWeightingFunc(currentTimeDelays, self.decayConstant)
        personalizedMapWeights = personalizedMapWeights / np.sum(personalizedMapWeights)  # Normalize the weights

        # Perform a weighted average of all the personalized maps.
        currentMap = np.sum(self.discretePersonalizedMap * personalizedMapWeights[:, np.newaxis, np.newaxis], axis=0)
        currentMap = currentMap / (np.sum(currentMap))  # Normalize along the temperature axis.

        return currentMap

    # ------------------------ Heuristic Interface ------------------------ #

    def initializeHeuristicMap(self, numPoints):
        if self.simulateTherapy:
            # Get the simulated data points.
            initialStates = self.simulationProtocols.generateSimulatedMap(numPoints)
        else:
            # Get the real data points.
            initialStates = None

        # Get the heuristic matrix from the simulated points.
        initialData = self.compileLossStates(initialStates)        # InitialData dim: numPoints, (T, L)
        heuristicMatrix = self.getProbabilityMatrix(initialData)    # Adding Gaussians and normalizing probability

        return heuristicMatrix

    def createGaussianMap(self, gausMean, gausSTD):
        # Generate a grid for Gaussian distribution calculations
        x, y = np.meshgrid(self.loss_bins, self.temp_bins)

        # Calculate Gaussian distribution values across the grid
        gaussMatrix = np.exp(-0.5 * ((x - gausMean[0]) ** 2 / gausSTD[0] ** 2 + (y - gausMean[1]) ** 2 / gausSTD[1] ** 2))
        gaussMatrix = gaussMatrix / (np.sum(gaussMatrix))  # Normalize the Gaussian matrix

        return gaussMatrix

    def getProbabilityMatrix(self, initialData):
        # dim initialData = temp x loss value
        # Initialize probability matrix holder.
        probabilityMatrix = np.zeros((len(self.temp_bins), len(self.loss_bins)))

        # Calculate the probability matrix.
        for initialDataPoints in initialData:
            currentUserTemp, currentUserLoss = initialDataPoints

            # Generate 2D gaussian matrix.
            gaussianMatrix = self.createGaussianMap(gausMean=(currentUserLoss, currentUserTemp), gausSTD=(0.2, 5))

            # Add the gaussian map to the matrix
            probabilityMatrix += gaussianMatrix

        # Normalize the probability matrix.
        probabilityMatrix /= np.sum(probabilityMatrix)

        return probabilityMatrix
