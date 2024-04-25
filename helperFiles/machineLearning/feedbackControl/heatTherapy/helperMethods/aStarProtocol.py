# General
import numpy as np
from scipy.ndimage import gaussian_filter

from .generalProtocol import generalProtocol


class aStarProtocol(generalProtocol):
    def __init__(self, temperatureBounds, tempBinWidth, simulationParameters, learningRate=5):
        super().__init__(temperatureBounds, tempBinWidth, simulationParameters)
        # Define update parameters.
        self.gausSTD = np.array([0.05, 2.5])  # The standard deviation for the Gaussian distribution.
        self.learningRate = learningRate  # The learning rate for the therapy.
        self.discretePersonalizedMap = []  # The discrete personalized map.

        # Bias terms.
        self.percentHeuristic = 1  # The percentage of the heuristic map to use.
        self.explorationBias = 1  # The bias for exploration.
        self.uncertaintyBias = 1  # The bias for uncertainty.

        # Specific A Star Protocol parameters.
        self.tempBinsVisited = np.full(self.numTempBins, False)
        self.decayConstant = 1 / (2 * 3600)  # The decay constant for the personalized map.
        self.applyGaussianFilter = False  # Whether to apply a Gaussian filter on the discrete maps.

        # Initialize the heuristic and personalized maps.
        self.heuristicMap = self.initializeHeuristicMaps()  # Estimate on what temperatures you like. Based on population average.
        self.initializeFirstPersonalizedMap()  # list of probability maps.

    def updateTherapyState(self):
        # Unpack the current user state.
        currentUserState = self.userStatePath[-1]  # Order: (T, Loss) (get the latest user state) each (T, Loss) stored in the userStatePath list
        currentUserTemp, currentUserLoss = currentUserState

        # Update the temperatures visited.
        tempBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)
        self.tempBinsVisited[tempBinIndex] = True  # documents the temperature visited by indexing the temperature bin and set the value to true

        # Update the personalized user map.
        self.trackCurrentState(currentUserState)  # Keep track of each discrete temperature-loss pair.
        personalizedMap = self.getUpdatedPersonalizedMap()  # Convert the discrete map to a probability distribution.

        # Combine the heuristic and personalized maps and update the weighting.
        probabilityMap = self.percentHeuristic * self.heuristicMap + (1 - self.percentHeuristic) * personalizedMap
        self.updateAlpha()

        # Find the best temperature in the gradient direction.
        newUserTemp, benefitFunction = self.findOptimalDirection(probabilityMap, currentUserState)
        newUserTemp = newUserTemp + self.uncertaintyBias * np.random.normal(loc=0, scale=0.5)  # Add noise to the gradient.
        # deltaTemp = self.findNewTemperature(currentUserState, gradientDirection)
        # deltaTemp = deltaTemp + self.uncertaintyBias * np.random.normal(loc=0, scale=0.5)  # Add noise to the gradient.

        # Calculate the new temperature.
        newUserTemp = self.boundNewTemperature(newUserTemp, bufferZone=1)

        return newUserTemp, (benefitFunction, self.heuristicMap, personalizedMap, self.simulationProtocols.simulatedMap)

    def findNewTemperature(self, currentUserState, gradientDirection):
        # Unpack the current user state.
        currentUserTemp, currentUserLoss = currentUserState

        # Determine a direction based on temperature gradient at current loss
        tempBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)
        tempGradient = gradientDirection[tempBinIndex]

        return tempGradient

    # ------------------------ Update Parameters ------------------------ #

    def findOptimalDirection(self, probabilityMap, currentUserState):
        # Calculate benefits/loss of exploring/moving.
        potentialLossBenefit = self.loss_bins
        probabilityMap = probabilityMap / probabilityMap.sum(axis=1)[:, np.newaxis]  # Normalize the probability map.

        # Calculate the expected rewards.
        potentialRewards = potentialLossBenefit[np.newaxis, :]
        expectedRewards = probabilityMap * potentialRewards

        # Find the best temperature bin index in the rewards.
        expectedRewardAtTemp = expectedRewards.sum(axis=1)  # Normalize across temperature bins.
        bestTempBinIndex = np.argmin(expectedRewardAtTemp)

        return self.temp_bins[bestTempBinIndex] + self.tempBinWidth / 2, expectedRewards

        # # Compute the gradient.
        # potentialTemperatureRewards = np.gradient(potentialTemperatureRewards)  # Dimension: 2, numTempBins, numLossBins

    def updateAlpha(self):
        # Calculate the percentage of the temperature bins visited.
        percentConfidence = self.tempBinsVisited.sum() / len(self.tempBinsVisited)

        # Update the confidence flags.
        self.percentHeuristic = min(self.percentHeuristic, 1 - percentConfidence) - 0.001
        self.percentHeuristic = min(1.0, max(0.0, self.percentHeuristic))

        # Update the bias terms.
        self.explorationBias = self.percentHeuristic  # TODO
        self.uncertaintyBias = self.percentHeuristic  # TODO

    # ------------------------ Personalization Interface ------------------------ #

    def trackCurrentState(self, currentUserState):
        # Smoothen out the discrete map into a probability distribution.
        probabilityMatrix = self.getProbabilityMatrix([currentUserState])
        self.discretePersonalizedMap.append(probabilityMatrix)  # the discretePersonalizedMap list will store the probability matrix

    @staticmethod
    def personalizedMapWeightingFunc(timeDelays, decay_constant):
        # Ebbinghaus forgetting curve.
        return np.exp(-decay_constant * np.asarray(timeDelays))

    def getUpdatedPersonalizedMap(self):
        # Assert the integrity of the state tracking.
        assert len(self.temperatureTimepoints) == len(self.discretePersonalizedMap), \
            f"The time delays and discrete maps are not the same length. {self.temperatureTimepoints} {self.discretePersonalizedMap}"
        # Unpack the temperature-timepoints relation.
        tempTimepoints = np.asarray(self.temperatureTimepoints)
        associatedTempInds = tempTimepoints[:, 1]
        timePoints = tempTimepoints[:, 0]

        # Get the weighting for each discrete temperature-loss pair.
        currentTimeDelays = np.abs(timePoints - timePoints[-1])
        personalizedMapWeights = self.personalizedMapWeightingFunc(currentTimeDelays, self.decayConstant)

        # For each temperature bin.
        for tempIndex in range(self.numTempBins):
            # If the temperature bin has been visited.
            if tempIndex in associatedTempInds:
                tempIndMask = associatedTempInds == tempIndex

                # Normalize the weights per this bin.
                personalizedMapWeights[tempIndMask] = personalizedMapWeights[tempIndMask] / personalizedMapWeights[tempIndMask].sum()

        # Perform a weighted average of all the personalized maps.
        personalizedMap = np.sum(self.discretePersonalizedMap * personalizedMapWeights[:, np.newaxis, np.newaxis], axis=0)

        if self.applyGaussianFilter:
            # Smoothen the personalized map.
            personalizedMap = self.smoothenArray(personalizedMap, sigma=self.gausSTD[::-1])

        # Normalize the personalized map.
        personalizedMap = personalizedMap / personalizedMap.sum()  # Normalize along the temperature axis.

        return personalizedMap

    def initializeFirstPersonalizedMap(self):
        # Initialize a uniform personalized map. No bias.
        uniformMap = np.ones((self.numTempBins, self.numLossBins))
        uniformMap = uniformMap / uniformMap.sum()

        # Store the initial personalized map estimate.
        # self.discretePersonalizedMap.append(uniformMap)
        # self.timePoints.append((0, ))

    # ------------------------ Heuristic Interface ------------------------ #

    def initializeHeuristicMaps(self):
        if self.simulateTherapy:
            # Get the simulated data points.
            initialHeuristicStates = self.simulationProtocols.generateSimulatedMap(self.simulationProtocols.numSimulationHeuristicSamples, simulatedMapType=self.simulationProtocols.heuristicMapType)
            initialSimulatedStates = self.simulationProtocols.generateSimulatedMap(self.simulationProtocols.numSimulationTrueSamples, simulatedMapType=self.simulationProtocols.simulatedMapType)
            # initialHeuristicStates dimension: numSimulationHeuristicSamples, (T, PA, NA, SA); 2D array
            # initialSimulatedStates dimension: numSimulationTrueSamples, (T, PA, NA, SA); 2D array
            print('intialHeuristic', initialHeuristicStates)
            # Get the simulated matrix from the simulated points.
            initialSimulatedData = self.compileLossStates(initialSimulatedStates)  # initialSimulatedData dimension: numSimulationTrueSamples, (T, L).
            self.simulationProtocols.simulatedMap = self.getProbabilityMatrix(initialSimulatedData)  # Spreading delta function probability.
        else:
            # Get the real data points.
            initialHeuristicStates = self.empatchProtocols.getTherapyData()
            # dimension of initialHeuristicStates: (T, PA, NA, SA); 2D array (we have 6 points for now)
            print('intialHeuristic', initialHeuristicStates)
            # Get the simulated data points.
            initialSimulatedStates = self.simulationProtocols.generateSimulatedMap(self.simulationProtocols.numSimulationTrueSamples, simulatedMapType=self.simulationProtocols.simulatedMapType)
            initialSimulatedData = self.compileLossStates(initialSimulatedStates)  # initialSimulatedData dimension: numSimulationTrueSamples, (T, L).
            self.simulationProtocols.simulatedMap = self.getProbabilityMatrix(initialSimulatedData)  # Spreading delta function probability.

        # Get the heuristic matrix from the simulated points.
        initialHeuristicData = self.compileLossStates(initialHeuristicStates)  # InitialData dim: numPoints, (T, L)
        heuristicMap = self.getProbabilityMatrix(initialHeuristicData)  # Adding Gaussian distributions and normalizing the probability

        return heuristicMap

    @staticmethod
    def smoothenArray(deltaFunctionMatrix, sigma):
        return gaussian_filter(deltaFunctionMatrix, sigma=sigma)

    def createGaussianMap(self, gausMean, gausSTD):
        # Generate a grid for Gaussian distribution calculations
        x, y = np.meshgrid(self.loss_bins, self.temp_bins)

        # Calculate Gaussian distribution values across the grid
        gaussMatrix = np.exp(-0.5 * ((x - gausMean[0]) ** 2 / gausSTD[0] ** 2 + (y - gausMean[1]) ** 2 / gausSTD[1] ** 2))
        gaussMatrix = gaussMatrix / gaussMatrix.sum()  # Normalize the Gaussian matrix

        return gaussMatrix

    def getProbabilityMatrix(self, initialData):
        """ initialData: numPoints, (T, L); 2D array"""
        # Initialize probability matrix holder.
        probabilityMatrix = np.zeros((self.numTempBins, self.numLossBins))

        # Calculate the probability matrix.
        for initialDataPoints in initialData:
            currentUserTemp, currentUserLoss = initialDataPoints

            if self.applyGaussianFilter:
                # Generate a delta function probability.
                tempBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)
                lossBinIndex = self.getBinIndex(self.loss_bins, currentUserLoss)
                probabilityMatrix[tempBinIndex, lossBinIndex] += 1  # map out bins and fill out with discrete values
            else:
                # Generate 2D gaussian matrix.
                gaussianMatrix = self.createGaussianMap(gausMean=(currentUserLoss, currentUserTemp), gausSTD=self.gausSTD)
                probabilityMatrix += gaussianMatrix  # Add the gaussian map to the matrix

        if self.applyGaussianFilter:
            # Smoothen the probability matrix.
            probabilityMatrix = self.smoothenArray(probabilityMatrix, sigma=self.gausSTD[::-1])

        # Normalize the probability matrix.
        probabilityMatrix = probabilityMatrix / probabilityMatrix.sum()

        return probabilityMatrix
