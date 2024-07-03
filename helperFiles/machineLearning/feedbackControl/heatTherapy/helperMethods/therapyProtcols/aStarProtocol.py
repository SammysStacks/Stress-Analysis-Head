# General
import numpy as np
import torch
from .generalTherapyProtocol import generalTherapyProtocol


class aStarTherapyProtocol(generalTherapyProtocol):
    def __init__(self, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod, learningRate=5):
        super().__init__(initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod)
        # Define update parameters.
        # TODO: rexamine the gausSTD
        self.gausParam_STD = torch.tensor([0.3333]) #self.gausParameterSTDs  # The standard deviation for the Gaussian distribution.
        self.gausLoss_STD = torch.tensor([0.05]) #self.gausLossSTDs
        self.learningRate = learningRate  # The learning rate for the therapy.
        self.discretePersonalizedMap = []  # The discrete personalized map.

        # Bias terms.
        self.percentHeuristic = 1  # The percentage of the heuristic map to use.
        self.explorationBias = 1  # The bias for exploration.
        self.uncertaintyBias = 1  # The bias for uncertainty.

        # Specific A Star Protocol parameters.
        self.paramBinsVisited = np.full(self.allNumParameterBins, False)
        self.decayConstant = 1 / (2 * 3600)  # The decay constant for the personalized map.

        # Initialize the heuristic and personalized maps.
        self.heuristicMap = self.initializeHeuristicMaps()  # Estimate on what temperatures you like. Based on population average.
        self.initializeFirstPersonalizedMap()  # list of probability maps.

        # resampled bins for the parameter and prediction bins
        self.allParameterBins_resampled, self.allPredictionBins_resampled = self.generalMethods.resampleBins(self.allParameterBins, self.allPredictionBins, eventlySpacedBins=False)

    def updateTherapyState(self):
        # Get the current user state.
        currentParam = self.paramStatePath[-1].squeeze(0)
        currentCompiledLoss = self.userMentalStateCompiledLoss[-1]
        # Update the temperatures visited.
        paramBinIndex = self.dataInterface.getBinIndex(self.allParameterBins_resampled, currentParam)
        print('allParameterBins_resampled: ', self.allParameterBins_resampled)
        print('paramBinIndex: ', paramBinIndex)
        self.paramBinsVisited[paramBinIndex] = True  # documents the temperature visited by indexing the temperature bin and set the value to true
        # Update the personalized user map.
        self.trackCurrentState(currentParam, currentCompiledLoss)  # Keep track of each discrete temperature-loss pair.
        personalizedMap = self.getUpdatedPersonalizedMap()  # Convert the discrete map to a probability distribution.

        # Get the current time point.
        timePoint, userState = self.getCurrentState()
        self.temperatureTimepoints.append((timePoint, tempBinIndex))

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
        # potentialTemperatureRewards = np.gradient(potentialTemperatureRewards)  # Dimension: 2, allNumParameterBins, numPredictionBins

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

    def trackCurrentState(self, currentParam, currentCompiledLoss):
        print('currentParam: ', currentParam)
        print('currentCompiledLoss: ', currentCompiledLoss)
        initialSingleEmotionData = torch.cat((currentParam, currentCompiledLoss), dim=1).unsqueeze(3)
        print('initialSingleEmotionData: ', initialSingleEmotionData)
        print('initialSingleEmotionData size: ', initialSingleEmotionData.size())
        # Smoothen out the discrete map into a probability distribution.
        print('self.allpredictionBins_resampled: ', self.allPredictionBins_resampled)
        probabilityMatrix = self.generalMethods.getProbabilityMatrix(initialSingleEmotionData, self.allParameterBins_resampled, self.allPredictionBins_resampled[0], self.gausParam_STD, self.gausLoss_STD, noise=0.0, applyGaussianFilter=True)
        self.discretePersonalizedMap.append(probabilityMatrix)  # the discretePersonalizedMap list will store the probability matrix

    @staticmethod
    def personalizedMapWeightingFunc(timeDelays, decay_constant):
        # Ebbinghaus forgetting curve.
        return np.exp(-decay_constant * np.asarray(timeDelays))

    def getUpdatedPersonalizedMap(self):
        # Assert the integrity of the state tracking.
        print(f"Length of parameterPath: {len(self.paramStatePath)}")
        print(f'Length of TimePoints: {len(self.timePoints)}')
        print(f'Content of ParameterPath: {self.paramStatePath}')
        print(f'Content of TimePoints: {self.timePoints}')
        print(f"Length of discretePersonalizedMap: {len(self.discretePersonalizedMap)}")
        print(f"Content of discretePersonalizedMap: {self.discretePersonalizedMap}")
        assert len(self.paramStatePath) == len(self.timePoints) == len(self.discretePersonalizedMap), \
            f"The time delays and discrete maps are not the same length. {len(self.paramStatePath)} {len(self.timePoints)} {len(self.discretePersonalizedMap)}"
        # Unpack the temperature-timepoints relation.
        tempTimepoints = np.asarray(self.temperatureTimepoints)
        associatedTempInds = tempTimepoints[:, 1]
        timePoints = tempTimepoints[:, 0]

        # Get the weighting for each discrete temperature-loss pair.
        currentTimeDelays = np.abs(timePoints - timePoints[-1])
        personalizedMapWeights = self.personalizedMapWeightingFunc(currentTimeDelays, self.decayConstant)

        # For each temperature bin.
        for tempIndex in range(self.allNumParameterBins):
            # If the temperature bin has been visited.
            if tempIndex in associatedTempInds:
                tempIndMask = associatedTempInds == tempIndex

                # Normalize the weights per this bin.
                personalizedMapWeights[tempIndMask] = personalizedMapWeights[tempIndMask] / personalizedMapWeights[tempIndMask].sum()

        # Perform a weighted average of all the personalized maps.
        personalizedMap = np.sum(self.discretePersonalizedMap * personalizedMapWeights[:, np.newaxis, np.newaxis], axis=0)

        if self.applyGaussianFilter:
            # Smoothen the personalized map.
            personalizedMap = self.generalMethods.smoothenArray(personalizedMap, sigma=self.gausSTD[::-1])

        # Normalize the personalized map.
        personalizedMap = personalizedMap / personalizedMap.sum()  # Normalize along the temperature axis.

        return personalizedMap

    def initializeFirstPersonalizedMap(self):
        # Initialize a uniform personalized map. No bias.
        uniformMap = np.ones((max([len(paramBins) for paramBins in self.allParameterBins]), max([len(predBins) for predBins in self.allPredictionBins])))
        uniformMap = uniformMap / uniformMap.sum()

        # Store the initial personalized map estimate.
        # self.discretePersonalizedMap.append(uniformMap)
        # self.timePoints.append((0, ))

    # ------------------------ Heuristic Interface ------------------------ #

    def initializeHeuristicMaps(self):
        if self.simulateTherapy:
            print("simulating compiled loss map for heuristic map", self.simulationProtocols.simulatedMapCompiledLoss)
            return self.simulationProtocols.simulatedMapCompiledLoss
        else:
            # Get the real data points.
            initialHeuristicStates = self.empatchProtocols.getTherapyData()

            # Get the heuristic matrix from the simulated points.
            initialHeuristicData = self.compileLossStates(initialHeuristicStates)  # InitialData dim: numPoints, (T, L)
            heuristicMap = self.getProbabilityMatrix(initialHeuristicData)  # Adding Gaussian distributions and normalizing the probability

            return heuristicMap
