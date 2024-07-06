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
        timePoint = self.timePoints[-1]

        # Combine the heuristic and personalized maps and update the weighting.
        probabilityMap = self.percentHeuristic * self.heuristicMap + (1 - self.percentHeuristic) * personalizedMap
        self.updateAlpha()

        # Find the best temperature in the gradient direction.
        newUserParam, benefitFunction = self.findOptimalDirection(probabilityMap, currentParam, currentCompiledLoss)
        print('newUserParam: ', newUserParam)
        print('beneiftFunction: ', benefitFunction)
        newUserParam = newUserParam + self.uncertaintyBias * np.random.normal(loc=0, scale=0.5)  # Add noise to the gradient.
        # deltaTemp = self.findNewTemperature(currentUserState, gradientDirection)
        # deltaTemp = deltaTemp + self.uncertaintyBias * np.random.normal(loc=0, scale=0.5)  # Add noise to the gradient.

        print('parameter bounds',  self.initialParameterBounds)
        # Calculate the new temperature.
        newUserParam = self.boundNewTemperature(newUserParam, bufferZone=1)
        print('passed counter!@#!@#!@#!232')
        return newUserParam, (benefitFunction, self.heuristicMap, personalizedMap, self.simulationProtocols.simulatedMapCompiledLoss)

    def boundNewTemperature(self, newUserParam, bufferZone=0.01):
        # Bound the new temperature.
        # TODO: current implementation only for heat therapy (1D), so we extract the parameter bounds at the 1st dimension. For music therapy, we need both dimensions
        newUserTemp = max((self.initialParameterBounds[0][0]).numpy() + bufferZone, min((self.initialParameterBounds[0][1]).numpy() - bufferZone, newUserParam))
        return newUserTemp

    def findNewTemperature(self, currentUserState, gradientDirection):
        # Unpack the current user state.
        currentUserTemp, currentUserLoss = currentUserState

        # Determine a direction based on temperature gradient at current loss
        tempBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)
        tempGradient = gradientDirection[tempBinIndex]

        return tempGradient

    # ------------------------ Update Parameters ------------------------ #

    def findOptimalDirection(self, probabilityMap, currentParam, currentCompiledLoss):
        # Calculate benefits/loss of exploring/moving.
        potentialLossBenefit = np.array(self.allPredictionBins_resampled[0]) # doesn't matter which prediction bin, they are essentially the same.
        probabilityMap = probabilityMap / probabilityMap.sum(axis=1)[:, np.newaxis]  # Normalize the probability map.

        # Calculate the expected rewards.
        print('potnetialLossBenefit: ', potentialLossBenefit)
        potentialRewards = potentialLossBenefit[np.newaxis, :]
        expectedRewards = probabilityMap * potentialRewards

        # Find the best temperature bin index in the rewards.
        expectedRewardAtTemp = expectedRewards.sum(axis=1)  # Normalize across temperature bins.
        bestTempBinIndex = torch.argmin(expectedRewardAtTemp).item()
        print('bestTempBinIndex: ', bestTempBinIndex)
        print('self.allParameterBins_resampled: ', self.allParameterBins_resampled)
        print('self.allParameterBins_resampled[0]: ', self.allParameterBins_resampled[0])
        print('self.allParametersBins_resampled[0][bestTempBinIndex]: ', self.allParameterBins_resampled[0][bestTempBinIndex])
        print('expectedRewards', expectedRewards)
        # Convert parameterBinWidths to numpy array or list
        parameterBinWidths = self.parameterBinWidths.numpy() if isinstance(self.parameterBinWidths, torch.Tensor) else self.parameterBinWidths
        print('self.parameterBinWidths: ', self.parameterBinWidths)
        #TODO: Note current instrumentation is for heat therapy, which allParameterBins_resampled is a 2D array but only 1st index is used
        return self.allParameterBins_resampled[0][bestTempBinIndex] + parameterBinWidths / 2, expectedRewards

        # # Compute the gradient.
        # potentialTemperatureRewards = np.gradient(potentialTemperatureRewards)  # Dimension: 2, allNumParameterBins, numPredictionBins

    def updateAlpha(self):
        # Calculate the percentage of the temperature bins visited.
        percentConfidence = self.paramBinsVisited.sum() / len(self.paramBinsVisited)

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
        print('self.ParamStatePath: ', self.paramStatePath)
        print(f'Content of ParameterPath: {self.paramStatePath}')
        print(f'Content of TimePoints: {self.timePoints}')
        print(f"Length of discretePersonalizedMap: {len(self.discretePersonalizedMap)}")
        print(f"Content of discretePersonalizedMap: {self.discretePersonalizedMap}")
        print(f"Type of three parameters for comparison: {type(self.paramStatePath)}, {type(self.timePoints)}, {type(self.discretePersonalizedMap)}")
        assert len(self.paramStatePath) == len(self.timePoints) == len(self.discretePersonalizedMap), \
            f"The time delays and discrete maps are not the same length. {len(self.paramStatePath)} {len(self.timePoints)} {len(self.discretePersonalizedMap)}"

        # Unpack the temperature-timepoints relation.
        associatedParamInds = np.asarray(self.paramStatePath)
        associatedTimePoints = np.asarray(self.timePoints)

        # Get the weighting for each discrete temperature-loss pair.
        currentTimeDelays = np.abs(associatedTimePoints - associatedTimePoints[-1])
        personalizedMapWeights = self.personalizedMapWeightingFunc(currentTimeDelays, self.decayConstant)

        # For each temperature bin.
        for paramIndex in range(self.allNumParameterBins[0]):
            # If the temperature bin has been visited.
            if paramIndex in associatedParamInds:
                paramIndexMask = associatedParamInds == paramIndex

                # Normalize the weights per this bin.
                personalizedMapWeights[paramIndexMask] = personalizedMapWeights[paramIndexMask] / personalizedMapWeights[paramIndexMask].sum()

        # Perform a weighted average of all the personalized maps.
        personalizedMap = np.sum(self.discretePersonalizedMap * personalizedMapWeights[:, np.newaxis, np.newaxis], axis=0)

        if self.applyGaussianFilter:
            combinedSTD = torch.cat((self.gausParam_STD, self.gausLoss_STD))
            # Smoothen the personalized map.
            print('personalizedMap: ', personalizedMap)
            print('personalizedMap size: ', personalizedMap.squeeze(0).shape)
            print('combinedSTD: ', combinedSTD)
            personalizedMap = self.generalMethods.smoothenArray(personalizedMap.squeeze(0), sigma=combinedSTD.numpy())

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
