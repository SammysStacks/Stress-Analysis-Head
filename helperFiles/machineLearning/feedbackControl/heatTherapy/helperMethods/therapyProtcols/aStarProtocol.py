# General
import numpy as np
import torch
from .generalTherapyProtocol import generalTherapyProtocol


class aStarTherapyProtocol(generalTherapyProtocol):
    def __init__(self, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod, learningRate=5):
        super().__init__(initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod)
        # Define update parameters.
        self.gausParam_STD = self.gausParameterSTDs # The standard deviation for the Gaussian distribution.
        self.gausLoss_STD = torch.tensor([0.0580]) # self.gausLossSTDs
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
        currentParam = self.paramStatePath[-1] # dim should be torch.Size([1, 1, 1, 1]); actual parameter value
        currentCompiledLoss = self.userMentalStateCompiledLoss[-1] # actual compiled loss value

        # Update the temperatures visited.
        paramBinIndex = self.dataInterface.getBinIndex(self.allParameterBins_resampled, currentParam)
        self.paramBinsVisited[paramBinIndex] = True  # documents the temperature visited by indexing the temperature bin and set the value to true

        # Update the personalized user map.
        self.trackCurrentState(currentParam, currentCompiledLoss)  # Keep track of each discrete temperature-loss pair.
        personalizedMap = self.getUpdatedPersonalizedMap()  # Convert the discrete map to a probability distribution.

        # Get the current time point.
        timePoint = self.timepoints[-1]

        # Combine the heuristic and personalized maps and update the weighting.
        probabilityMap = self.percentHeuristic * self.heuristicMap + (1 - self.percentHeuristic) * personalizedMap
        self.updateAlpha()

        # Find the best temperature in the gradient direction.
        newUserParam, benefitFunction = self.findOptimalDirection(probabilityMap, currentParam, currentCompiledLoss)

        newUserParam = newUserParam + self.uncertaintyBias * np.random.normal(loc=0, scale=0.5)  # Add noise to the gradient.
        # Calculate the new temperature.
        #newUserParam = self.boundNewTemperature(newUserParam, bufferZone=1) # newUserParam = torch.Size([1, 1, 1, 1])
        newUserParam = torch.tensor(newUserParam).view(1, 1, 1, 1) # actual userParam, not probability
        # bound the parameter
        newUserParam = torch.clamp(newUserParam, min=0, max=1)
        return newUserParam, (benefitFunction, self.heuristicMap, personalizedMap, probabilityMap)

    def normalizeParameter(self, newUserParam):
        """Used to get the normalized temperature given the actual temperature in C not needed i feel"""
        # Normalize the parameter.
        newUserParam = (newUserParam - self.initialParameterBounds[0][0]) / (self.initialParameterBounds[0][1] - self.initialParameterBounds[0][0])
        return newUserParam

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
        potentialLossBenefit = np.asarray(self.allPredictionBins_resampled[0]) # doesn't matter which prediction bin, they are essentially the same.
        probabilityMap = probabilityMap / probabilityMap.sum(axis=1)[:, np.newaxis]  # Normalize the probability map.

        # Calculate the expected rewards.
        potentialRewards = potentialLossBenefit[np.newaxis, :] # add 1 dimension from potentialLossBenefit
        expectedRewards = probabilityMap * potentialRewards
        # Find the best temperature bin index in the rewards.
        expectedRewardAtTemp = expectedRewards.sum(axis=1)

        bestTempBinIndex = torch.argmin(expectedRewardAtTemp).item() # minimizing losses, so lower values are better

        # Convert parameterBinWidths to numpy array or list
        parameterBinWidths = self.parameterBinWidths.numpy() if isinstance(self.parameterBinWidths, torch.Tensor) else self.parameterBinWidths # used to place the temperature at the center of the bin

        return self.allParameterBins_resampled[0][bestTempBinIndex] + parameterBinWidths / 2, expectedRewards

        # # Compute the gradient.
        # potentialTemperatureRewards = np.gradient(potentialTemperatureRewards)  # Dimension: 2, allNumParameterBins, numPredictionBins

    def updateAlpha(self):
        # Calculate the percentage of the temperature bins visited.
        percentConfidence = self.paramBinsVisited.sum() / len(self.paramBinsVisited)

        # Update the confidence flags.
        self.percentHeuristic = min(self.percentHeuristic, 1 - percentConfidence) - 0.001
        self.percentHeuristic = min(1.0, max(0.0, self.percentHeuristic))
        print('self.percentHeuristic:', self.percentHeuristic)

        # Update the bias terms.
        self.explorationBias = self.percentHeuristic  # TODO
        self.uncertaintyBias = self.percentHeuristic  # TODO

    # ------------------------ Personalization Interface ------------------------ #

    def trackCurrentState(self, currentParam, currentCompiledLoss):

        initialSingleEmotionData = torch.cat((currentParam, currentCompiledLoss), dim=1) # dim: torch.Size([1, 2, 1, 1])
        # Smoothen out the discrete map into a probability distribution.
        probabilityMatrix = self.generalMethods.getProbabilityMatrix(initialSingleEmotionData, self.allParameterBins_resampled, self.allPredictionBins_resampled[0], self.gausParam_STD, self.gausLoss_STD, noise=0.1, applyGaussianFilter=True)
        self.discretePersonalizedMap.append(probabilityMatrix)  # the discretePersonalizedMap list will store the probability matrix


    @staticmethod
    def personalizedMapWeightingFunc(timeDelays, decay_constant):
        # Ebbinghaus forgetting curve.
        return np.exp(-decay_constant * np.asarray(timeDelays))

    def getUpdatedPersonalizedMap(self):
        # Assert the integrity of the state tracking.
        # print(f"Length of parameterPath: {len(self.paramStatePath)}")
        # print(f'Length of TimePoints: {len(self.timepoints)}')
        # print('self.ParamStatePath: ', self.paramStatePath)
        # print(f'Content of ParameterPath: {self.paramStatePath}')
        # print(f'Content of TimePoints: {self.timepoints}')
        # print(f"Length of discretePersonalizedMap: {len(self.discretePersonalizedMap)}")
        # print(f"Content of discretePersonalizedMap: {self.discretePersonalizedMap}")
        # print(f"Type of three parameters for comparison: {type(self.paramStatePath)}, {type(self.timepoints)}, {type(self.discretePersonalizedMap)}")
        assert len(self.paramStatePath) == len(self.timepoints) == len(self.discretePersonalizedMap), \
            f"The time delays and discrete maps are not the same length. {len(self.paramStatePath)} {len(self.timepoints)} {len(self.discretePersonalizedMap)}"

        # Unpack the temperature-timepoints relation.
        associatedParamInd = []
        associatedParams = np.asarray(self.paramStatePath).flatten()
        associatedTimePoints = np.asarray(self.timepoints)
        for i in range(len(associatedTimePoints)):
            associatedParamInd.append(self.dataInterface.getBinIndex(self.allParameterBins_resampled, associatedParams[i]))

        # Get the weighting for each discrete temperature-loss pair.
        currentTimeDelays = np.abs(associatedTimePoints - associatedTimePoints[-1])
        personalizedMapWeights = self.personalizedMapWeightingFunc(currentTimeDelays, self.decayConstant)

        # For each parameter bin.
        for paramIndex in range(self.allNumParameterBins[0]):
            # If the temperature bin has been visited.
            if paramIndex in associatedParamInd:
                paramIndexMask = np.isin(associatedParamInd, paramIndex) # only output a list of boolean values dim = numParambins
                # Normalize the weights per this bin.
                personalizedMapWeights[paramIndexMask] = personalizedMapWeights[paramIndexMask] / personalizedMapWeights[paramIndexMask].sum()

        # Perform a weighted average of all the personalized maps.
        personalizedMap = np.sum(self.discretePersonalizedMap * personalizedMapWeights[:, np.newaxis], axis=0)
        if self.applyGaussianFilter:
            combinedSTD = torch.cat((self.gausParam_STD, self.gausLoss_STD))
            # Smoothen the personalized map.
            personalizedMap = torch.tensor(personalizedMap)

            personalizedMap = self.generalMethods.smoothenArray(personalizedMap.squeeze(), sigma=combinedSTD.numpy())

        # Normalize the personalized map.
        personalizedMap = personalizedMap / personalizedMap.sum()  # Normalize along the temperature axis.
        # convert to torch tensor
        personalizedMap = torch.tensor(personalizedMap)
        return personalizedMap

    def initializeFirstPersonalizedMap(self):
        # Initialize a uniform personalized map. No bias.
        uniformMap = np.ones((max([len(paramBins) for paramBins in self.allParameterBins]), max([len(predBins) for predBins in self.allPredictionBins])))
        uniformMap = uniformMap / uniformMap.sum()

        # Store the initial personalized map estimate.
        # self.discretePersonalizedMap.append(uniformMap)
        # self.timepoints.append((0, ))

    # ------------------------ Heuristic Interface ------------------------ #

    def initializeHeuristicMaps(self):
        if self.simulateTherapy:
            #print("simulating compiled loss map for heuristic map", self.simulationProtocols.simulatedMapCompiledLoss)
            return self.simulationProtocols.simulatedMapCompiledLoss
        else:
            # Initialize a heuristic map.
            return self.simulationProtocols.realSimMapCompiledLoss

    def updateEmotionPredState(self, userName, currentTimePoints, currentParamValues, currentEmotionStates):
        if currentTimePoints is None or currentParamValues is None or currentEmotionStates is None:
            return
        # Track the user state and time delay.

        self.timepoints.append(currentTimePoints) # timepoints: list of tensor: [tensor(0)]
        self.paramStatePath.append(currentParamValues) # self.paramStatePath: list of tensor: [torch.Size([1, 1, 1, 1])
        self.userMentalStatePath.append(currentEmotionStates) # emotionstates: list of tensor: torch.Size([1, 3, 1, 1])
        # Calculate the initial user loss.
        compiledLoss = self.dataInterface.calculateCompiledLoss(currentEmotionStates[-1])  # compile the loss state for the current emotion state; torch.Size([1, 1, 1, 1])
        self.userMentalStateCompiledLoss.append(compiledLoss) #  list of tensor torch.Size([1, 1, 1, 1])
        self.userName.append(userName) # list: username
        userParamBinIndex = self.dataInterface.getBinIndex(self.allParameterBins, currentParamValues)
        userParam = self.unNormalizedAllParameterBins[0][userParamBinIndex]  # bound the initial temperature (1D)
        userParam = self.boundNewTemperature(userParam, bufferZone=0.01)  # bound the initial temperature (1D)
        self.unNormalizedParameter.append(userParam) # list of tensor torch.Size([1, 1, 1, 1])



