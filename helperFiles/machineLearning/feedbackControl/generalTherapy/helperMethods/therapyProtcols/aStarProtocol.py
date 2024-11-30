# General
import numpy as np
import torch
from .generalTherapyProtocol import generalTherapyProtocol


class aStarTherapyProtocol(generalTherapyProtocol):
    def __init__(self, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapySelection, therapyMethod, learningRate=5):
        super().__init__(initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapySelection, therapyMethod)
        # ================================================Update Parameters=================================================
        self.gausParam_STD = self.gausParameterSTDs  # The standard deviation for the Gaussian distribution.
        self.gausLoss_STD = torch.tensor([0.0580])  # self.gausLossSTDs
        self.discretePersonalizedMap = []  # The discrete personalized map.
        self.learningRate = learningRate  # The learning rate for the therapy.

        # ===================================================Bias Terms=====================================================
        self.percentHeuristic = 1  # The percentage of the heuristic map to use.
        self.explorationBias = 1  # The bias for exploration.
        self.uncertaintyBias = 1  # The bias for uncertainty.
        self.decayConstant = 1 / (2 * 3600)  # The decay constant for the personalized map.

        # ========================================Init heuristic and personal Maps==========================================
        self.heuristicMap = self.initializeHeuristicMaps()  # Estimate on what temperatures you like. Based on population average.
        self.initializeFirstPersonalizedMap()  # list of probability maps.

        # ==========================================Therapy specific Parameters=============================================
        if self.therapySelection == 'Heat':
            # Specific A Star Protocol parameters.
            self.paramBinsVisited = np.full(self.allNumParameterBins, False)
            self.allParameterBins_resampled, self.allPredictionBins_resampled = self.generalMethods.resampleBins(self.allParameterBins, self.allPredictionBins, eventlySpacedBins=False)

        elif self.therapySelection == 'BinauralBeats':
            # Specific A Star Protocol parameters.
            self.paramBinsVisited = np.full(self.allNumParameterBins, False)
            resampledParameterBins_single, self.allPredictionBins_resampled = self.generalMethods.resampleBins(self.allParameterBins[0], self.allPredictionBins, eventlySpacedBins=False)
            self.allParameterBins_resampled = [resampledParameterBins_single, resampledParameterBins_single]

    """Update the therapy parameters"""
    def updateTherapyState(self):
        # Get the current user state.
        currentParam = self.paramStatePath[-1]
        currentCompiledLoss = self.userMentalStateCompiledLoss[-1]
        paramBinIndex = self.dataInterface.getBinIndex(self.allParameterBins_resampled, currentParam)
        self.paramBinsVisited[paramBinIndex] = True

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
        #newUserParam = self.boundNewTemperature(newUserParam, bufferZone=1)
        newUserParam = torch.tensor(newUserParam).view(1, 1, 1, 1)

        # Bound the parameter
        newUserParam = torch.clamp(newUserParam, min=0, max=1)
        return newUserParam, (benefitFunction, self.heuristicMap, personalizedMap, probabilityMap)

    """Update the therapy parameters for Binaural Beats"""
    def updateTherapyState3D(self):
        # Get the current user state.
        currentParam = self.paramStatePath[-1]
        currentCompiledLoss = self.userMentalStateCompiledLoss[-1]

        paramBinIndex_1 = self.dataInterface.getBinIndex(self.allParameterBins_resampled[0], currentParam[:, 0])
        paramBinIndex_2 = self.dataInterface.getBinIndex(self.allParameterBins_resampled[1], currentParam[:, 1])
        self.paramBinsVisited[paramBinIndex_1, paramBinIndex_2] = True

        # Update the personalized user map.
        self.trackCurrentState(currentParam, currentCompiledLoss)  # Keep track of each discrete params-loss pair.
        personalizedMap = self.getUpdatedPersonalizedMaps3D()  # Convert the discrete map to a probability distribution.

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
        newUserParam = torch.tensor(newUserParam).view(1, 2, 1, 1)

        # Bound the parameter
        newUserParam = torch.clamp(newUserParam, min=0, max=1)
        return newUserParam, (benefitFunction, self.heuristicMap, personalizedMap, probabilityMap)

    """Data processing"""
    def normalizeParameter(self, newUserParam):
        # Normalize the parameter.
        newUserParam = (newUserParam - self.initialParameterBounds[0][0]) / (self.initialParameterBounds[0][1] - self.initialParameterBounds[0][0])
        return newUserParam

    #TODO: to be deleted, double check not needed
    def findNewTemperature(self, currentUserState, gradientDirection):
        # Unpack the current user state.
        currentUserTemp, currentUserLoss = currentUserState

        # Determine a direction based on temperature gradient at current loss
        tempBinIndex = self.getBinIndex(self.temp_bins, currentUserTemp)
        tempGradient = gradientDirection[tempBinIndex]

        return tempGradient

    # ================================================Update Parameters=================================================

    def findOptimalDirection(self, probabilityMap, currentParam, currentCompiledLoss):

        # Calculate benefits/loss of exploring/moving.
        potentialLossBenefit = np.asarray(self.allPredictionBins_resampled[0])

        if self.therapySelection == 'Heat':
            probabilityMap = probabilityMap / probabilityMap.sum(axis=1)[:, np.newaxis]  # Normalize the probability map.
            potentialRewards = potentialLossBenefit[np.newaxis, :]  # Add 1 dimension from potentialLossBenefit
            expectedRewards = probabilityMap * potentialRewards

            # Find the best temperature bin index in the rewards.
            expectedRewardAtTemp = expectedRewards.sum(axis=1)
            bestTempBinIndex = torch.argmin(expectedRewardAtTemp).item()  # Minimizing losses

            # Convert parameterBinWidths to numpy array or list
            parameterBinWidths = self.parameterBinWidths.numpy() if isinstance(self.parameterBinWidths, torch.Tensor) else self.parameterBinWidths  # Center of the bin
            return self.allParameterBins_resampled[0][bestTempBinIndex] + parameterBinWidths / 2, expectedRewards

        elif self.therapySelection == 'BinauralBeats':
            # Normalize probabilityMap along the last axis (dim=2)
            probabilityMap = probabilityMap / probabilityMap.sum(dim=2, keepdim=True)
            potentialRewards = potentialLossBenefit[np.newaxis, np.newaxis, :]  # add 2 dimension from potentialLossBenefit
            expectedRewards = probabilityMap * potentialRewards

            # Find the best temperature bin index in the rewards.
            expectedRewardAtParam = expectedRewards.sum(axis=2)

            minVal = float('inf')  # Initialize with a large value
            twoDimensionIndex = None

            # Loop through each element in the 2D tensor
            for i in range(len(expectedRewardAtParam)):  # Iterate over rows
                for j in range(len(expectedRewardAtParam[i])):  # Iterate over columns
                    if expectedRewardAtParam[i][j] < minVal:  # Check for a new minimum
                        minVal = expectedRewardAtParam[i][j]
                        twoDimensionIndex = (i, j)

            # Convert parameterBinWidths to numpy array or list
            parameterBinWidths = self.parameterBinWidths.numpy() if isinstance(self.parameterBinWidths, torch.Tensor) else self.parameterBinWidths  # used to place the temperature at the center of the bin

            # Calculate the new parameters based on the best indices

            newUserParam1 = self.allParameterBins_resampled[0][0][twoDimensionIndex[0]] + parameterBinWidths[0] / 2
            newUserParam2 = self.allParameterBins_resampled[1][0][twoDimensionIndex[1]] + parameterBinWidths[1] / 2
            userParams = [newUserParam1.item(), newUserParam2.item()]

            # Return the new parameters and the expected rewards
            return userParams, expectedRewards

        # # Compute the gradient.
        # potentialTemperatureRewards = np.gradient(potentialTemperatureRewards)  # Dimension: 2, allNumParameterBins, numPredictionBins

    """Update the map weights alpha """
    def updateAlpha(self):
        # Calculate the percentage of the temperature bins visited.
        percentConfidence = self.paramBinsVisited.sum() / len(self.paramBinsVisited)

        # Update the confidence flags.
        self.percentHeuristic = min(self.percentHeuristic, 1 - percentConfidence) - 0.001
        self.percentHeuristic = min(1.0, max(0.0, self.percentHeuristic))

        # Update the bias terms.
        self.explorationBias = self.percentHeuristic
        self.uncertaintyBias = self.percentHeuristic

    # ============================================Personalization Interface=============================================
    def trackCurrentState(self, currentParam, currentCompiledLoss):
        if self.therapySelection == 'Heat':
            initialSingleEmotionData = torch.cat((currentParam, currentCompiledLoss), dim=1) # dim: torch.Size([1, 2, 1, 1])

            # Smoothen out the discrete map into a probability distribution.
            probabilityMatrix = self.generalMethods.getProbabilityMatrix(initialSingleEmotionData, self.allParameterBins_resampled, self.allPredictionBins_resampled[0], self.gausParam_STD, self.gausLoss_STD, noise=0.1, applyGaussianFilter=True)
            self.discretePersonalizedMap.append(probabilityMatrix)  # The discretePersonalizedMap list will store the probability matrix

        elif self.therapySelection == 'BinauralBeats':
            initialSingleEmotionData = torch.cat((currentParam, currentCompiledLoss), dim=1)  # dim: torch.Size([1, 3, 1, 1])
            probabilityMatrix = self.generalMethods.get3DProbabilityMatrix(initialSingleEmotionData, self.allParameterBins_resampled, self.allPredictionBins_resampled[0], self.gausParam_STD, self.gausLoss_STD, noise=0.1,
                                                                         applyGaussianFilter=True) # torch.Size([paramBinLen, paramBinLen, PredictBinLen])
            self.discretePersonalizedMap.append(probabilityMatrix)  # The discretePersonalizedMap list will store the probability matrix

    @staticmethod
    def personalizedMapWeightingFunc(timeDelays, decay_constant):
        # Ebbinghaus forgetting curve.
        return np.exp(-decay_constant * np.asarray(timeDelays))

    def getUpdatedPersonalizedMap(self):
        # Assert the integrity of the state tracking.
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

    """For BinauralBeats Specific"""
    def getUpdatedPersonalizedMaps3D(self):
        # Assert the integrity of the state tracking.
        assert len(self.paramStatePath) == len(self.timepoints) == len(self.discretePersonalizedMap), \
            f"The time delays and discrete maps are not the same length. {len(self.paramStatePath)} {len(self.timepoints)} {len(self.discretePersonalizedMap)}"

        # Unpack the parameter-timepoints relation.
        associatedParamInd1 = []
        associatedParamInd2 = []
        associatedTimePoints = np.asarray(self.timepoints)

        for idx in range(len(self.paramStatePath)):
            currentParam = self.paramStatePath[idx]
            param_values = currentParam.squeeze().numpy()  # Shape should be (2,)
            print('currentParam', currentParam)
            print('param_values', param_values)

            paramBinIndex_1 = self.dataInterface.getBinIndex(self.allParameterBins_resampled[0], param_values[0])
            paramBinIndex_2 = self.dataInterface.getBinIndex(self.allParameterBins_resampled[1], param_values[1])
            associatedParamInd1.append(paramBinIndex_1)
            associatedParamInd2.append(paramBinIndex_2)

        # Get the weighting for each discrete parameter-loss pair.
        currentTimeDelays = np.abs(associatedTimePoints - associatedTimePoints[-1])
        personalizedMapWeights = self.personalizedMapWeightingFunc(currentTimeDelays, self.decayConstant)

        # For each parameter bin.
        for paramIndex1 in range(self.allNumParameterBins[0]):
            for paramIndex2 in range(self.allNumParameterBins[1]):
                # If the parameter bin has been visited.

                if (paramIndex1, paramIndex2) in zip(associatedParamInd1, associatedParamInd2):
                    paramIndexMask = np.logical_and(
                        np.isin(associatedParamInd1, paramIndex1),
                        np.isin(associatedParamInd2, paramIndex2)
                    )  # Mask for bins visited in both dimensions
                    # Normalize the weights per this bin.
                    personalizedMapWeights[paramIndexMask] = personalizedMapWeights[paramIndexMask] / personalizedMapWeights[paramIndexMask].sum()

        # Perform a weighted average of all the personalized maps.
        personalizedMap = np.sum(self.discretePersonalizedMap * personalizedMapWeights[:, np.newaxis, np.newaxis], axis=0)

        # Apply Gaussian smoothing if needed.
        if self.applyGaussianFilter:
            combinedSTD = torch.cat((self.gausParam_STD, self.gausLoss_STD))
            # Smoothen the personalized map.
            personalizedMap = torch.tensor(personalizedMap)
            personalizedMap = self.generalMethods.smoothenArray(personalizedMap.numpy(), sigma=combinedSTD.numpy())

        # Normalize the personalized map.
        personalizedMap = personalizedMap / np.sum(personalizedMap)  # Normalize along the parameter axes.
        # Convert back to tensor.
        personalizedMap = torch.tensor(personalizedMap)
        return personalizedMap

    """Intializations of the personalized map"""
    def initializeFirstPersonalizedMap(self):
        # Initialize a uniform personalized map. No bias.
        uniformMap = np.ones((max([len(paramBins) for paramBins in self.allParameterBins]), max([len(predBins) for predBins in self.allPredictionBins])))
        uniformMap = uniformMap / uniformMap.sum()

        # Store the initial personalized map estimate.
        # self.discretePersonalizedMap.append(uniformMap)
        # self.timepoints.append((0, ))

    # ==============================================Heuristic Interface=================================================
    def initializeHeuristicMaps(self):
        if self.simulateTherapy:
            return self.simulationProtocols.simulatedMapCompiledLoss
        else:
            # Initialize a heuristic map.
            return self.simulationProtocols.realSimMapCompiledLoss

    def updateEmotionPredState(self, userName, currentTimePoints, currentParamValues, currentEmotionStates):
        if currentTimePoints is None or currentParamValues is None or currentEmotionStates is None:
            return

        # Track the user state and time delay.
        self.timepoints.append(currentTimePoints)  # timepoints: list of tensor: [tensor(0)]
        self.paramStatePath.append(currentParamValues)  # self.paramStatePath: list of tensor: [torch.Size([1, 1, 1, 1])
        self.userMentalStatePath.append(currentEmotionStates) # emotionstates: list of tensor: torch.Size([1, 3, 1, 1])
        # Calculate the initial user loss.
        compiledLoss = self.dataInterface.calculateCompiledLoss(currentEmotionStates[-1])  # Compile the loss state for the current emotion state; torch.Size([1, 1, 1, 1])
        self.userMentalStateCompiledLoss.append(compiledLoss)  # List of tensor torch.Size([1, 1, 1, 1])
        self.userName.append(userName) # list: username
        userParamBinIndex = self.dataInterface.getBinIndex(self.allParameterBins, currentParamValues)
        userParam = self.unNormalizedAllParameterBins[0][userParamBinIndex]  # Bound the initial Param (1D)
        userParam = self.boundNewTemperature(userParam, bufferZone=0.01)  # Bound the initial Param (1D)
        self.unNormalizedParameter.append(userParam)  # List of tensor torch.Size([1, 1, 1, 1])



