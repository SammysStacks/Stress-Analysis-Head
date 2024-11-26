# General
import numpy as np
import random
import torch
# Import files.
from .generalTherapyProtocol import generalTherapyProtocol


class basicTherapyProtocol(generalTherapyProtocol):
    def __init__(self, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod):
        super().__init__(initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod)
        # Specific basic protocol parameters
        self.discretePersonalizedMap = [] # store the probability matrix
        self.gausParam_STD = self.gausParameterSTDs  # self.gausParameterSTDs  # The standard deviation for the Gaussian distribution.
        self.gausLoss_STD = torch.tensor([0.0580]) # resampled bin width TODO: need to change it to be more general
        self.uncertaintyBias = 1  # The bias for uncertainty.
        self.finishedTherapy = False    # Whether the therapy has finished.
        self.numParamsConsider = 2  # Number of temperatures to consider for the next step
        self.percentHeuristic = 1  # The percentage of the heuristic map to use.
        self.heuristicMap = self.initializeStartingMaps()

        # resampled bins for the parameter and prediction bins
        self.allParameterBins_resampled, self.allPredictionBins_resampled = self.generalMethods.resampleBins(self.allParameterBins, self.allPredictionBins, eventlySpacedBins=False)

    # ------------------------ Update Parameters ------------------------ #
    def updateTherapyState(self):
        # get the current user state
        currentParam = self.paramStatePath[-1]  # dim should be torch.Size([1, 1, 1, 1]); actual parameter value
        currentCompiledLoss = self.userMentalStateCompiledLoss[-1]  # actual compiled loss value

        # Update temperature towards smaller loss
        newUserParam = self.updateTemperature(currentParam, currentCompiledLoss)
        if self.simulateTherapy:
            return newUserParam, self.simulationProtocols.simulatedMapCompiledLoss
        else:
            return newUserParam, self.simulationProtocols.realSimMapCompiledLoss

    def updateTemperature(self, currentUserParam, currentUserLoss):
        # Define the new temperature update step.
        paramStep = self.parameterBinWidths   # arbitrary direction step for temperature update
        # first movement in the parameter space
        if len(self.paramStatePath) <= 1:
            # Randomly move in a direction.
            paramStepTensor = torch.tensor(random.uniform(-paramStep, paramStep)).view(1, 1, 1, 1)
            return currentUserParam + paramStepTensor

        # get Bin index for next condition comparison
        currentParamBinIndex = self.dataInterface.getBinIndex(self.allParameterBins[0], currentUserParam)
        unboundedCurrentParam = self.unNormalizedAllParameterBins[0][currentParamBinIndex]
        # Look at the previous temperature states.
        previousUserParam = self.paramStatePath[-2]
        previousCompiledLoss = self.userMentalStateCompiledLoss[-2]
        prevParamBinIndex = self.dataInterface.getBinIndex(self.allParameterBins[0], previousUserParam)
        unboundedPrevParam = self.unNormalizedAllParameterBins[0][prevParamBinIndex]

        #TODO: need to double check
        # if torch.rand(1) < 0.1:
        #     # move for larger steps
        #     return currentUserParam + torch.tensor(random.uniform(-10 * paramStep, 10 * paramStep)).view(1, 1, 1, 1)

        # If we didn't move temperature last time. This can happen at boundaries.
        if unboundedCurrentParam == unboundedPrevParam:
            # Randomly move in a direction.
            return currentUserParam + torch.tensor(random.uniform(-paramStep, paramStep)).view(1, 1, 1, 1)
        else:
            interpSlope = (currentUserLoss - previousCompiledLoss) / (currentUserParam - previousUserParam)
            # Determine the direction of paramStep based on the sign of interpSlope
            # if interpSlope > 0:
            #     # If slope is positive, decrease the parameter if the current parameter is greater than the previous, otherwise increase
            #     return currentUserParam - paramStep if currentUserParam > previousUserParam else currentUserParam + paramStep
            # else:
            #     # If slope is negative, increase the parameter if the current parameter is less than the previous, otherwise decrease
            #     return currentUserParam + paramStep if currentUserParam < previousUserParam else currentUserParam - paramStep
            if interpSlope < 0:
                return currentUserParam - paramStep
            return currentUserParam + paramStep

    def trackCurrentState(self, currentUserState):
        # Smoothen out the discrete map into a probability distribution.
        probabilityMatrix = self.getProbabilityMatrix([currentUserState])
        self.discretePersonalizedMap.append(probabilityMatrix)  # the discretePersonalizedMap list will store the probability matrix

    def updatePersonalizedMap(self, currentUserState):
        # Update the personalized user map.
        self.trackCurrentState(currentUserState)

    # ------------------------ Personalized Map prob generation ------------------------ #

    def initializeFirstPersonalizedMap(self):
        # Initialize a uniform personalized map. No bias.
        uniformMap = np.ones((self.allNumParameterBins, self.numPredictionBins))
        uniformMap /= uniformMap.sum()
        self.personalizedMap = uniformMap

    # ------------------------ initialize simulated Map ------------------------ #
    def initializeStartingMaps(self):
        if self.simulateTherapy:
            return self.simulationProtocols.simulatedMapCompiledLoss
        else:
            # TODO: eventually, this should return the fully compiled map with real data (once we finished data collection)
            return self.simulationProtocols.realSimMapCompiledLoss
