# General
import random
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from .generalProtocol import generalProtocol


# make a stupid algorithm

class basicProtocol(generalProtocol):
    def __init__(self, temperatureBounds, tempBinWidth, simulationParameters, personalizedMap=None):
        super().__init__(temperatureBounds, tempBinWidth, simulationParameters)
        # Specific basic protocol parameters
        self.discretePersonalizedMap = [] # store the probability matrix
        self.gausSTD = np.array([0.05, 2.5])  # The standard deviation for the Gaussian distribution
        self.applyGaussianFilter = False  # Whether to apply a Gaussian filter on the discrete maps.
        self.uncertaintyBias = 1  # The bias for uncertainty.
        self.finishedTherapy = False    # Whether the therapy has finished.
        self.numTempsConsider = 2  # Number of temperatures to consider for the next step
        self.percentHeuristic = 1  # The percentage of the heuristic map to use.
        self.heuristicMap = self.initializeHeuristicMaps()


    # ------------------------ Update Parameters ------------------------ #
    def updateTherapyState(self):
        # get the current user state
        currentUserState = self.userStatePath[-1]
        currentUserTemp, currentUserLoss = currentUserState

        # Update temperature towards smaller loss
        newUserTemp = self.updateTemperature(currentUserTemp, currentUserLoss)
        newUserTemp = self.boundNewTemperature(newUserTemp)

        return newUserTemp, self.simulationProtocols.simulatedMap

    def updateTemperature(self, currentUserTemp, currentUserLoss):
        # Define the new temperature update step.
        temperatureStep = self.tempBinWidth   # arbitrary direction step for temperature update

        # If we have not visited the current temperature bin.
        if len(self.userStatePath) <= 1:
            # Randomly move in a direction.
            return currentUserTemp + random.uniform(-temperatureStep, temperatureStep)

        # Look at the previous temperature states.
        previousUserTemp, previousUserLoss = self.userStatePath[-2]

        # If we didn't move temperature last time. This can happen at boundaries.
        if currentUserTemp == previousUserTemp:
            # Randomly move in a direction.
            return currentUserTemp + random.uniform(-temperatureStep, temperatureStep)
        else:
            interpSlope = (currentUserLoss - previousUserLoss) / (currentUserTemp - previousUserTemp)

        if interpSlope < 0:
            return currentUserTemp - temperatureStep
        return currentUserTemp + temperatureStep

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
        uniformMap = np.ones((self.numTempBins, self.numLossBins))
        uniformMap /= uniformMap.sum()
        self.personalizedMap = uniformMap

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

    # ------------------------ initialize simulated Map ------------------------ #
    def initializeHeuristicMaps(self):
        if self.simulateTherapy:
            # Get the simulated data points.
            initialHeuristicStates = self.simulationProtocols.generateSimulatedMap(self.simulationProtocols.numSimulationHeuristicSamples, simulatedMapType=self.simulationProtocols.heuristicMapType)
            initialSimulatedStates = self.simulationProtocols.generateSimulatedMap(self.simulationProtocols.numSimulationTrueSamples, simulatedMapType=self.simulationProtocols.simulatedMapType)
            # initialHeuristicStates dimension: numSimulationHeuristicSamples, (T, PA, NA, SA); 2D array
            # initialSimulatedStates dimension: numSimulationTrueSamples, (T, PA, NA, SA); 2D array

            # Get the simulated matrix from the simulated points.
            initialSimulatedData = self.compileLossStates(initialSimulatedStates)  # initialSimulatedData dimension: numSimulationTrueSamples, (T, L).
            self.simulationProtocols.simulatedMap = self.getProbabilityMatrix(initialSimulatedData)  # Spreading delta function probability.
        else:
            # Get the real data points.
            initialHeuristicStates = self.empatchProtocols.getTherapyData()
            # dimension of initialHeuristicStates: (T, PA, NA, SA); 2D array (we have 6 points for now)
            print('intialHeuristic', initialHeuristicStates)
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





