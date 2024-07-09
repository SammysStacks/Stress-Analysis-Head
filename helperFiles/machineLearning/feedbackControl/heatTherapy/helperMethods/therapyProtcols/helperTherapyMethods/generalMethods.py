# General
from scipy.ndimage import gaussian_filter
import torch
import numpy as np

# Import helper files.
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.dataInterface.dataInterface import dataInterface


class generalMethods:

    def __init__(self):
        # Initialize helper classes.
        self.dataInterface = dataInterface


    @staticmethod
    def resampleBins(allParameterBins, allPredictionBins, eventlySpacedBins=True):
        maxLengthParamBins = max([len(paramBins) for paramBins in allParameterBins])
        # get the max index of the bins with maxlengthParamBins
        maxParamIndex = max([i for i, paramBins in enumerate(allParameterBins) if len(paramBins) == maxLengthParamBins])
        maxLengthPredBins = max([len(predBins) for predBins in allPredictionBins])
        # get the max index of the bins with maxlengthPredBins
        maxPredIndex = max([i for i, predBins in enumerate(allPredictionBins) if len(predBins) == maxLengthPredBins])
        # assert that all the bins are normalized between 0 and 1
        # assert min(predBins[0] for predBins in allPredictionBins) == 0.0 and max(predBins[-1] for predBins in allPredictionBins) == 1.0, "Prediction bins must be normalized between 0 and 1"
        if len(allParameterBins) == 1:
            if eventlySpacedBins:
                # Resample the bins to be evenly spaced.
                allParameterBins = [np.linspace(0, 1, maxLengthParamBins) for _ in allParameterBins]
                allPredictionBins = [np.linspace(0, 1, maxLengthPredBins) for _ in allPredictionBins]
            else:
                # just duplicate the bins with largest binlength within allParameterBins and allPredictionBins
                allParameterBins = [allParameterBins[maxParamIndex] for _ in allParameterBins]
                allPredictionBins = [allPredictionBins[maxPredIndex] for _ in allPredictionBins]

        return allParameterBins, allPredictionBins



    @staticmethod
    def smoothenArray(deltaFunctionMatrix, sigma):
        return gaussian_filter(deltaFunctionMatrix, sigma=sigma)

    @staticmethod
    def createGaussianArray(inputData, gausMean, gausSTD, torchFlag=True):
        library = torch if torchFlag else None

        xValues = library.arange(len(inputData), dtype=library.float32)
        gaussianArray = library.exp(-0.5 * ((xValues - gausMean) / gausSTD) ** 2)
        gaussianArray = gaussianArray / gaussianArray.sum()  # Normalize the Gaussian array
        
        return gaussianArray

    @staticmethod
    def createGaussianMap(allParameterBins, predictionBins, gausMean, gausSTD):
        # convert input to tensors
        allParameterBins = torch.tensor(allParameterBins).squeeze()
        predictionBins = torch.tensor(predictionBins).squeeze()
        # Generate a grid for Gaussian distribution calculations
        x, y = torch.meshgrid(allParameterBins, predictionBins, indexing='ij')

        # Calculate Gaussian distribution values across the grid
        gaussMatrix = torch.exp(-0.5 * ((x - gausMean[0]) ** 2 / gausSTD[0] ** 2 + (y - gausMean[1]) ** 2 / gausSTD[1] ** 2))
        gaussMatrix = gaussMatrix / gaussMatrix.sum()  # Normalize the Gaussian matrix

        return gaussMatrix

    @staticmethod
    def separateUneven2DArray(inputArray, index):
        return inputArray[index]

    def getProbabilityMatrix(self, initialSingleEmotionData, allParameterBins, singlePredictionBins, gausParamSTD,  gausLossSTD, noise=0.0, applyGaussianFilter=True):
        """Note: single emotion data can be (T, PA), (T, NA), (T, SA), and (T, compiledLoss) corresponding to different types of map we have for heat therapy"""
        # allParameterBins is a 2D array of size (numParameters, numBins)
        probabilityMatrix = torch.zeros((len(allParameterBins[0]), len(singlePredictionBins))) # dim: torch.Size([numParameterBins[0], singlePredictionBins])
        # Calculate the probability matrix.

        for initialDataPoints in initialSingleEmotionData:
            currentUserTemp = initialDataPoints[0] # within loop: torch.Size([1, 1])
            currentUserLoss = initialDataPoints[1] # within loop: torch.Size([1, 1])

            if applyGaussianFilter:
                # Generate a delta function probability.
                tempBinIndex = self.dataInterface.getBinIndex(allParameterBins, currentUserTemp)
                lossBinIndex = self.dataInterface.getBinIndex(singlePredictionBins, currentUserLoss)
                probabilityMatrix[tempBinIndex][lossBinIndex] += 1  # map out bins and fill out with discrete values # parameterbins x lossbins

            else:
                # Generate 2D gaussian matrix.
                gaussianMatrix = self.createGaussianMap(allParameterBins, singlePredictionBins, gausMean=(currentUserTemp, currentUserLoss), gausSTD=(gausParamSTD, gausLossSTD))
                probabilityMatrix += gaussianMatrix  # Add the gaussian map to the matrix

        # gauss data structure change for input
        if gausLossSTD.dim() == 0:
            gausLossSTD = gausLossSTD.unsqueeze(0)

        # Concatenate tensors
        combinedSTD = torch.cat((gausParamSTD, gausLossSTD))

        if applyGaussianFilter:
            # Smoothen the probability matrix.
            probabilityMatrix = self.smoothenArray(probabilityMatrix, sigma=combinedSTD.numpy())
            probabilityMatrix = torch.tensor(probabilityMatrix)
        # Normalize the probability matrix.

        probabilityMatrix += noise * torch.randn(*probabilityMatrix.size())  # Add random noise
        probabilityMatrix = torch.clamp(probabilityMatrix, min=0.0, max=None)  # Ensure no negative probabilities
        probabilityMatrix = probabilityMatrix / probabilityMatrix.sum()
        return probabilityMatrix



