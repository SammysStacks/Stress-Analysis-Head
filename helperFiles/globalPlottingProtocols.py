import math

import matplotlib.pyplot as plt
import shutil
import os

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class globalPlottingProtocols:

    def __init__(self, interactivePlots=True):
        # Setup matplotlib
        self.hpcFlag = 'HPC' in modelConstants.userInputParams['deviceListed']
        self.baseFolderName = "_basePlots/"
        if interactivePlots: plt.ion()
        plt.rcdefaults()

        # Specify the color order.
        self.lightColors = ["#F17FB1", "#5DCBF2", "#B497C9", "#90D6AD", "#FFC162", '#6f4a1f', "#231F20"]  # Red, Blue, Purple, Green, Orange, Brown, Grey
        self.darkColors = ["#F3757A", "#489AD4", "#7E71B4", "#50BC84", "#F9A770", '#4c3007', "#4A4546"]  # Red, Blue, Purple, Green, Orange, Brown, Grey
        self.blackColor = "#231F20"

        # Set the saving folder
        self.baseSavingDataFolder = None
        self.saveDataFolder = None
        self.datasetName = None

    def setSavingFolder(self, baseSavingDataFolder, stringID, datasetName):
        self.baseSavingDataFolder = baseSavingDataFolder + self.baseFolderName
        self.saveDataFolder = baseSavingDataFolder + stringID
        self.datasetName = datasetName

        if baseSavingDataFolder:
            self._createFolder(self.baseSavingDataFolder)
            if stringID: self._createFolder(self.saveDataFolder)

    @staticmethod
    def getRowsCols(numModuleLayers, combineSharedLayers=False):
        numSpecificEncoderLayers = modelConstants.userInputParams['numSpecificEncoderLayers']
        numSharedEncoderLayers = modelConstants.userInputParams['numSharedEncoderLayers']
        nCols = 2 + math.log2(modelConstants.userInputParams['encodedDimension'] // modelConstants.userInputParams['minWaveletDim'])  # TODO: Change this to a more general formula!
        nRows = numSpecificEncoderLayers + (1 if combineSharedLayers else numSharedEncoderLayers)

        return nRows, nCols

    @staticmethod
    def _createFolder(filePath):
        if filePath: os.makedirs(os.path.dirname(filePath), exist_ok=True)

    @staticmethod
    def clearFigure(fig=None, legend=None, showPlot=True):
        if showPlot: plt.show()  # Ensure the plot is displayed

        # Clear and close the figure/legend if provided
        if legend is not None: legend.remove()
        if fig: plt.close(fig)
        else: plt.close('all')

    def displayFigure(self, saveFigureLocation, saveFigureName, baseSaveFigureName=None, fig=None, showPlot=True, clearFigure=True):
        self._createFolder(self.saveDataFolder + saveFigureLocation)
        fig = fig or plt.gcf()

        # Save to base location if specified
        if baseSaveFigureName is not None:
            base_path = os.path.join(self.baseSavingDataFolder, f"{self.datasetName} {baseSaveFigureName[:1].upper()}{baseSaveFigureName[1:]}")
            fig.savefig(base_path, transparent=True, dpi=300, format='pdf')

            # Copy the saved figure to the second location
            if saveFigureName is not None: shutil.copy(base_path, os.path.join(self.saveDataFolder, f"{saveFigureLocation}{saveFigureName.lower()}"))
        else: fig.savefig(os.path.join(self.saveDataFolder, f"{saveFigureLocation}{saveFigureName[:1].upper()}{saveFigureName[1:]}"), transparent=True, dpi=300, format='pdf')

        if clearFigure: self.clearFigure(fig=fig, legend=None, showPlot=showPlot)  # Clear the figure after saving
        elif showPlot: plt.show()
        plt.close(fig)
