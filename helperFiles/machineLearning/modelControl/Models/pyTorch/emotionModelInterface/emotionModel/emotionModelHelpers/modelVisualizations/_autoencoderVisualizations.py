# General
import matplotlib.pyplot as plt
import numpy as np

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class autoencoderVisualizations(globalPlottingProtocols):

    def __init__(self, saveDataFolder):
        super(autoencoderVisualizations, self).__init__()
        # General parameters
        self.saveDataFolder = None

        # Set the location for saving the models.
        self.setSavingFolder(saveDataFolder)

    def setSavingFolder(self, saveDataFolder):
        self.saveDataFolder = saveDataFolder

    # ---------------------------------------------------------------------- #
    # --------------------- Visualize Model Parameters --------------------- #

    def plotSignalComparison(self, originalSignal, comparisonSignal, epoch, plotTitle, numSignalPlots=1):
        """ originalSignal dimension: batchSize, numSignals, numTotalPoints """
        # Assert the integrity of the incoming data
        assert originalSignal.shape[0:2] == comparisonSignal.shape[0:2], f"{originalSignal.shape} {comparisonSignal.shape}"

        # Extract the shapes of the data
        batchSize, numSignals, numTotalPoints = originalSignal.shape
        batchSize, numSignals, numEncodedPoints = comparisonSignal.shape

        if batchSize == 0: return None

        batchInd = 0
        # For each signal
        for signalInd in range(numSignals):
            # Plot both the signals alongside each other.
            plt.plot(originalSignal[batchInd, signalInd], 'k', marker='o', linewidth=2)
            plt.plot(np.linspace(0, numTotalPoints, numEncodedPoints), comparisonSignal[batchInd, signalInd], 'tab:red', marker='o', linewidth=1)

            # Format the plotting      
            plt.ylabel("Arbitrary Axis (AU)")
            plt.xlabel("Points")
            plt.title(plotTitle)

            # Save the plot
            if self.saveDataFolder: self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs = {epoch} signalInd = {signalInd}.pdf")
            else: self.clearFigure()

            # There are too many signals to plot.
            if signalInd + 1 == numSignalPlots: break

    def plotAllSignalComparisons(self, distortedSignals, reconstructedDistortedSignals, trueSignal, epoch, signalInd, plotTitle):
        numSignals, numTotalPoints = reconstructedDistortedSignals.shape
        alphas = np.linspace(0.1, 1, numSignals)

        # Plot all signals in 'distortedSignals'
        for i in range(numSignals):
            plt.plot(distortedSignals[i], '-', color='k', alpha=alphas[i], linewidth=1, markersize=2, zorder=0)
            plt.plot(trueSignal, 'o', color='tab:blue', linewidth=1, markersize=2, zorder=10)
            plt.plot(reconstructedDistortedSignals[i], '-', color='tab:red', linewidth=1, markersize=2, alpha=alphas[i], zorder=5)

        plt.title(plotTitle + " at Epoch " + str(epoch))
        plt.xlabel('Time (Seconds)')
        plt.ylabel("Arbitrary Axis (AU)")
        plt.legend(['Noisy Signal', 'True Signal', 'Reconstructed Signal'], loc='best')
        # Save the plot
        if self.saveDataFolder: self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs = {epoch} signalInd{signalInd}.pdf")
        else: self.clearFigure()

    def plotSignalComparisonHeatmap(self, originalSignal, comparisonSignal):
        # Assert the integrity of the incoming data
        assert originalSignal.shape[0:2] == comparisonSignal.shape[0:2]

        # Extract the shapes of the data
        batchSize, numSignals, numTotalPoints = originalSignal.shape
        batchSize, numSignals, numEncodedPoints = comparisonSignal.shape

        # Separate out all the signals
        signalData = originalSignal.reshape(batchSize * numSignals, numTotalPoints)
        encodedData = comparisonSignal.reshape(batchSize * numSignals, numEncodedPoints)

        # Plot the heatmap
        self.heatmap(signalData[0:25], saveDataPath=None, title="Initial Signals", xlabel="Points", ylabel="Signal Index")
        self.heatmap(encodedData[0:25], saveDataPath=None, title="Encoded Signals", xlabel="Points", ylabel="Signal Index")

