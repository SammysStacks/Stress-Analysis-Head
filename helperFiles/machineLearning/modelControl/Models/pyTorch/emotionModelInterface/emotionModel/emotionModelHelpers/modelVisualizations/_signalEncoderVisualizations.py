# General
import matplotlib.pyplot as plt
import numpy as np

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class signalEncoderVisualizations(globalPlottingProtocols):

    def __init__(self, saveDataFolder):
        super(signalEncoderVisualizations, self).__init__()
        # General parameters
        self.saveDataFolder = None

        # Set the location for saving the models.
        self.setSavingFolder(saveDataFolder)

    def setSavingFolder(self, saveDataFolder):
        self.saveDataFolder = saveDataFolder

    # --------------------- Visualize Model Parameters --------------------- #

    def plotPhysiologicalProfile(self, physiologicalTimes, physiologicalProfile, epoch=0, plotTitle="Signal Encoding"):
        # Plot the signal reconstruction.
        plt.plot(physiologicalTimes[64:128], physiologicalProfile[0][64:128], c=self.blackColor, label=f"Physiological profile", linewidth=2, alpha=0.8)

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle.split('/')[-1]}")
        plt.ylabel("Signal (AU)")
        plt.legend()

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs{epoch}.pdf")
        else: self.clearFigure()

    def plotPhysiologicalReconstruction(self, physiologicalTimes, physiologicalProfile, reconstructedPhysiologicalProfile, epoch=0, plotTitle="Signal Encoding"):
        # Extract the signal dimensions.
        batchSize, numSignals, sequenceLength = reconstructedPhysiologicalProfile.shape
        batchInd = 0

        try:
            # Plot the signal reconstruction.
            plt.plot(physiologicalTimes, physiologicalProfile[batchInd], c=self.blackColor, label=f"Physiological profile", linewidth=2, alpha=0.8)
            for signalInd in range(numSignals): plt.plot(physiologicalTimes, reconstructedPhysiologicalProfile[batchInd, signalInd], c=self.lightColors[1], label=f"Reconstructed Physiological profile", linewidth=1, alpha=0.1)
        except Exception as e: print(f"Error: {e}")

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle.split('/')[-1]}")
        plt.ylabel("Signal (AU)")

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs{epoch}.pdf")
        else: self.clearFigure()

    def plotEncoder(self, initialSignalData, reconstructedSignals, comparisonTimes, comparisonSignal, epoch, plotTitle="Encoder Prediction", numSignalPlots=1):
        # Assert the integrity of the incoming data
        assert initialSignalData.shape[0:2] == comparisonSignal.shape[0:2], f"{initialSignalData.shape} {comparisonSignal.shape}"
        batchSize, numSignals, numEncodedPoints = comparisonSignal.shape
        if batchSize == 0: return None

        # Get the signals to plot.
        plottingSignals = np.arange(0, numSignalPlots)
        plottingSignals = np.concatenate((plottingSignals, np.sort(numSignals - plottingSignals - 1)))
        assert plottingSignals[-1] == numSignals - 1, f"{plottingSignals} {numSignals}"

        # Unpack the data
        datapoints = emotionDataInterface.getChannelData(initialSignalData, channelName=modelConstants.signalChannel)
        timepoints = emotionDataInterface.getChannelData(initialSignalData, channelName=modelConstants.timeChannel)

        batchInd = 0
        for signalInd in plottingSignals:
            # Plot the signal reconstruction.
            plt.plot(timepoints[batchInd, signalInd, :], datapoints[batchInd, signalInd, :], 'o', color=self.blackColor, markersize=2, alpha=0.5, label="Initial Signal")
            plt.plot(timepoints[batchInd, signalInd, :], reconstructedSignals[batchInd, signalInd, :], 'o', color=self.lightColors[0], markersize=2, alpha=0.5, label="Reconstructed Signal")
            plt.plot(comparisonTimes, comparisonSignal[batchInd, signalInd, :], self.lightColors[1], linewidth=2, alpha=0.8, label="Resampled Signal")
            plt.xlabel("Points")
            plt.ylabel("Signal (AU)")
            plt.title(f"{plotTitle.split('/')[-1]}; Signal {signalInd + 1}")
            plt.legend(loc="best")
            if self.saveDataFolder: self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf")
            else: self.clearFigure()
