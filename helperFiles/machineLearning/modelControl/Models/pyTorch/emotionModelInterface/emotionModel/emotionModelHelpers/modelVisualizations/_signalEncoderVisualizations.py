# General
import matplotlib.pyplot as plt
import numpy as np

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class signalEncoderVisualizations(globalPlottingProtocols):

    def __init__(self, baseSavingFolder, stringID):
        super(signalEncoderVisualizations, self).__init__()
        self.setSavingFolder(baseSavingFolder, stringID, datasetName)

    # --------------------- Visualize Model Parameters --------------------- #

    def plotPhysiologicalProfile(self, physiologicalTimes, physiologicalProfile, epoch=0, saveFigureLocation="", plotTitle="Signal Encoding"):
        # Plot the signal reconstruction.
        plt.plot(physiologicalTimes[64:128], physiologicalProfile[0][64:128], c=self.blackColor, label=f"Physiological profile", linewidth=2, alpha=0.8)

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle} epoch{epoch}")
        plt.ylabel("Signal (AU)")
        plt.legend()

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure()

    def plotPhysiologicalReconstruction(self, physiologicalTimes, physiologicalProfile, reconstructedPhysiologicalProfile, epoch=0, saveFigureLocation="", plotTitle="Signal Encoding"):
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
        plt.title(f"{plotTitle} epoch{epoch}")
        plt.ylabel("Signal (AU)")

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure()

    def plotEncoder(self, initialSignalData, reconstructedSignals, comparisonTimes, comparisonSignal, epoch, saveFigureLocation="", plotTitle="Encoder Prediction", numSignalPlots=1):
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
            plt.title(f"{plotTitle} epoch{epoch} signal{signalInd + 1}")
            plt.legend(loc="best")

            # Save the figure.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
            else: self.clearFigure()

    def plotSignalComparison(self, originalSignal, comparisonSignal, epoch, saveFigureLocation, plotTitle, numSignalPlots=1):
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
            plt.title(f"{plotTitle} epoch{epoch} signal{signalInd + 1}")

            # Save the plot
            if self.saveDataFolder: self.displayFigure(saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
            else: self.clearFigure()

            # There are too many signals to plot.
            if signalInd + 1 == numSignalPlots: break

    def plotAllSignalComparisons(self, distortedSignals, reconstructedDistortedSignals, trueSignal, epoch, signalInd, saveFigureLocation, plotTitle):
        numSignals, numTotalPoints = reconstructedDistortedSignals.shape
        alphas = np.linspace(0.1, 1, numSignals)

        # Plot all signals in 'distortedSignals'
        for i in range(numSignals):
            plt.plot(distortedSignals[i], '-', color='k', alpha=alphas[i], linewidth=1, markersize=2, zorder=0)
            plt.plot(trueSignal, 'o', color='tab:blue', linewidth=1, markersize=2, zorder=10)
            plt.plot(reconstructedDistortedSignals[i], '-', color='tab:red', linewidth=1, markersize=2, alpha=alphas[i], zorder=5)

        plt.title(f"{plotTitle} epoch{epoch} signal{signalInd + 1}")
        plt.xlabel('Time (Seconds)')
        plt.ylabel("Arbitrary Axis (AU)")
        plt.legend(['Noisy Signal', 'True Signal', 'Reconstructed Signal'], loc='best')

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
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

