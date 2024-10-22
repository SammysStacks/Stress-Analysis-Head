# General
import matplotlib.pyplot as plt

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

    def plotOneSignalEncoding(self, encodedTimes, allEncodedData, epoch=0, plotTitle="Signal Encoding", numSignalPlots=1):
        # Plot the signal reconstruction.
        for batchInd in range(numSignalPlots): plt.plot(encodedTimes, allEncodedData[batchInd], c=self.darkColors[batchInd], label=f"Physiological profile: #{batchInd}", linewidth=1, alpha=1)

        # Plotting aesthetics.
        plt.xlabel("Encoding Dimension (Points)")
        plt.title(f"{plotTitle.split("/")[-1]}")
        plt.ylabel("Signal (AU)")
        plt.legend()

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs{epoch} batchInd{batchInd}.pdf")
        else: plt.show()

    def plotSignalEncodingMap(self, physiologicalTimes, allPhysiologicalProfiles, allSignalData, epoch, plotTitle="Signal Encoding", numBatchPlots=1, numSignalPlots=1):
        datapoints = emotionDataInterface.getChannelData(allSignalData, channelName=modelConstants.signalChannel)
        timepoints = emotionDataInterface.getChannelData(allSignalData, channelName=modelConstants.timeChannel)

        for batchInd in range(allSignalData.size(0)):
            # Plot the signals.
            for signalInd in range(allSignalData.size(1)):
                plt.plot(timepoints[batchInd, signalInd], datapoints[batchInd, signalInd], 'k', label="Initial Signal", linewidth=1, alpha=0.5)
                plt.plot(physiologicalTimes, allPhysiologicalProfiles[batchInd], 'tab:red', linewidth=1, label="Resampled Signal")

                # Plotting aesthetics.
                plt.title(f"batchInd-{batchInd}_signalInd-{signalInd}_{plotTitle.split("/")[-1]}")
                plt.xlabel("Signal Dimension (Points)")
                plt.ylabel("Signal (AU)")
                plt.legend()

                # Save the figure.
                if self.saveDataFolder: self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs{epoch} batchInd{batchInd}.pdf")
                else: plt.show(); plt.close('all')

                # There are too many signals to plot.
                if signalInd + 1 == numSignalPlots: break

            # There are too many signals to plot.
            if batchInd + 1 == numBatchPlots: break
