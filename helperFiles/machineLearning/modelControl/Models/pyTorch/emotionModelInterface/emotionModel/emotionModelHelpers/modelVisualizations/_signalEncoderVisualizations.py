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

    def plotPhysiologicalProfile(self, physiologicalTimes, physiologicalProfile, epoch=0, plotTitle="Signal Encoding"):
        # Plot the signal reconstruction.
        plt.plot(physiologicalTimes, physiologicalProfile[0], c=self.blackColor, label=f"Physiological profile", linewidth=2, alpha=0.8)

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle.split("/")[-1]}")
        plt.ylabel("Signal (AU)")
        plt.legend()

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs{epoch}.pdf")
        else: plt.show()

    def plotPhysiologicalReconstruction(self, physiologicalTimes, physiologicalProfile, reconstructedPhysiologicalProfile, epoch=0, plotTitle="Signal Encoding"):
        # Plot the signal reconstruction.
        plt.plot(physiologicalTimes, physiologicalProfile[0], c=self.blackColor, label=f"Physiological profile", linewidth=2, alpha=0.8)
        for signalInd in range(reconstructedPhysiologicalProfile.shape[1]): plt.plot(physiologicalTimes, reconstructedPhysiologicalProfile[0, signalInd], c=self.lightColors[1], label=f"Reconstructed Physiological profile", linewidth=1, alpha=0.1)

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle.split("/")[-1]}")
        plt.ylabel("Signal (AU)")

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(self.saveDataFolder + f"{plotTitle} epochs{epoch}.pdf")
        else: plt.show()

    def plotSignalEncodingMap(self, physiologicalTimes, allPhysiologicalProfiles, allSignalData, epoch, plotTitle="Signal Encoding", numBatchPlots=1, numSignalPlots=1):
        datapoints = emotionDataInterface.getChannelData(allSignalData, channelName=modelConstants.signalChannel)
        timepoints = emotionDataInterface.getChannelData(allSignalData, channelName=modelConstants.timeChannel)

        for batchInd in range(allSignalData.size(0)):
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
