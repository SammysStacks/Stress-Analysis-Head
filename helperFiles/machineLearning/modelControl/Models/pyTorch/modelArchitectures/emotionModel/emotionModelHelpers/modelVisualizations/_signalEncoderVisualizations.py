# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Sklearn
from sklearn.decomposition import PCA

# Plotting
import matplotlib.pyplot as plt

# Visualization protocols
from .........globalPlottingProtocols import globalPlottingProtocols


class signalEncoderVisualizations(globalPlottingProtocols):

    def __init__(self, saveDataFolder):
        super(signalEncoderVisualizations, self).__init__()
        # General parameters
        self.saveDataFolder = None

        # Set the location for saving the models.
        self.setSavingFolder(saveDataFolder)

    def setSavingFolder(self, saveDataFolder):
        self.saveDataFolder = saveDataFolder

    # ---------------------------------------------------------------------- #
    # --------------------- Visualize Model Parameters --------------------- #

    def plotSignalEncoding(self, allEncodedData, epoch, plotTitle="Signal Encoding"):
        batchSize, numCondensedSignals, compressedLength = allEncodedData.shape
        # allEncodedData dimension: batchSize, numCondensedSignals, compressedLength 

        # Plot the signal reconstruction.
        plt.plot(allEncodedData[0].view(-1), 'k', linewidth=2, alpha=1)
        plt.plot(allEncodedData[1].view(-1), 'k', linewidth=2, alpha=0.6)
        plt.plot(allEncodedData[-2].view(-1), 'tab:red', linewidth=2, alpha=1)
        plt.plot(allEncodedData[-1].view(-1), 'tab:red', linewidth=2, alpha=0.6)

        plt.xlabel("All Encoding Dimensions (Points)")
        plt.ylabel("Signal (AU)")
        plt.title(f"{plotTitle}")
        if self.saveDataFolder:
            self.saveFigure(self.saveDataFolder + f"{plotTitle} epochs{epoch}.pdf")
        plt.show()

    def plotOneSignalEncoding(self, allEncodedData, epoch, plotTitle="Signal Encoding", numBatchPlots=1):
        batchSize, numCondensedSignals, compressedLength = allEncodedData.shape
        # allEncodedData dimension: batchSize, numCondensedSignals, compressedLength  

        for batchInd in range(batchSize):
            # Plot the signal reconstruction.
            plt.plot(allEncodedData[batchInd, 0], 'k', linewidth=2, alpha=1)
            plt.plot(allEncodedData[batchInd, 1], 'tab:blue', linewidth=2, alpha=1)
            plt.plot(allEncodedData[batchInd, -2], 'tab:red', linewidth=2, alpha=1)
            plt.plot(allEncodedData[batchInd, -1], 'tab:green', linewidth=2, alpha=1)

            plt.xlabel("Encoding Dimension (Points)")
            plt.ylabel("Signal (AU)")
            plt.title(f"{plotTitle}")
            if self.saveDataFolder:
                self.saveFigure(self.saveDataFolder + f"{plotTitle} epochs{epoch} batchInd{batchInd}.pdf")
            plt.show()

            # There are too many signals to plot.
            if batchInd + 1 == numBatchPlots: break
