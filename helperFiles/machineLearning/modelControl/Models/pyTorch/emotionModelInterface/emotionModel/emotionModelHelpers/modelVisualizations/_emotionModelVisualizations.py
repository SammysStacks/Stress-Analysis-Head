# General

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols


class emotionModelVisualizations(globalPlottingProtocols):

    def __init__(self, baseSavingFolder, stringID, datasetName):
        super(emotionModelVisualizations, self).__init__(interactivePlots=False)
        self.setSavingFolder(baseSavingFolder, stringID, datasetName)

    # --------------------- Visualize Model Parameters --------------------- #

    def plotPredictedMatrix(self, allTrainingLabels, allTestingLabels, allPredictedTrainingLabels, allPredictedTestingLabels, numClasses, epoch, saveFigureLocation, plotTitle):
        allPredictedTrainingLabels, allPredictedTestingLabels = allPredictedTrainingLabels.astype(int), allPredictedTestingLabels.astype(int)
        allTrainingLabels, allTestingLabels = allTrainingLabels.astype(int), allTestingLabels.astype(int)

        # Calculate confusion matrices
        training_confusion_matrix = confusion_matrix(allTrainingLabels, allPredictedTrainingLabels, labels=np.arange(numClasses), normalize='true')
        testing_confusion_matrix = confusion_matrix(allTestingLabels, allPredictedTestingLabels, labels=np.arange(numClasses), normalize='true')
        training_confusion_matrix = np.round(training_confusion_matrix, decimals=2)
        testing_confusion_matrix = np.round(testing_confusion_matrix, decimals=2)

        # Define a gridspec with width ratios for subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1]})

        # Plot the confusion matrices as heatmaps
        im0 = axes[0].imshow(training_confusion_matrix, cmap='BuGn', vmin=0, vmax=1, aspect='auto')
        axes[0].set_title('Training Confusion Matrix')
        axes[0].set_xlabel('Predicted Labels')
        axes[0].set_ylabel('True Labels')
        axes[0].invert_yaxis()
        fig.colorbar(im0, ax=axes[0], format='%.2f')

        for i in range(numClasses):
            for j in range(numClasses):
                axes[0].text(j, i, f'{training_confusion_matrix[i, j] * 100:.2f}', ha='center', va='center', color='black', fontsize=10)

        im1 = axes[1].imshow(testing_confusion_matrix, cmap='BuGn', vmin=0, vmax=1, aspect='auto')
        axes[1].set_title('Testing Confusion Matrix')
        axes[1].set_xlabel('Predicted Labels')
        axes[1].set_ylabel('True Labels')
        axes[1].invert_yaxis()
        fig.colorbar(im1, ax=axes[1], format='%.2f')

        for i in range(numClasses):
            for j in range(numClasses):
                axes[1].text(j, i, f'{testing_confusion_matrix[i, j] * 100:.2f}', ha='center', va='center', color='black', fontsize=10)
        axes[1].invert_yaxis()  # Reverse the order of y-axis ticks
        fig.suptitle(f"{plotTitle} at epoch{epoch}", fontsize=24)
        fig.tight_layout()

        # Save the figure is desired.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} at epoch{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", fig=fig, clearFigure=True, showPlot=not self.hpcFlag)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotDistributions(self, allProfiles, distributionNames, batchInd, epoch, showMinimumPlots, saveFigureLocation="", plotTitle="profile distribution"):
        # Assert the integrity of the incoming data
        batchSize, numDistributions, encodedDimension = allProfiles.shape
        if batchSize == 0: return None

        for distributionInd in range(numDistributions):
            profile = allProfiles[batchInd, distributionInd, :]
            fig, ax = plt.subplots(figsize=(6.4, 4.8))

            # Plot the signal reconstruction.
            ax.plot(profile, 'o-', color=self.blackColor, markersize=2, alpha=0.75, label="Initial Signal")
            # ax.axhline(y=0, color=self.blackColor, linewidth=0.5, alpha=0.25)

            # Plotting aesthetics.
            ax.set_title(f"{plotTitle} {distributionNames[distributionInd]} epoch{epoch}")
            ax.set_ylabel("Signal amplitude (au)")
            ax.legend(loc="best")
            ax.set_xlabel("Time (s)")

            # Save the figure.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} {distributionNames[distributionInd]} epoch{epoch}.pdf", baseSaveFigureName=f"{plotTitle} {distributionNames[distributionInd]}.pdf", fig=fig, clearFigure=True, showPlot=not self.hpcFlag)
            else: self.clearFigure(fig=fig, legend=None, showPlot=True)
            plt.close(fig)

            # Only plot all when needed.
            if showMinimumPlots: break
