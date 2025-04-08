# General
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Wedge
from shap.plots.colors._colors import lch2rgb
from sklearn.metrics import confusion_matrix

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleLieLayer import reversibleLieLayer


class emotionModelVisualizations(globalPlottingProtocols):

    def __init__(self, baseSavingFolder, stringID, datasetName):
        super(emotionModelVisualizations, self).__init__(interactivePlots=False)
        self.setSavingFolder(baseSavingFolder, stringID, datasetName)

        # Create custom colormap (as in your original code)
        blue_lch = [54., 70., 4.6588]
        red_lch = [54., 90., 0.35470565 + 2 * np.pi]
        blue_rgb = lch2rgb(blue_lch)
        red_rgb = lch2rgb(red_lch)
        white_rgb = np.asarray([1., 1., 1.])

        # Create the colormap
        colors = []; num_steps = 200
        for alpha in np.linspace(start=1, stop=0, num=num_steps):
            c = blue_rgb * alpha + (1 - alpha) * white_rgb
            colors.append(c)
        for alpha in np.linspace(start=0, stop=1, num=num_steps):
            c = red_rgb * alpha + (1 - alpha) * white_rgb
            colors.append(c)

        # Create the colormap
        self.custom_cmap = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

    # --------------------- Visualize Model Parameters --------------------- #

    def plotPredictedMatrix(self, allTrainingLabels, allTestingLabels, allPredictedTrainingLabels, allPredictedTestingLabels, numClasses, epoch, saveFigureLocation, plotTitle):
        # Calculate confusion matrices
        training_confusion_matrix = confusion_matrix(allTrainingLabels, allPredictedTrainingLabels, labels=np.arange(numClasses), normalize='true')
        testing_confusion_matrix = confusion_matrix(allTestingLabels, allPredictedTestingLabels, labels=np.arange(numClasses), normalize='true')

        # Define a gridspec with width ratios for subplots
        fig, axes = plt.subplots(nRows=1, nCols=2, figsize=np.asarray([15, 5]), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1]})

        # Plot the confusion matrices as heatmaps
        im0 = axes[0].imshow(training_confusion_matrix,
                             cmap='BuGn', vmin=0, vmax=1, aspect='auto')
        axes[0].set_title('Training Confusion Matrix')
        axes[0].set_xlabel('Predicted Labels')
        axes[0].set_ylabel('True Labels')
        axes[0].invert_yaxis()  # Reverse the order of y-axis ticks
        fig.colorbar(im0, ax=axes[0], format='%.2f')

        # Display percentages in the boxes
        for i in range(numClasses):
            for j in range(numClasses):
                axes[0].text(j, i, f'{training_confusion_matrix[i, j] * 100:.2f}%', ha='center', va='center', color='black')

        im1 = axes[1].imshow(testing_confusion_matrix, cmap='BuGn', vmin=0, vmax=1, aspect='auto')
        axes[1].set_title('Testing Confusion Matrix')
        axes[1].set_xlabel('Predicted Labels')
        axes[1].set_ylabel('True Labels')
        fig.colorbar(im1, ax=axes[1], format='%.2f')

        # Display percentages in the boxes
        for i in range(numClasses):
            for j in range(numClasses):
                axes[1].text(j, i, f'{testing_confusion_matrix[i, j] * 100:.2f}%',
                             ha='center', va='center', color='black')

        axes[1].invert_yaxis()  # Reverse the order of y-axis ticks
        fig.suptitle(f"{plotTitle} at epoch{epoch}", fontsize=24)
        fig.tight_layout()

        # Save the figure is desired.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} at epoch{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", fig=fig, clearFigure=True, showPlot=not self.hpcFlag)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotDistributions(self, allProfiles, distributionNames, batchInd, epoch, saveFigureLocation="", plotTitle="profile distribution"):
        # Assert the integrity of the incoming data
        batchSize, numDistributions, encodedDimension = allProfiles.shape
        if batchSize == 0: return None

        for distributionInd in range(numDistributions):
            profile = allProfiles[batchInd, distributionInd, :]
            fig, ax = plt.subplots(figsize=(6.4, 4.8))

            # Plot the signal reconstruction.
            ax.plot(profile, 'o-', color=self.blackColor, markersize=2, alpha=0.75, label="Initial Signal")
            ax.axhline(y=0, color=self.blackColor, linewidth=0.5, alpha=0.25)

            # Plotting aesthetics.
            ax.set_title(f"{plotTitle} {distributionNames[distributionInd]} epoch{epoch}")
            ax.set_ylabel("Signal amplitude (au)")
            ax.legend(loc="best")
            ax.set_xlabel("Time (sec)")
            ax.set_ylim((-1.75, 1.75))

            # Save the figure.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} {distributionNames[distributionInd]} epoch{epoch}.pdf", baseSaveFigureName=f"{plotTitle} {distributionNames[distributionInd]}.pdf", fig=fig, clearFigure=True, showPlot=not self.hpcFlag)
            else: self.clearFigure(fig=fig, legend=None, showPlot=True)
            plt.close(fig)
