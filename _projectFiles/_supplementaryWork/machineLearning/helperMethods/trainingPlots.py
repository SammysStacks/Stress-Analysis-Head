# General

# Plotting
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import uniform_filter1d

from helperFiles.machineLearning.dataInterface.compileModelData import compileModelData
# Import files for machine learning
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelMigration import modelMigration
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class trainingPlots(globalPlottingProtocols):
    def __init__(self, modelName, datasetNames, sharedModelWeights, savingBaseFolder, accelerator=None):
        super(trainingPlots, self).__init__(interactivePlots=True)
        # General parameters
        self.sharedModelWeights = sharedModelWeights  # Possible models: [modelConstants.signalEncoderModel, modelConstants.autoencoderModel, modelConstants.signalMappingModel, modelConstants.specificEmotionModel, modelConstants.sharedEmotionModel]
        self.datasetNames = datasetNames  # Specify which datasets to compile
        self.savingFolder = savingBaseFolder  # The folder to save the figures.
        self.accelerator = accelerator  # Hugging face model optimizations.

        # Initialize relevant classes.
        self.modelCompiler = compileModelData(useTherapyData=False, accelerator=accelerator)
        self.modelMigration = modelMigration(accelerator=accelerator)

        self.rawDataOrder = ['EOG', 'EEG', 'EDA', 'Temp']
        self.rawDataColors = [
            '#3498db',  # Blue shades
            '#9ED98F',  # Green shades
            '#918ae1',  # Purple shades
            '#fc827f',  # Red shades
        ]

        self.activityOrder = ['CPT', 'Exersice', 'Music', 'VR']
        self.activityColors = [
            '#3498db',  # Blue shades
            '#fc827f',  # Red shades
            '#9ED98F',  # Green shades
            '#918ae1',  # Purple shades
        ]

        self.darkColors = [
            '#3498db',  # Blue shades
            '#fc827f',  # Red shades
            '#9ED98F',  # Green shades
            '#918ae1',  # Purple shades
            '#eca163',  # Orange shades
            '#f0d0ff',  # Pink shades (ADDED TO HAVE ENOUGH COLORS, CHANGE HEX)
        ]

    # ---------------------------------------------------------------------- #
    # ------------------------- Feature Label Plots ------------------------ #

    @staticmethod
    def getSubmodel(metaModel, submodel):
        if submodel == modelConstants.signalEncoderModel:
            return metaModel.model.specificSignalEncoderModel
        elif submodel == modelConstants.emotionModel:
            return metaModel.model.emotionModel
        else:
            raise Exception()

    @staticmethod
    def getSmoothedLosses(current_losses, window_length=5):
        # Check if the current time window has enough epochs to apply the filter.
        if len(current_losses) >= window_length:
            # Apply a moving average filter to the losses.
            smoothed_losses = uniform_filter1d(current_losses, size=window_length, output=np.float64)
        else:
            # If not enough epochs, use the original losses as smoothed losses.
            smoothed_losses = current_losses

        return smoothed_losses

    def getSmoothedFinalLosses(self, losses, window_length=5):
        """ Expected format: (numTimeWindows, numEpochs) """
        # Initialize an array to store the minimum loss for each time window.
        finalLosses = np.zeros(len(losses))

        # Iterate over each time window.
        for i in range(finalLosses.shape[0]):
            smoothedLosses = self.getSmoothedLosses(losses[i], window_length=window_length)

            # Find the minimum loss in the smoothed losses for the current time window.
            finalLosses[i] = np.min(smoothedLosses)

        return finalLosses

    def plot_heatmap(self, data, column_labels, row_labels, columnLabel, rowLabel, title=None, color_map='viridis', cbar_label="Value", useLogNorm=False, saveFigurePath=None, cmapBounds=[None, None]):
        # Create the figure and the heatmap.
        fig, ax = plt.subplots(figsize=(10, 8))

        # Set normalization
        if useLogNorm:
            norm = LogNorm(vmin=cmapBounds[0], vmax=cmapBounds[1])
        else:
            norm = Normalize(vmin=cmapBounds[0], vmax=cmapBounds[1])

        # Plot the heatmap
        heatmap = ax.imshow(data, cmap=color_map, aspect='auto', norm=norm)

        # Set the title if provided
        if title:
            ax.set_title(title)

        # Label the axes
        ax.set_xlabel(columnLabel)
        ax.set_ylabel(rowLabel)

        # Assign the row and column labels correctly
        ax.set_xticks(np.arange(len(column_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(column_labels)
        ax.set_yticklabels(row_labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add a color bar with the label.
        cbar = ax.figure.colorbar(heatmap, ax=ax)
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

        # Save the figure if desired.
        self.displayFigure(saveFigurePath)
