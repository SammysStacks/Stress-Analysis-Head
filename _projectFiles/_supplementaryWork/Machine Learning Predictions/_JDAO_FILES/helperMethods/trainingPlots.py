# General
import os

# Plotting
import matplotlib.pyplot as plt
import numpy as np

from helperFiles.machineLearning.dataInterface.compileModelData import compileModelData
# Import files for machine learning
from helperFiles.machineLearning.modelControl.Models.pyTorch.modelMigration import modelMigration
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class trainingPlots(globalPlottingProtocols):
    def __init__(self, modelName, datasetNames, sharedModelWeights, savingBaseFolder, accelerator=None):
        super(trainingPlots, self).__init__()
        # General parameters
        self.sharedModelWeights = sharedModelWeights  # Possible models: [modelConstants.signalEncoderModel, modelConstants.autoencoderModel, modelConstants.signalMappingModel, modelConstants.specificEmotionModel, modelConstants.sharedEmotionModel]
        self.datasetNames = datasetNames  # Specify which datasets to compile
        self.savingFolder = savingBaseFolder  # The folder to save the figures.
        self.accelerator = accelerator  # Hugging face model optimizations.
        self.modelName = modelName  # The emotion model's unique identifier. Options: emotionModel

        # Initialize relevant classes.
        self.modelCompiler = compileModelData(submodel=None, userInputParams={}, accelerator=accelerator)
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

    def plot_heatmap(self, data, column_labels, row_labels, columnLabel, rowLabel, title=None, color_map='viridis', cbar_label="Value", saveFigurePath=None):
        """
        Plot a heatmap with given labels and title.

        Parameters:
        - data: 2D numpy array containing the heatmap data.
        - column_labels: List of labels for the columns.
        - row_labels: List of labels for the rows.
        - xlabel: Label for the x-axis.
        - ylabel: Label for the y-axis.
        - title: Title of the heatmap. Optional.
        - color_map: Colormap for the heatmap. Default is 'viridis'.
        - cbar_label: Label for the colorbar. Default is "Value".
        """
        # Create the figure and the heatmap.
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.imshow(data, cmap=color_map, aspect='auto', interpolation='spline16')

        # Set the title if provided
        if title: ax.set_title(title)
        # Label the axes
        ax.set_xlabel(columnLabel)
        ax.set_ylabel(rowLabel)

        # Assign the row and column labels
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

    def timeLossComparison(self, allMetaModelPipelines, metaLearnedInfo, userInputParams, plotTitle="AutoEncoder Time Loss Plots"):
        print(f"\nPlotting the {plotTitle} Information")

        # Unpack the model information.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = metaLearnedInfo

        # Update the compiler information for this model.
        self.modelCompiler.addSubmodelParameters(loadSubmodel, userInputParams)

        timeWindows = modelConstants.timeWindows

        # Initialize saving folder
        saveAutoencoderLossPlots = self.savingFolder + "/Time Analysis Plots/"
        os.makedirs(saveAutoencoderLossPlots, exist_ok=True)
        print(loadSubmodel, loadSubmodelDate, loadSubmodelEpochs)

        # Load in the previous model weights and attributes.
        self.modelMigration.loadModels(allMetaModelPipelines, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True)

        # sys.exit()

        # ---- Time Analysis Loss versus Epoch, all sub models, one plot per time window ----
        # For each timeWindow
        for timeWindowInd, timeWindow in enumerate(timeWindows):
            # For each metalearning model
            for metaModelInd in range(len(allMetaModelPipelines)):
                metaModel = self.getSubmodel(allMetaModelPipelines[metaModelInd], loadSubmodel)
                metadatasetName = allMetaModelPipelines[metaModelInd].datasetName
                print(metadatasetName, metaModel.testingLosses_timeLayerAnalysis[timeWindowInd])
                print(metadatasetName, metaModel.trainingLosses_timeLayerAnalysis[timeWindowInd])
                print(metadatasetName, metaModel.numEncodingsPath_timeAnalysis[timeWindowInd])

                # Plot the training loss.
                plt.plot(metaModel.trainingLosses_timeLayerAnalysis[timeWindowInd], label=f'{metadatasetName} Training Loss', color=self.darkColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeLayerAnalysis[timeWindowInd], label=f'{metadatasetName} Testing Loss', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metadatasetName} Num Encodings', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.legend(loc="upper right")
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, timeWindow={timeWindow}s")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.displayFigure(saveAutoencoderLossPlots + f"{plotTitle}{timeWindow}.pdf")
            plt.show()

        # ---- Time Analysis Loss versus Epoch, all time windows, one plot per sub model ----
        for metaModelInd in range(len(allMetaModelPipelines)):
            metaModel = self.getSubmodel(allMetaModelPipelines[metaModelInd], loadSubmodel)
            metadatasetName = allMetaModelPipelines[metaModelInd].datasetName
            # Plot the training loss.
            for timeWindowInd, timeWindow in enumerate(timeWindows):
                plt.plot(metaModel.trainingLosses_timeLayerAnalysis[timeWindowInd], label=f'{metadatasetName} Training Loss, {timeWindow}s', color=self.darkColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeLayerAnalysis[timeWindowInd], label=f'{metadatasetName} Testing Loss, {timeWindow}s', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metadatasetName} Num Encodings, {timeWindow}s', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.legend(loc="upper right")
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, dataset={metadatasetName}")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.displayFigure(saveAutoencoderLossPlots + f"{plotTitle}{metadatasetName}.pdf")
            plt.show()

    def reconstructionLossComparison(self, allMetaModelPipelines, metaLearnedInfo, userInputParams, plotTitle="AutoEncoder Reconstruction Loss Plots"):
        print(f"\nPlotting the {plotTitle} Information")

        # Unpack the model information.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = metaLearnedInfo

        # Update the compiler information for this model.
        self.modelCompiler.addSubmodelParameters(loadSubmodel, userInputParams)

        timeWindows = modelConstants.timeWindows

        # Initialize saving folder
        saveAutoencoderLossPlots = self.savingFolder + "/Time Analysis Plots/"
        os.makedirs(saveAutoencoderLossPlots, exist_ok=True)
        print('here')

        # Load in the previous model weights and attributes.
        self.modelMigration.loadModels(allMetaModelPipelines, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True)

        # ---- Time Analysis Loss versus Epoch, all sub models, one plot per time window ----
        # For each timeWindow
        for timeWindowInd, timeWindow in enumerate(timeWindows):
            # For each metalearning model
            for metaModelInd in range(len(allMetaModelPipelines)):
                metaModel = self.getSubmodel(allMetaModelPipelines[metaModelInd], loadSubmodel)
                metadatasetName = allMetaModelPipelines[metaModelInd].datasetName
                print(metadatasetName, metaModel.trainingLosses_signalReconstruction[timeWindowInd])
                print(metadatasetName, metaModel.testingLosses_signalReconstruction[timeWindowInd])
                print(metadatasetName, metaModel.numEncodingsPath_timeAnalysis[timeWindowInd])

                # Plot the training loss.
                plt.plot(metaModel.trainingLosses_signalReconstruction[timeWindowInd], label=f'{metadatasetName} Training Loss', color=self.darkColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_signalReconstruction[timeWindowInd], label=f'{metadatasetName} Testing Loss', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metadatasetName} Num Encodings', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.legend(loc="upper right")
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, timeWindow={timeWindow}s")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.displayFigure(saveAutoencoderLossPlots + f"{plotTitle}{timeWindow}.pdf")
            plt.show()

        # ---- Time Analysis Loss versus Epoch, all time windows, one plot per sub model ----
        for metaModelInd in range(len(allMetaModelPipelines)):
            metaModel = self.getSubmodel(allMetaModelPipelines[metaModelInd], loadSubmodel)
            metadatasetName = allMetaModelPipelines[metaModelInd].datasetName
            # Plot the training loss.
            for timeWindowInd, timeWindow in enumerate(timeWindows):
                plt.plot(metaModel.trainingLosses_signalReconstruction[timeWindowInd], label=f'{metadatasetName} Training Loss, {timeWindow}s', color=self.darkColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_signalReconstruction[timeWindowInd], label=f'{metadatasetName} Testing Loss, {timeWindow}s', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metadatasetName} Num Encodings, {timeWindow}s', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            # no legend for now
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, dataset={metadatasetName}")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.displayFigure(saveAutoencoderLossPlots + f"{plotTitle}{metadatasetName}.pdf")
            plt.show()

        # ---- Heatmap of the reconstruction loss, collected only ----
        metaModel = self.getSubmodel(allMetaModelPipelines[0], loadSubmodel)
        metadatasetName = allMetaModelPipelines[0].datasetName

        # x = time window, y = num encodings
        accuracy = []
        for timeWindowInd, timeWindow in enumerate(timeWindows):
            # find where change in compression factor occurs by using numEncodingsPath_timeAnalysis
            accuracy_per_comp_factor = []
            numEncodings_list = metaModel.numEncodingsPath_timeAnalysis[timeWindowInd]
            for i in range(1, len(numEncodings_list)):
                if numEncodings_list[i] > numEncodings_list[i - 1]:
                    accuracy_per_comp_factor.append(metaModel.trainingLosses_signalReconstruction[timeWindowInd][i])
            accuracy_per_comp_factor.append(metaModel.trainingLosses_signalReconstruction[timeWindowInd][-1])
            accuracy.append(accuracy_per_comp_factor)
        # plot heatmap from 2d list
        plt.imshow(np.asarray(accuracy).T, cmap='Blues', interpolation='nearest')
        plt.xlabel('Time Window')
        plt.xticks(range(len(timeWindows)), timeWindows)
        plt.ylabel('Number of Encodings')
        yticks = np.sort(np.unique(np.asarray(metaModel.numEncodingsPath_timeAnalysis).flatten()))
        print(accuracy)
        plt.yticks(range(len(yticks)), yticks)
        plt.colorbar(label='Reconstruction Loss')
        if self.savingFolder:
            self.displayFigure(saveAutoencoderLossPlots + f"{plotTitle}{metadatasetName}_heatmap.pdf")
        plt.show()

    def meanLossComparison(self, allMetaModelPipelines, metaLearnedInfo, userInputParams, plotTitle="AutoEncoder Mean Loss Plots"):
        print(f"\nPlotting the {plotTitle} Information")

        # Unpack the model information.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = metaLearnedInfo

        # Update the compiler information for this model.
        self.modelCompiler.addSubmodelParameters(loadSubmodel, userInputParams)

        timeWindows = modelConstants.timeWindows

        # Initialize saving folder
        saveAutoencoderLossPlots = self.savingFolder + "/Time Analysis Plots/"
        os.makedirs(saveAutoencoderLossPlots, exist_ok=True)
        print('here')

        # Load in the previous model weights and attributes.
        self.modelMigration.loadModels(allMetaModelPipelines, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True)

        # ---- Time Analysis Loss versus Epoch, all sub models, one plot per time window ----
        # For each timeWindow
        for timeWindowInd, timeWindow in enumerate(timeWindows):
            # For each metalearning model
            for metaModelInd in range(len(allMetaModelPipelines)):
                metaModel = self.getSubmodel(allMetaModelPipelines[metaModelInd], loadSubmodel)
                metadatasetName = allMetaModelPipelines[metaModelInd].datasetName
                print(metadatasetName, metaModel.trainingLosses_timeMeanAnalysis[timeWindowInd])
                print(metadatasetName, metaModel.testingLosses_timeMeanAnalysis[timeWindowInd])
                print(metadatasetName, metaModel.numEncodingsPath_timeAnalysis[timeWindowInd])

                # Plot the training loss.
                plt.plot(metaModel.trainingLosses_timeMeanAnalysis[timeWindowInd], label=f'{metadatasetName} Training Loss', color=self.darkColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeMeanAnalysis[timeWindowInd], label=f'{metadatasetName} Testing Loss', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metadatasetName} Num Encodings', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.legend(loc="upper right")
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, timeWindow={timeWindow}s")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.displayFigure(saveAutoencoderLossPlots + f"{plotTitle}{timeWindow}.pdf")
            plt.show()

        # ---- Time Analysis Loss versus Epoch, all time windows, one plot per sub model ----
        for metaModelInd in range(len(allMetaModelPipelines)):
            metaModel = self.getSubmodel(allMetaModelPipelines[metaModelInd], loadSubmodel)
            metadatasetName = allMetaModelPipelines[metaModelInd].datasetName
            # Plot the training loss.
            for timeWindowInd, timeWindow in enumerate(timeWindows):
                plt.plot(metaModel.trainingLosses_timeMeanAnalysis[timeWindowInd], label=f'{metadatasetName} Training Loss, {timeWindow}s', color=self.darkColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeMeanAnalysis[timeWindowInd], label=f'{metadatasetName} Testing Loss, {timeWindow}s', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metadatasetName} Num Encodings, {timeWindow}s', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, dataset={metadatasetName}")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.displayFigure(saveAutoencoderLossPlots + f"{plotTitle}{metadatasetName}.pdf")
            plt.show()

    def stdLossComparison(self, allMetaModelPipelines, metaLearnedInfo, userInputParams, plotTitle="AutoEncoder Std Loss Plots"):
        print(f"\nPlotting the {plotTitle} Information")

        # Unpack the model information.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = metaLearnedInfo

        # Update the compiler information for this model.
        self.modelCompiler.addSubmodelParameters(loadSubmodel, userInputParams)

        timeWindows = modelConstants.timeWindows

        # Initialize saving folder
        saveAutoencoderLossPlots = self.savingFolder + "/Time Analysis Plots/"
        os.makedirs(saveAutoencoderLossPlots, exist_ok=True)
        print('here')

        # Load in the previous model weights and attributes.
        self.modelMigration.loadModels(allMetaModelPipelines, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True)

        # ---- Time Analysis Loss versus Epoch, all sub models, one plot per time window ----
        # For each timeWindow
        for timeWindowInd, timeWindow in enumerate(timeWindows):
            # For each metalearning model
            for metaModelInd in range(len(allMetaModelPipelines)):
                metaModel = self.getSubmodel(allMetaModelPipelines[metaModelInd], loadSubmodel)
                metadatasetName = allMetaModelPipelines[metaModelInd].datasetName
                print(metadatasetName, metaModel.trainingLosses_timeSTDAnalysis[timeWindowInd])
                print(metadatasetName, metaModel.testingLosses_timeSTDAnalysis[timeWindowInd])
                print(metadatasetName, metaModel.numEncodingsPath_timeAnalysis[timeWindowInd])

                # Plot the training loss.
                plt.plot(metaModel.trainingLosses_timeSTDAnalysis[timeWindowInd], label=f'{metadatasetName} Training Loss', color=self.darkColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeSTDAnalysis[timeWindowInd], label=f'{metadatasetName} Testing Loss', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metadatasetName} Num Encodings', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.legend(loc="upper right")
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, timeWindow={timeWindow}s")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.displayFigure(saveAutoencoderLossPlots + f"{plotTitle}{timeWindow}.pdf")
            plt.show()

        # ---- Time Analysis Loss versus Epoch, all time windows, one plot per sub model ----
        for metaModelInd in range(len(allMetaModelPipelines)):
            metaModel = self.getSubmodel(allMetaModelPipelines[metaModelInd], loadSubmodel)
            metadatasetName = allMetaModelPipelines[metaModelInd].datasetName
            # Plot the training loss.
            for timeWindowInd, timeWindow in enumerate(timeWindows):
                plt.plot(metaModel.trainingLosses_timeSTDAnalysis[timeWindowInd], label=f'{metadatasetName} Training Loss, {timeWindow}s', color=self.darkColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeSTDAnalysis[timeWindowInd], label=f'{metadatasetName} Testing Loss, {timeWindow}s', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metadatasetName} Num Encodings, {timeWindow}s', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, dataset={metadatasetName}")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.displayFigure(saveAutoencoderLossPlots + f"{plotTitle}{metadatasetName}.pdf")
            plt.show()
    '''
    def autoencoderLossComparison(self, allMetaModelPipelines, allMetadataLoaders, metaLearnedInfo, modelComparisonInfo, comparingModelInd, plotTitle="Autoencoder Loss Plots"):
        # Initialize saving folder
        saveAutoencoderLossPlots = self.savingFolder + "/Autoencoder Plots/"
        os.makedirs(saveAutoencoderLossPlots, exist_ok=True)
        print("\tPlotting the Autoencoder Loss Information")

        # Load in the previous model weights and attributes.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = metaLearnedInfo
        self.modelMigration.loadModels(allMetaModelPipelines, allMetadataLoaders, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True)
        # For each metalearning model
        for metaModelInd in range(len(allMetaModelPipelines)):
            metaModel = allMetaModelPipelines[metaModelInd]
            metadatasetName = metaModel.datasetName

            # Plot the training loss.
            plt.plot(metaModel.model.autoencoderModel.trainingLosses_signalReconstruction, label=f'{metadatasetName} Training Loss', color=self.darkColors[metaModelInd], linewidth=2, alpha=1)
            plt.plot(metaModel.model.autoencoderModel.testingLosses_signalReconstruction, label=f'{metadatasetName} Testing Loss', color=self.darkColors[metaModelInd], linewidth=2, alpha=0.5)

        # Load in the previous model weights and attributes.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = modelComparisonInfo
        self.modelMigration.loadModels(allMetaModelPipelines, allMetadataLoaders, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True)

        # Plot the training loss.
        metadatasetName = allMetaModelPipelines[comparingModelInd].datasetName
        plt.plot(allMetaModelPipelines[comparingModelInd].model.autoencoderModel.trainingLosses_signalReconstruction, label=f'{metadatasetName} Training Loss', color='k', linewidth=2, alpha=1)
        plt.plot(allMetaModelPipelines[comparingModelInd].model.autoencoderModel.testingLosses_signalReconstruction, label=f'{metadatasetName} Testing Loss', color='k', linewidth=2, alpha=0.5)

        # Set y-axis to logarithmic scale
        plt.yscale('log')

        # Label the plot.
        # plt.legend(loc="upper right")
        plt.xlabel("Training Epoch")
        plt.ylabel("Loss Values")
        plt.title(f"{plotTitle}")

        # Save the figure if desired.
        if self.savingFolder:
            self.saveFigure(saveAutoencoderLossPlots + f"{plotTitle}.pdf")
        plt.show()
    '''