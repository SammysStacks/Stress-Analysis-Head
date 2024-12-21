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

    def timeLossComparison(self, allMetaModelPipelines, metaLearnedInfo, userInputParams, plotTitle="AutoEncoder Time Loss Plots"):
        print(f"\nPlotting the {plotTitle} Information")

        # Unpack the model information.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = metaLearnedInfo

        # Update the compiler information for this model.
        self.modelCompiler.addSubmodelParameters(loadSubmodel, userInputParams)

        timeWindows = modelConstants.modelTimeWindow

        # Initialize saving folder
        saveAutoencoderLossPlots = "/Time Analysis Plots/"
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
            if self.savingFolder: self.displayFigure(saveAutoencoderLossPlots, saveFigureName=f"{plotTitle}{timeWindow}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")

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
            if self.savingFolder: self.displayFigure(saveAutoencoderLossPlots, saveFigureName=f"{plotTitle}{metadatasetName}.pdf", baseSaveFigureName=f"{plotTitle}{metadatasetName}.pdf")

    def reconstructionLossComparison(self, allMetaModelPipelines, metaLearnedInfo, userInputParams, plotTitle="AutoEncoder Reconstruction Loss Plots"):
        print(f"\nPlotting the {plotTitle} Information")

        # Unpack the model information.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = metaLearnedInfo

        # Update the compiler information for this model.
        self.modelCompiler.addSubmodelParameters(loadSubmodel, userInputParams)

        timeWindows = modelConstants.modelTimeWindow

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
            if self.savingFolder: self.displayFigure(saveAutoencoderLossPlots, saveFigureName=f"{plotTitle}{timeWindow}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")

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
            if self.savingFolder: self.displayFigure(saveAutoencoderLossPlots, saveFigureName=f"{plotTitle}{metadatasetName}.pdf", baseSaveFigureName=f"{plotTitle}{metadatasetName}.pdf")

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
        if self.savingFolder: self.displayFigure(saveAutoencoderLossPlots, saveFigureName=f"{plotTitle}{metadatasetName}_heatmap.pdf", baseSaveFigureName=f"{plotTitle}{metadatasetName}_heatmap.pdf")
        plt.show()
