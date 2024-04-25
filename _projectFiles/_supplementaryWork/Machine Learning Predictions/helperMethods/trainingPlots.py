# General
import os

# Plotting
import matplotlib.pyplot as plt
import numpy as np

from helperFiles.machineLearning.dataInterface.compileModelData import compileModelData
# Import files for machine learning
from helperFiles.machineLearning.modelControl.Models.pyTorch.Helpers.modelMigration import modelMigration
from helperFiles.globalPlottingProtocols import globalPlottingProtocols


class trainingPlots(globalPlottingProtocols):
    def __init__(self, modelName, datasetNames, sharedModelWeights, savingBaseFolder, accelerator=None):
        super(trainingPlots, self).__init__()
        # General parameters
        self.timeWindows = [90, 120, 150, 180, 210, 240]  # A list of all time windows to consider for the encoding.
        self.sharedModelWeights = sharedModelWeights  # Possible models: ["trainingInformation", "signalEncoderModel", "autoencoderModel", "signalMappingModel", "specificEmotionModel", "sharedEmotionModel"]
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

        self.lossColors = [
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
        if submodel == "signalEncoder":
            return metaModel.model.signalEncoderModel
        elif submodel == "autoencoder":
            return metaModel.model.autoencoderModel
        elif submodel == "emotionPrediction":
            return metaModel.model.sharedEmotionModel
        else:
            raise Exception()

    def timeLossComparison(self, allMetaModelPipelines, metaLearnedInfo, userInputParams, plotTitle="AutoEncoder Time Loss Plots"):
        print(f"\nPlotting the {plotTitle} Information")

        # Unpack the model information.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = metaLearnedInfo

        # Update the compiler information for this model.
        self.modelCompiler.addSubmodelParameters(loadSubmodel, userInputParams)

        timeWindows = allMetaModelPipelines[0].model.timeWindows

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
                metaDatasetName = allMetaModelPipelines[metaModelInd].datasetName
                print(metaDatasetName, metaModel.testingLosses_timeLayerAnalysis[timeWindowInd])
                print(metaDatasetName, metaModel.trainingLosses_timeLayerAnalysis[timeWindowInd])
                print(metaDatasetName, metaModel.numEncodingsPath_timeAnalysis[timeWindowInd])

                # Plot the training loss.
                plt.plot(metaModel.trainingLosses_timeLayerAnalysis[timeWindowInd], label=f'{metaDatasetName} Training Loss', color=self.lossColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeLayerAnalysis[timeWindowInd], label=f'{metaDatasetName} Testing Loss', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metaDatasetName} Num Encodings', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.legend(loc="upper right")
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, timeWindow={timeWindow}s")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.saveFigure(saveAutoencoderLossPlots + f"{plotTitle}{timeWindow}.pdf")
            plt.show()

        # ---- Time Analysis Loss versus Epoch, all time windows, one plot per sub model ----
        for metaModelInd in range(len(allMetaModelPipelines)):
            metaModel = self.getSubmodel(allMetaModelPipelines[metaModelInd], loadSubmodel)
            metaDatasetName = allMetaModelPipelines[metaModelInd].datasetName
            # Plot the training loss.
            for timeWindowInd, timeWindow in enumerate(timeWindows):
                plt.plot(metaModel.trainingLosses_timeLayerAnalysis[timeWindowInd], label=f'{metaDatasetName} Training Loss, {timeWindow}s', color=self.lossColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeLayerAnalysis[timeWindowInd], label=f'{metaDatasetName} Testing Loss, {timeWindow}s', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metaDatasetName} Num Encodings, {timeWindow}s', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.legend(loc="upper right")
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, dataset={metaDatasetName}")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.saveFigure(saveAutoencoderLossPlots + f"{plotTitle}{metaDatasetName}.pdf")
            plt.show()

    def reconstructionLossComparison(self, allMetaModelPipelines, metaLearnedInfo, userInputParams, plotTitle="AutoEncoder Reconstruction Loss Plots"):
        print(f"\nPlotting the {plotTitle} Information")

        # Unpack the model information.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = metaLearnedInfo

        # Update the compiler information for this model.
        self.modelCompiler.addSubmodelParameters(loadSubmodel, userInputParams)

        timeWindows = allMetaModelPipelines[0].model.timeWindows

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
                metaDatasetName = allMetaModelPipelines[metaModelInd].datasetName
                print(metaDatasetName, metaModel.trainingLosses_timeReconstructionAnalysis[timeWindowInd])
                print(metaDatasetName, metaModel.testingLosses_timeReconstructionAnalysis[timeWindowInd])
                print(metaDatasetName, metaModel.numEncodingsPath_timeAnalysis[timeWindowInd])

                # Plot the training loss.
                plt.plot(metaModel.trainingLosses_timeReconstructionAnalysis[timeWindowInd], label=f'{metaDatasetName} Training Loss', color=self.lossColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeReconstructionAnalysis[timeWindowInd], label=f'{metaDatasetName} Testing Loss', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metaDatasetName} Num Encodings', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.legend(loc="upper right")
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, timeWindow={timeWindow}s")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.saveFigure(saveAutoencoderLossPlots + f"{plotTitle}{timeWindow}.pdf")
            plt.show()

        # ---- Time Analysis Loss versus Epoch, all time windows, one plot per sub model ----
        for metaModelInd in range(len(allMetaModelPipelines)):
            metaModel = self.getSubmodel(allMetaModelPipelines[metaModelInd], loadSubmodel)
            metaDatasetName = allMetaModelPipelines[metaModelInd].datasetName
            # Plot the training loss.
            for timeWindowInd, timeWindow in enumerate(timeWindows):
                plt.plot(metaModel.trainingLosses_timeReconstructionAnalysis[timeWindowInd], label=f'{metaDatasetName} Training Loss, {timeWindow}s', color=self.lossColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeReconstructionAnalysis[timeWindowInd], label=f'{metaDatasetName} Testing Loss, {timeWindow}s', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metaDatasetName} Num Encodings, {timeWindow}s', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            # no legend for now
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, dataset={metaDatasetName}")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.saveFigure(saveAutoencoderLossPlots + f"{plotTitle}{metaDatasetName}.pdf")
            plt.show()

        # ---- Heatmap of the reconstruction loss, collected only ----
        metaModel = self.getSubmodel(allMetaModelPipelines[0], loadSubmodel)
        metaDatasetName = allMetaModelPipelines[0].datasetName

        # x = time window, y = num encodings
        accuracy = []
        for timeWindowInd, timeWindow in enumerate(timeWindows):
            # find where change in compression factor occurs by using numEncodingsPath_timeAnalysis
            accuracy_per_comp_factor = []
            numEncodings_list = metaModel.numEncodingsPath_timeAnalysis[timeWindowInd]
            for i in range(1, len(numEncodings_list)):
                if numEncodings_list[i] > numEncodings_list[i - 1]:
                    accuracy_per_comp_factor.append(metaModel.trainingLosses_timeReconstructionAnalysis[timeWindowInd][i])
            accuracy_per_comp_factor.append(metaModel.trainingLosses_timeReconstructionAnalysis[timeWindowInd][-1])
            accuracy.append(accuracy_per_comp_factor)
        # plot heatmap from 2d list
        plt.imshow(np.array(accuracy).T, cmap='Blues', interpolation='nearest')
        plt.xlabel('Time Window')
        plt.xticks(range(len(timeWindows)), timeWindows)
        plt.ylabel('Number of Encodings')
        yticks = np.sort(np.unique(np.array(metaModel.numEncodingsPath_timeAnalysis).flatten()))
        print(accuracy)
        plt.yticks(range(len(yticks)), yticks)
        plt.colorbar(label='Reconstruction Loss')
        if self.savingFolder:
            self.saveFigure(saveAutoencoderLossPlots + f"{plotTitle}{metaDatasetName}_heatmap.pdf")
        plt.show()

    def meanLossComparison(self, allMetaModelPipelines, metaLearnedInfo, userInputParams, plotTitle="AutoEncoder Mean Loss Plots"):
        print(f"\nPlotting the {plotTitle} Information")

        # Unpack the model information.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = metaLearnedInfo

        # Update the compiler information for this model.
        self.modelCompiler.addSubmodelParameters(loadSubmodel, userInputParams)

        timeWindows = allMetaModelPipelines[0].model.timeWindows

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
                metaDatasetName = allMetaModelPipelines[metaModelInd].datasetName
                print(metaDatasetName, metaModel.trainingLosses_timeMeanAnalysis[timeWindowInd])
                print(metaDatasetName, metaModel.testingLosses_timeMeanAnalysis[timeWindowInd])
                print(metaDatasetName, metaModel.numEncodingsPath_timeAnalysis[timeWindowInd])

                # Plot the training loss.
                plt.plot(metaModel.trainingLosses_timeMeanAnalysis[timeWindowInd], label=f'{metaDatasetName} Training Loss', color=self.lossColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeMeanAnalysis[timeWindowInd], label=f'{metaDatasetName} Testing Loss', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metaDatasetName} Num Encodings', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.legend(loc="upper right")
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, timeWindow={timeWindow}s")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.saveFigure(saveAutoencoderLossPlots + f"{plotTitle}{timeWindow}.pdf")
            plt.show()

        # ---- Time Analysis Loss versus Epoch, all time windows, one plot per sub model ----
        for metaModelInd in range(len(allMetaModelPipelines)):
            metaModel = self.getSubmodel(allMetaModelPipelines[metaModelInd], loadSubmodel)
            metaDatasetName = allMetaModelPipelines[metaModelInd].datasetName
            # Plot the training loss.
            for timeWindowInd, timeWindow in enumerate(timeWindows):
                plt.plot(metaModel.trainingLosses_timeMeanAnalysis[timeWindowInd], label=f'{metaDatasetName} Training Loss, {timeWindow}s', color=self.lossColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeMeanAnalysis[timeWindowInd], label=f'{metaDatasetName} Testing Loss, {timeWindow}s', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metaDatasetName} Num Encodings, {timeWindow}s', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, dataset={metaDatasetName}")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.saveFigure(saveAutoencoderLossPlots + f"{plotTitle}{metaDatasetName}.pdf")
            plt.show()

    def stdLossComparison(self, allMetaModelPipelines, metaLearnedInfo, userInputParams, plotTitle="AutoEncoder Std Loss Plots"):
        print(f"\nPlotting the {plotTitle} Information")

        # Unpack the model information.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = metaLearnedInfo

        # Update the compiler information for this model.
        self.modelCompiler.addSubmodelParameters(loadSubmodel, userInputParams)

        timeWindows = allMetaModelPipelines[0].model.timeWindows

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
                metaDatasetName = allMetaModelPipelines[metaModelInd].datasetName
                print(metaDatasetName, metaModel.trainingLosses_timeSTDAnalysis[timeWindowInd])
                print(metaDatasetName, metaModel.testingLosses_timeSTDAnalysis[timeWindowInd])
                print(metaDatasetName, metaModel.numEncodingsPath_timeAnalysis[timeWindowInd])

                # Plot the training loss.
                plt.plot(metaModel.trainingLosses_timeSTDAnalysis[timeWindowInd], label=f'{metaDatasetName} Training Loss', color=self.lossColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeSTDAnalysis[timeWindowInd], label=f'{metaDatasetName} Testing Loss', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metaDatasetName} Num Encodings', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.legend(loc="upper right")
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, timeWindow={timeWindow}s")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.saveFigure(saveAutoencoderLossPlots + f"{plotTitle}{timeWindow}.pdf")
            plt.show()

        # ---- Time Analysis Loss versus Epoch, all time windows, one plot per sub model ----
        for metaModelInd in range(len(allMetaModelPipelines)):
            metaModel = self.getSubmodel(allMetaModelPipelines[metaModelInd], loadSubmodel)
            metaDatasetName = allMetaModelPipelines[metaModelInd].datasetName
            # Plot the training loss.
            for timeWindowInd, timeWindow in enumerate(timeWindows):
                plt.plot(metaModel.trainingLosses_timeSTDAnalysis[timeWindowInd], label=f'{metaDatasetName} Training Loss, {timeWindow}s', color=self.lossColors[metaModelInd], linewidth=2, alpha=1)
                plt.plot(metaModel.testingLosses_timeSTDAnalysis[timeWindowInd], label=f'{metaDatasetName} Testing Loss, {timeWindow}s', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5)
                plt.plot(metaModel.numEncodingsPath_timeAnalysis[timeWindowInd], label=f'{metaDatasetName} Num Encodings, {timeWindow}s', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5, linestyle='--')

            # Label the plot.
            plt.xlabel("Training Epoch")
            plt.ylabel("Loss Values")
            plt.title(f"{plotTitle}, dataset={metaDatasetName}")
            plt.yscale('log')

            # Save the figure if desired.
            if self.savingFolder:
                self.saveFigure(saveAutoencoderLossPlots + f"{plotTitle}{metaDatasetName}.pdf")
            plt.show()
    '''
    def autoencoderLossComparison(self, allMetaModelPipelines, allMetaDataLoaders, metaLearnedInfo, modelComparisonInfo, comparingModelInd, plotTitle="Autoencoder Loss Plots"):
        # Initialize saving folder
        saveAutoencoderLossPlots = self.savingFolder + "/Autoencoder Plots/"
        os.makedirs(saveAutoencoderLossPlots, exist_ok=True)
        print("\tPlotting the Autoencoder Loss Information")

        # Load in the previous model weights and attributes.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = metaLearnedInfo
        self.modelMigration.loadModels(allMetaModelPipelines, allMetaDataLoaders, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True)
        # For each metalearning model
        for metaModelInd in range(len(allMetaModelPipelines)):
            metaModel = allMetaModelPipelines[metaModelInd]
            metaDatasetName = metaModel.datasetName

            # Plot the training loss.
            plt.plot(metaModel.model.autoencoderModel.trainingLosses_signalReconstruction, label=f'{metaDatasetName} Training Loss', color=self.lossColors[metaModelInd], linewidth=2, alpha=1)
            plt.plot(metaModel.model.autoencoderModel.testingLosses_signalReconstruction, label=f'{metaDatasetName} Testing Loss', color=self.lossColors[metaModelInd], linewidth=2, alpha=0.5)

        # Load in the previous model weights and attributes.
        loadSubmodel, loadSubmodelDate, loadSubmodelEpochs = modelComparisonInfo
        self.modelMigration.loadModels(allMetaModelPipelines, allMetaDataLoaders, loadSubmodel, loadSubmodelDate, loadSubmodelEpochs, metaTraining=True, loadModelAttributes=True)

        # Plot the training loss.
        metaDatasetName = allMetaModelPipelines[comparingModelInd].datasetName
        plt.plot(allMetaModelPipelines[comparingModelInd].model.autoencoderModel.trainingLosses_signalReconstruction, label=f'{metaDatasetName} Training Loss', color='k', linewidth=2, alpha=1)
        plt.plot(allMetaModelPipelines[comparingModelInd].model.autoencoderModel.testingLosses_signalReconstruction, label=f'{metaDatasetName} Testing Loss', color='k', linewidth=2, alpha=0.5)

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