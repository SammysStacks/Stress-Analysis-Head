# General

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from torchviz import make_dot

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class generalVisualizations(globalPlottingProtocols):

    def __init__(self, baseSavingFolder, stringID, datasetName):
        super(generalVisualizations, self).__init__(interactivePlots=False)
        self.setSavingFolder(baseSavingFolder, stringID, datasetName)
        
    # ---------------------------------------------------------------------- #
    # --------------------- Visualize Model Parameters --------------------- #
                
    @staticmethod
    def plotGradientFlow(model, currentVar, saveName):
        make_dot(currentVar, params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True).render(saveName, format="svg")

    @staticmethod
    def plotPredictions(allTrainingLabels, allTestingLabels, allPredictedTrainingLabels, allPredictedTestingLabels, numClasses, plotTitle="Emotion Prediction"):
        # Plot the data correlation.
        plt.plot(allPredictedTrainingLabels, allTrainingLabels, 'ko', markersize=6, alpha=0.3, label="Training Points")
        plt.plot(allPredictedTestingLabels, allTestingLabels, '*', color='tab:blue', markersize=6, alpha=0.6, label="Testing Points")
        plt.xlabel("Predicted Emotion Rating")
        plt.xlim((-0.1, numClasses-0.9))
        plt.ylim((-0.1, numClasses-0.9))
        plt.ylabel("Emotion Rating")
        plt.title(f"{plotTitle}")
        plt.legend(loc="best")
        plt.show()

    @staticmethod
    def plotDistributions(trueDistributions, predictionDistributions, numClasses, plotTitle="Emotion Distributions"):
        assert (predictionDistributions >= 0).all()
        assert (trueDistributions >= 0).all()

        # Plot the data correlation.
        xAxis = np.arange(0, numClasses, numClasses /
                          len(trueDistributions[0])) - 0.5
        # plt.plot(xAxis, trueDistributions[0], 'k', linewidth=2, alpha = 0.4,label = "True Emotion Distribution")
        plt.plot(xAxis, predictionDistributions[0], 'tab:red',
                 linewidth=2, alpha=0.6, label="Predicted Emotion Distribution")
        plt.ylabel("Probability (au)")
        plt.xlabel("Emotion Rating")
        plt.title(f"{plotTitle}")
        plt.legend(loc="best")
        plt.show()

    def plotPredictedMatrix(self, allTrainingLabels, allTestingLabels, allPredictedTrainingLabels, allPredictedTestingLabels, numClasses, epoch, emotionName, saveFigureLocation):
        # Assert the correct data format
        allTestingLabels = np.asarray(allTestingLabels)
        allTrainingLabels = np.asarray(allTrainingLabels)
        allPredictedTestingLabels = np.asarray(allPredictedTestingLabels)
        allPredictedTrainingLabels = np.asarray(allPredictedTrainingLabels)

        # Calculate confusion matrices
        training_confusion_matrix = confusion_matrix(
            allTrainingLabels, allPredictedTrainingLabels, labels=np.arange(numClasses), normalize='true')
        testing_confusion_matrix = confusion_matrix(
            allTestingLabels, allPredictedTestingLabels, labels=np.arange(numClasses), normalize='true')

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
        fig.suptitle(f"{emotionName}", fontsize=24)
        fig.tight_layout()

        # Save the figure is desired.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=None, baseSaveFigureName=f"{emotionName}.pdf", fig=fig, clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotTrainingLosses(self, trainingLosses, testingLosses, lossLabels, saveFigureLocation="", plotTitle="Model Convergence Loss", logY=True):
        # Assert the validity of the input data.
        assert len(trainingLosses) == len(lossLabels), "Number of loss labels must match the number of loss indices."
        if len(trainingLosses[0]) == 0: return None
        numModels, numEpochs = len(trainingLosses), len(trainingLosses[0])
        fig, ax = plt.subplots(figsize=(6.4, 4.8))

        # Plot the average losses.
        for modelInd in range(numModels):
            modelTrainingLosses = np.asarray(trainingLosses[modelInd])
            # modelTrainingLosses: numEpochs, numSignals

            # Calculate the average and standard deviation of the training losses.
            N = np.count_nonzero(~np.isnan(modelTrainingLosses), axis=1)
            trainingStandardError = np.nanstd(modelTrainingLosses, ddof=1, axis=-1) / np.sqrt(N)
            trainingLoss = np.nanmean(modelTrainingLosses, axis=-1)

            # Plot the training losses.
            ax.errorbar(x=np.arange(len(trainingLoss)), y=trainingLoss, yerr=trainingStandardError, color=self.darkColors[modelInd], linewidth=1)
            ax.plot(modelTrainingLosses, color=self.darkColors[modelInd], linewidth=1, alpha=0.05)

            if testingLosses is not None:
                modelTestingLosses = np.asarray(testingLosses[modelInd])
                # Calculate the average and standard deviation of the testing losses.
                N = np.count_nonzero(~np.isnan(modelTestingLosses), axis=-1)
                testingStd = np.nanstd(modelTestingLosses, ddof=1, axis=-1) / np.sqrt(N)
                testingLoss = np.nanmean(modelTestingLosses, axis=-1)
                modelTestingLosses[np.isnan(modelTestingLosses)] = None

                # Plot the testing losses.
                ax.errorbar(x=np.arange(len(testingLoss)), y=testingLoss, yerr=testingStd, color=self.darkColors[modelInd], linewidth=1)
                ax.plot(modelTestingLosses, '-', color=self.darkColors[modelInd], linewidth=1, alpha=0.025)

        # Plot gridlines.
        ax.hlines(y=0.1, xmin=0, xmax=len(trainingLosses[0]) + 1, colors=self.blackColor, linestyles='dashed', linewidth=1)
        for i in range(2, 10): ax.hlines(y=0.01*i, xmin=0, xmax=len(trainingLosses[0]) + 1, colors=self.blackColor, linestyles='dashed', linewidth=1, alpha=0.25)
        ax.hlines(y=0.01, xmin=0, xmax=len(trainingLosses[0]) + 1, colors=self.blackColor, linestyles='dashed', linewidth=1)
        ax.set_xlim((0, max(128 if 'profile' not in plotTitle.lower() else 0, len(trainingLosses[0]) + 1)))
        ax.set_ylim((0.0025, 0.75))
        ax.grid(True)

        # Label the plot.
        if logY: ax.set_yscale('log')
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Loss values")
        ax.set_title(f"{plotTitle}", fontsize=16)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1), borderaxespad=0)

        # Save the figure if desired.
        saveFigureName = f"{plotTitle}.pdf" if "profile" in plotTitle.lower() else None
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=saveFigureName, baseSaveFigureName=f"{plotTitle}.pdf", fig=fig, clearFigure=True, showPlot=not self.hpcFlag)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotActivationFlowCompressed(self, activationParamsPaths, moduleNames, modelLabels, paramNames, saveFigureLocation="", plotTitle="Model Convergence Loss"):
        activationParamsPaths = np.asarray(activationParamsPaths)
        if len(activationParamsPaths.shape) == 2: return "No data to plot."

        # activationParamsPaths: numModels, numEpochs, numModuleLayers, numParams
        numModels, numEpochs, numModuleLayers, numActivationParams = activationParamsPaths.shape
        nRows, nCols = min(1, numActivationParams // 3), numActivationParams
        numParams = len(paramNames)

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex=True, sharey='col')
        axes = axes.flatten()  # Flatten axes for easy indexing if you prefer

        for paramInd in range(numParams):
            ax = axes[paramInd]  # which subplot to use
            paramName = paramNames[paramInd]

            for modelInd in range(numModels):
                for layerInd in range(numModuleLayers):
                    activationParams = activationParamsPaths[modelInd, :, layerInd, paramInd]
                    moduleName = moduleNames[modelInd, layerInd].lower()

                    # Remove shared layers
                    if "shared" in moduleName and modelInd != 0: continue

                    # Set the line color and alpha
                    if "specific" in moduleName: lineColor = self.darkColors[modelInd]; alpha = 0.8
                    elif "shared" in moduleName: lineColor = self.blackColor; alpha = 0.75
                    else: raise ValueError("Module name must contain 'specific' or 'shared'.")

                    if modelInd == 0: modelLabel = modelLabels[modelInd]
                    else: modelLabel = None

                    # Plot the activation parameters.
                    ax.plot(activationParams, color=lineColor, linewidth=0.8, alpha=alpha, label=modelLabel)
            ax.set_xlabel("Epoch (training)")
            ax.set_title(paramName, fontsize=16)
            if 'Infinite' in paramName: ax.set_ylim((0, 1.1))
            elif 'Linearity' in paramName: ax.set_ylim((0, 10.1))
            elif 'Convergent' in paramName: ax.set_ylim((0, 2.1))
            ax.set_xlim((0, numEpochs))
            ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.8)

        # Label the plot.
        fig.suptitle(f"{plotTitle}", fontsize=24)
        fig.tight_layout()

        # Save the figure if desired.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", fig=fig, clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotActivationFlow(self, activationParamsPaths, moduleNames, paramNames, saveFigureLocation="", plotTitle="Model Convergence Loss"):
        activationParamsPaths = np.asarray(activationParamsPaths)
        if len(activationParamsPaths.shape) == 2: return "No data to plot."

        # activationParamsPaths: numModels, numEpochs, numModuleLayers, numParams
        numModels, numEpochs, numModuleLayers, numActivationParams = activationParamsPaths.shape
        nRows, nCols = self.getRowsCols(combineSharedLayers=True)
        numParams = len(paramNames)
        x = np.arange(numEpochs)

        # Get the parameters.
        numSpecificLayers = modelConstants.userInputParams['numSpecificEncoderLayers']
        numSharedLayers = modelConstants.userInputParams['numSharedEncoderLayers']

        for paramInd in range(numParams):
            # Create a figure and axes array
            fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex=True, sharey=True)
            numLow, numSpecificHigh, numSharedHigh = 0, 0, 0
            paramName = paramNames[paramInd]

            for layerInd in range(numModuleLayers):
                moduleName = moduleNames[0][layerInd].lower()

                if "low" in moduleName: rowInd, colInd = numLow, nCols - 1; numLow += 1
                elif "high" in moduleName:
                    if 'shared' in moduleName: rowInd = numSpecificLayers + (numSharedHigh % numSharedLayers); colInd = numSharedHigh // numSharedLayers; numSharedHigh += 1
                    elif 'specific' in moduleName: rowInd = numSpecificHigh % numSpecificLayers; colInd = numSpecificHigh // numSpecificLayers; numSpecificHigh += 1
                    else: raise ValueError("Module name must contain 'shared' or 'specific'")
                else: raise ValueError("Module name must contain 'low' or 'high'.")
                rowInd = min(rowInd, nRows - 1)
                ax = axes[rowInd, colInd]

                # Label the plot.
                if colInd == 0 and 'shared' in moduleName.lower(): ax.set_ylabel("Shared layers", fontsize=16)
                if colInd == 0 and 'specific' in moduleName.lower(): ax.set_ylabel("Specific layers", fontsize=16)

                if 'infinite' in paramName.lower(): ax.set_ylim((0, 1.1))
                elif 'linearity' in paramName.lower(): ax.set_ylim((0, 10.1))
                elif 'convergent' in paramName.lower(): ax.set_ylim((0, 2.1))

                ax.set_title(moduleName.capitalize(), fontsize=16)
                ax.set_xlabel("Training epoch")
                ax.set_xlim((0, numEpochs))

                for modelInd in range(numModels):
                    if "shared" in moduleName and modelInd != 0: continue

                    if "specific" in moduleName: lineColor = self.darkColors[modelInd]
                    elif "shared" in moduleName: lineColor = self.blackColor
                    else: raise ValueError("Module name must contain 'specific' or 'shared'.")

                    # Plot the training losses.
                    plottingParams = activationParamsPaths[modelInd, :, layerInd, paramInd]
                    if 'specific' in moduleName: ax.plot(x, plottingParams, color=lineColor, linewidth=1, alpha=1)
                    else:
                        numValues = 1 if len(plottingParams.shape) == 1 else plottingParams.shape[1]
                        alphas = np.linspace(0, 1, numValues)
                        for axisLineInd in range(numValues):
                            ax.plot(x, plottingParams, color=self.darkColors[0], linewidth=1, alpha=0.3 * alphas[axisLineInd])
                            ax.plot(x, plottingParams, color=self.darkColors[1], linewidth=1, alpha=0.6 * (1 - alphas[axisLineInd]))
                ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.8)

            # Label the plot.
            fig.suptitle(f"{plotTitle}: {paramName}", fontsize=24)
            fig.tight_layout()

            for specificLayerInd in range(numSpecificLayers):
                for colInd in range(nCols):
                    if colInd == 0 or colInd == nCols - 1: continue
                    axes[specificLayerInd, colInd].remove()

            # Save the figure if desired.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} {paramName}.pdf", baseSaveFigureName=f"{plotTitle} {paramName}.pdf", fig=fig, clearFigure=True, showPlot=False)
            else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotFreeParamFlow(self, numFreeModelParams, maxFreeParamsPath, moduleNames, paramNames, saveFigureLocation="", plotTitle="Model Convergence Loss"):
        # normalizationFactorsPaths: numModels, numEpochs, numModuleLayers, *numSignals*, numParams=1
        # maxFreeParamsPath: numModels, numModuleLayers
        numModels, numModuleLayers = np.asarray(maxFreeParamsPath).shape
        nRows, nCols = self.getRowsCols(combineSharedLayers=True)
        numEpochs = len(numFreeModelParams[0])
        numParams = len(paramNames)
        x = np.arange(numEpochs)
        if numEpochs <= 5: return "No data to plot."

        # Get the parameters.
        numSpecificLayers = modelConstants.userInputParams['numSpecificEncoderLayers']
        numSharedLayers = modelConstants.userInputParams['numSharedEncoderLayers']

        for paramInd in range(numParams):
            # Create a figure and axes array
            fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex=True, sharey=False)
            numLow, numSpecificHigh, numSharedHigh = 0, 0, 0
            paramName = paramNames[paramInd].lower()

            for layerInd in range(numModuleLayers):
                moduleName = moduleNames[0][layerInd].lower()

                if "low" in moduleName: rowInd, colInd = numLow, nCols - 1; numLow += 1
                elif "high" in moduleName:
                    if 'shared' in moduleName: rowInd = numSpecificLayers + (numSharedHigh % numSharedLayers); colInd = numSharedHigh // numSharedLayers; numSharedHigh += 1
                    elif 'specific' in moduleName: rowInd = numSpecificHigh % numSpecificLayers; colInd = numSpecificHigh // numSpecificLayers; numSpecificHigh += 1
                    else: raise ValueError("Module name must contain 'shared' or 'specific'")
                else: raise ValueError("Module name must contain 'low' or 'high'.")
                rowInd = min(rowInd, nRows - 1)
                ax = axes[rowInd, colInd]

                if colInd == 0: ax.set_ylabel("Number of rotations")
                ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.8)
                ax.set_xlabel("Training epoch")
                ax.set_xlim((0, numEpochs))
                ax.set_title(moduleName, fontsize=16)

                for modelInd in range(numModels):
                    if "shared" in moduleName and modelInd != 0: continue
                    maxFreeParams = maxFreeParamsPath[modelInd][layerInd]
                    sequenceLength = int((1 + (1 + 8 * maxFreeParams) ** 0.5) // 2)
                    if sequenceLength == 1: sequenceLength = 0

                    plottingParams = []
                    for epochInd in range(numEpochs):
                        plottingParams.append(numFreeModelParams[modelInd][epochInd][layerInd][:, paramInd])
                    plottingParams = np.asarray(plottingParams)

                    # Plot the training losses.
                    if 'specific' in moduleName:
                        # Calculate the average and standard deviation of the training losses.
                        N = np.count_nonzero(~np.isnan(plottingParams), axis=1)
                        standardError = np.nanstd(plottingParams, ddof=1, axis=-1) / np.sqrt(N)
                        meanValues = np.nanmean(plottingParams, axis=-1)

                        ax.errorbar(x=x, y=meanValues, yerr=standardError, color=self.darkColors[modelInd], linewidth=1)
                        ax.plot(x, plottingParams, color=self.darkColors[modelInd], linewidth=1, alpha=0.05)
                    else:
                        numValues = 1 if len(plottingParams.shape) == 1 else plottingParams.shape[1]
                        alphas = np.linspace(0, 1, numValues)
                        for axisLineInd in range(numValues):
                            ax.plot(x, plottingParams, color=self.darkColors[0], linewidth=1, alpha=0.3 * alphas[axisLineInd])
                            ax.plot(x, plottingParams, color=self.darkColors[1], linewidth=1, alpha=0.6 * (1 - alphas[axisLineInd]))
                    ax.hlines(y=sequenceLength, xmin=0, xmax=numEpochs + 1, colors=self.blackColor, linestyles='dashed', linewidth=1)
                    ax.hlines(y=maxFreeParams, xmin=0, xmax=numEpochs + 1, colors=self.blackColor, linestyles='dashed', linewidth=1)

            # Label the plot.
            fig.suptitle(f"{plotTitle}: {paramName}", fontsize=24)
            fig.tight_layout()

            for specificLayerInd in range(numSpecificLayers):
                for colInd in range(nCols):
                    if colInd == 0 or colInd == nCols - 1: continue
                    axes[specificLayerInd, colInd].remove()

            # Save the figure if desired.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=None, baseSaveFigureName=f"{plotTitle} {paramName}.pdf", fig=fig, clearFigure=False, showPlot=False)
            else: self.clearFigure(fig=fig, legend=None, showPlot=True)

            # Zoom into the plot.
            for ax in fig.axes: ax.set_ylim(0, 2500)

            # Save the figure if desired.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=None, baseSaveFigureName=f"{plotTitle} {paramName} (zoomed).pdf", fig=fig, clearFigure=True, showPlot=not self.hpcFlag)
            else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotNormalizationFactorFlow(self, normalizationFactorsPaths, moduleNames, paramNames, saveFigureLocation="", plotTitle="Model Convergence Loss"):
        # normalizationFactorsPaths: numModels, numEpochs, numModuleLayers, *numSignals*, numParams=1
        try: numModels, numEpochs, numModuleLayers = len(normalizationFactorsPaths), len(normalizationFactorsPaths[0]), len(normalizationFactorsPaths[0][0])
        except Exception as e: print("plotAngularFeaturesFlow:", e); return None
        nRows, nCols = self.getRowsCols(combineSharedLayers=True)
        numParams = len(paramNames)
        x = np.arange(numEpochs)

        # Get the parameters.
        numSpecificLayers = modelConstants.userInputParams['numSpecificEncoderLayers']
        numSharedLayers = modelConstants.userInputParams['numSharedEncoderLayers']

        for paramInd in range(numParams):
            # Create a figure and axes array
            fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex=True, sharey=True)
            numLow, numSpecificHigh, numSharedHigh = 0, 0, 0
            paramName = paramNames[paramInd]

            for layerInd in range(numModuleLayers):
                moduleName = moduleNames[0][layerInd].lower()

                if "low" in moduleName: rowInd, colInd = numLow, nCols - 1; numLow += 1
                elif "high" in moduleName:
                    if 'shared' in moduleName: rowInd = numSpecificLayers + (numSharedHigh % numSharedLayers); colInd = numSharedHigh // numSharedLayers; numSharedHigh += 1
                    elif 'specific' in moduleName: rowInd = numSpecificHigh % numSpecificLayers; colInd = numSpecificHigh // numSpecificLayers; numSpecificHigh += 1
                    else: raise ValueError("Module name must contain 'shared' or 'specific'")
                else: raise ValueError("Module name must contain 'low' or 'high'.")
                rowInd = min(rowInd, nRows - 1)
                ax = axes[rowInd, colInd]

                ax.set_xlabel("Training epoch")
                ax.set_title(moduleName, fontsize=16)
                ax.set_xlim((0, numEpochs))
                ax.set_ylim((0.925, 1.075))

                for modelInd in range(numModels):
                    if "shared" in moduleName and modelInd != 0: continue

                    plottingParams = []
                    for epochInd in range(numEpochs):
                        plottingParams.append(normalizationFactorsPaths[modelInd][epochInd][layerInd][:, paramInd])
                    plottingParams = np.asarray(plottingParams)

                    # Plot the training losses.
                    if 'specific' in moduleName:
                        # Calculate the average and standard deviation of the training losses.
                        N = np.count_nonzero(~np.isnan(plottingParams), axis=1)
                        standardError = np.nanstd(plottingParams, ddof=1, axis=-1) / np.sqrt(N)
                        meanValues = np.nanmean(plottingParams, axis=-1)

                        ax.errorbar(x=x, y=meanValues, yerr=standardError, color=self.darkColors[modelInd], linewidth=1)
                        ax.plot(x, plottingParams, color=self.darkColors[modelInd], linewidth=1, alpha=0.05)
                    else:
                        numValues = 1 if len(plottingParams.shape) == 1 else plottingParams.shape[1]
                        alphas = np.linspace(0, 1, numValues)
                        for axisLineInd in range(numValues):
                            ax.plot(x, plottingParams, color=self.darkColors[0], linewidth=1, alpha=0.3 * alphas[axisLineInd])
                            ax.plot(x, plottingParams, color=self.darkColors[1], linewidth=1, alpha=0.6 * (1 - alphas[axisLineInd]))
                ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.8)

            # Label the plot.
            fig.suptitle(f"{plotTitle}: {paramName}", fontsize=24)
            fig.tight_layout()

            for specificLayerInd in range(numSpecificLayers):
                for colInd in range(nCols):
                    if colInd == 0 or colInd == nCols - 1: continue
                    axes[specificLayerInd, colInd].remove()

            # Save the figure if desired.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=None, baseSaveFigureName=f"{plotTitle} {paramName}.pdf", fig=fig, clearFigure=True, showPlot=False)
            else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotGivensFeaturesPath(self, givensAnglesFeaturesPaths, moduleNames, paramNames, saveFigureLocation="", plotTitle="Model Convergence Loss"):
        # givensAnglesFeaturesPaths: numModels, numEpochs, numModuleLayers, numFeatures=5, numFeatureValues*
        try: numModels, numEpochs, numModuleLayers = len(givensAnglesFeaturesPaths), len(givensAnglesFeaturesPaths[0]), len(givensAnglesFeaturesPaths[0][0])
        except Exception as e: print("plotAngularFeaturesFlow:", e); return None
        nRows, nCols = self.getRowsCols(combineSharedLayers=True)
        numParams = len(paramNames)
        x = np.arange(numEpochs)

        # Get the parameters.
        numSpecificLayers = modelConstants.userInputParams['numSpecificEncoderLayers']
        numSharedLayers = modelConstants.userInputParams['numSharedEncoderLayers']

        for paramInd in range(numParams):
            # Create a figure and axes array
            fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex=True, sharey='col')
            numLow, numSpecificHigh, numSharedHigh = 0, 0, 0
            paramName = paramNames[paramInd]

            fig.suptitle(f"{plotTitle}: {paramName}", fontsize=24)
            for layerInd in range(numModuleLayers):
                moduleName = moduleNames[0][layerInd].lower()

                if "low" in moduleName: rowInd, colInd = numLow, nCols - 1; numLow += 1
                elif "high" in moduleName:
                    if 'shared' in moduleName: rowInd = numSpecificLayers + (numSharedHigh % numSharedLayers); colInd = numSharedHigh // numSharedLayers; numSharedHigh += 1
                    elif 'specific' in moduleName: rowInd = numSpecificHigh % numSpecificLayers; colInd = numSpecificHigh // numSpecificLayers; numSpecificHigh += 1
                    else: raise ValueError("Module name must contain 'shared' or 'specific'")
                else: raise ValueError("Module name must contain 'low' or 'high'.")
                rowInd = min(rowInd, nRows - 1)
                ax = axes[rowInd, colInd]

                # Label the plot.
                ax.set_xlabel("Training epoch")
                ax.set_title(moduleName, fontsize=16)
                ax.set_xlim((0, numEpochs))

                for modelInd in range(numModels):
                    if "shared" in moduleName and modelInd != 0: continue
                    if "specific" in moduleName: lineColor = self.darkColors[modelInd]
                    elif "shared" in moduleName: lineColor = self.blackColor
                    else: raise ValueError("Module name must contain 'specific' or 'shared'.")

                    plottingParams = np.zeros((numEpochs, len(givensAnglesFeaturesPaths[modelInd][0][layerInd][paramInd])))
                    for epochInd in range(numEpochs): plottingParams[epochInd, :] = givensAnglesFeaturesPaths[modelInd][epochInd][layerInd][paramInd]
                    # plottingParams: numEpochs, numFeatureValues

                    # Plot the training losses.
                    if 'specific' in moduleName:
                        # Calculate the average and standard deviation of the training losses.
                        N = np.count_nonzero(~np.isnan(plottingParams), axis=1)
                        standardError = np.nanstd(plottingParams, ddof=1, axis=-1) / np.sqrt(N)
                        meanValues = np.nanmean(plottingParams, axis=-1)

                        ax.errorbar(x=x, y=meanValues, yerr=standardError, color=lineColor, linewidth=1)
                        ax.plot(x, plottingParams, color=lineColor, linewidth=1, alpha=0.05)
                    else:
                        numValues = 1 if len(plottingParams.shape) == 1 else plottingParams.shape[1]
                        alphas = np.linspace(0, 1, numValues)
                        for axisLineInd in range(numValues):
                            ax.plot(x, plottingParams, color=self.darkColors[0], linewidth=1, alpha=0.3 * alphas[axisLineInd])
                            ax.plot(x, plottingParams, color=self.darkColors[1], linewidth=1, alpha=0.6 * (1 - alphas[axisLineInd]))
                ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.8)
            fig.tight_layout()

            for specificLayerInd in range(numSpecificLayers):
                for colInd in range(nCols):
                    if colInd == 0 or colInd == nCols - 1: continue
                    axes[specificLayerInd, colInd].remove()

            # Save the figure if desired.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=None, baseSaveFigureName=f"{plotTitle} {paramName}.pdf", fig=fig, clearFigure=True, showPlot=False)
            else: self.clearFigure(fig=fig, legend=None, showPlot=True)
