import matplotlib.pyplot as plt
import numpy as np

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class generalVisualizations(globalPlottingProtocols):

    def __init__(self, baseSavingFolder, stringID, datasetName):
        super(generalVisualizations, self).__init__(interactivePlots=False)
        self.setSavingFolder(baseSavingFolder, stringID, datasetName)
        
    # --------------------- Visualize Model Parameters --------------------- #

    def plotTrainingLosses(self, trainingLosses, testingLosses, numEpochs, saveFigureLocation="", plotTitle="Model Convergence Loss", logY=True):
        # Assert the validity of the input data.
        if len(trainingLosses[0]) == 0: return None
        profileFlag = 'profile' in plotTitle.lower()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        numModels = len(trainingLosses)
        emotionModel = 'emotion' in plotTitle.lower() or 'activity' in plotTitle.lower()

        # Plot the average losses.
        for modelInd in range(numModels):
            modelTrainingLosses = np.asarray(trainingLosses[modelInd])
            # modelTrainingLosses: numEpochs, numSignals or emotions or 0

            # Plot the training losses.
            if not emotionModel:
                # Calculate the average and standard deviation of the training losses.
                N = np.count_nonzero(~np.isnan(modelTrainingLosses), axis=1)
                trainingStandardError = np.nanstd(modelTrainingLosses, ddof=1, axis=-1) / np.sqrt(N)
                trainingLoss = np.nanmean(modelTrainingLosses, axis=-1)

                ax.errorbar(x=np.arange(len(trainingLoss)), y=trainingLoss, yerr=trainingStandardError, color=self.darkColors[modelInd], linewidth=1)
            ax.plot(modelTrainingLosses, color=self.darkColors[modelInd], linewidth=1, alpha=0.05 if not emotionModel else 1)

            if testingLosses is not None:
                modelTestingLosses = np.asarray(testingLosses[modelInd])
                modelTestingLosses[np.isnan(modelTestingLosses)] = None

                # Plot the testing losses.
                if not emotionModel:
                    # Calculate the average and standard deviation of the testing losses.
                    N = np.count_nonzero(~np.isnan(modelTestingLosses), axis=-1)
                    testingStd = np.nanstd(modelTestingLosses, ddof=1, axis=-1) / np.sqrt(N)
                    testingLoss = np.nanmean(modelTestingLosses, axis=-1)

                    ax.errorbar(x=np.arange(len(testingLoss)), y=testingLoss, yerr=testingStd, color=self.darkColors[modelInd], linewidth=1)
                ax.plot(modelTestingLosses, '-', color=self.darkColors[modelInd], linewidth=1, alpha=0.025 if not emotionModel else 0.75)

        # Plot gridlines.
        ax.hlines(y=0.1, xmin=0, xmax=len(trainingLosses[0]) + 1, colors=self.blackColor, linestyles='dashed', linewidth=1)
        for i in range(2, 10): ax.hlines(y=0.01*i, xmin=0, xmax=len(trainingLosses[0]) + 1, colors=self.blackColor, linestyles='dashed', linewidth=1, alpha=0.25)
        ax.hlines(y=0.01, xmin=0, xmax=len(trainingLosses[0]) + 1, colors=self.blackColor, linestyles='dashed', linewidth=1)
        ax.set_xlim((0, max(128 if not profileFlag else 0, len(trainingLosses[0]) + 1)))
        ax.set_ylim((0.0025, 0.75 if not emotionModel else 10))
        ax.grid(True)

        # Label the plot.
        if logY: ax.set_yscale('log')
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Loss values")
        ax.set_title(f"{plotTitle} at epoch {numEpochs}", fontsize=16)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1), borderaxespad=0)

        # Save the figure if desired.
        saveFigureName = f"{plotTitle} epoch{numEpochs}.pdf" if "profile" in plotTitle.lower() else None
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
                    if 'shared' in moduleName and modelInd != 0: continue

                    # Set the line color and alpha
                    if 'specific' in moduleName: lineColor = self.darkColors[modelInd]; alpha = 0.8
                    elif 'shared' in moduleName: lineColor = self.blackColor; alpha = 0.75
                    else: raise ValueError("Module name must contain 'specific' or 'shared'.")

                    if modelInd == 0: modelLabel = modelLabels[modelInd]
                    else: modelLabel = None

                    # Plot the activation parameters.
                    ax.plot(activationParams, color=lineColor, linewidth=0.8, alpha=alpha, label=modelLabel)
            ax.set_xlabel("Epoch (training)")
            ax.set_title(f"{paramName} at epoch {numEpochs}", fontsize=16)
            if 'Infinite' in paramName: ax.set_ylim((0, 1.1))
            elif 'Linearity' in paramName: ax.set_ylim((0, 10.1))
            elif 'Convergent' in paramName: ax.set_ylim((0, 2.1))
            ax.set_xlim((0, numEpochs))
            ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.8)

        # Label the plot.
        fig.suptitle(f"{plotTitle}", fontsize=24)
        fig.tight_layout()

        # Save the figure if desired.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epoch{numEpochs}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", fig=fig, clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotActivationFlow(self, activationParamsPaths, moduleNames, paramNames, saveFigureLocation="", plotTitle="Model Convergence Loss"):
        activationParamsPaths = np.asarray(activationParamsPaths)
        if len(activationParamsPaths.shape) == 2: return "No data to plot."

        # activationParamsPaths: numModels, numEpochs, numModuleLayers, numParams
        numModels, numEpochs, numModuleLayers, numActivationParams = activationParamsPaths.shape
        nRows, nCols = self.getRowsCols(combineSharedLayers=True, saveFigureLocation=saveFigureLocation)
        numParams = len(paramNames)
        x = np.arange(numEpochs)

        # Get the parameters.
        prefix = "numSharedEncoder" if modelConstants.userInputParams['submodel'] == modelConstants.signalEncoderModel else ("numActivityModel" if "activity" in saveFigureLocation.lower() else "numEmotionModel")
        numSpecificLayers = modelConstants.userInputParams['numSpecificEncoderLayers'] if modelConstants.userInputParams['submodel'] == modelConstants.signalEncoderModel else 1
        numSharedLayers = modelConstants.userInputParams[f'{prefix}Layers']

        for paramInd in range(numParams):
            # Create a figure and axes array
            fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex=True, sharey=True)
            numLow, numSpecificHigh, numSharedHigh = 0, 0, 0
            paramName = paramNames[paramInd]

            for layerInd in range(numModuleLayers):
                moduleName = moduleNames[0][layerInd].lower()

                if 'low' in moduleName or 'real' in moduleName: rowInd, colInd = numLow, nCols - 1; numLow += 1
                elif 'high' in moduleName or 'imaginary' in moduleName:
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
                    if 'shared' in moduleName and modelInd != 0: continue

                    if 'specific' in moduleName: lineColor = self.darkColors[modelInd]
                    elif 'shared' in moduleName: lineColor = self.blackColor
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
            fig.suptitle(f"{plotTitle}: {paramName.lower()} at epoch {numEpochs}", fontsize=24)
            fig.tight_layout()

            for specificLayerInd in range(numSpecificLayers):
                for colInd in range(nCols):
                    if colInd == 0 or colInd == nCols - 1: continue
                    axes[specificLayerInd, colInd].remove()

            # Save the figure if desired.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} {paramName} epoch{numEpochs}.pdf", baseSaveFigureName=f"{plotTitle} {paramName}.pdf", fig=fig, clearFigure=True, showPlot=False)
            else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotFreeParamFlow(self, numFreeModelParams, maxFreeParamsPath, moduleNames, paramNames, saveFigureLocation="", plotTitle="Model Convergence Loss"):
        # normalizationFactorsPaths: numModels, numEpochs, numModuleLayers, *numSignals*, numParams=1
        # maxFreeParamsPath: numModels, numModuleLayers
        numModels, numModuleLayers = np.asarray(maxFreeParamsPath).shape
        nRows, nCols = self.getRowsCols(combineSharedLayers=True, saveFigureLocation=saveFigureLocation)
        numEpochs = len(numFreeModelParams[0])
        numParams = len(paramNames)
        x = np.arange(numEpochs)
        if numEpochs <= 5: return "No data to plot."

        # Get the parameters.
        prefix = "numSharedEncoder" if modelConstants.userInputParams['submodel'] == modelConstants.signalEncoderModel else ("numActivityModel" if "activity" in saveFigureLocation.lower() else "numEmotionModel")
        numSpecificLayers = modelConstants.userInputParams['numSpecificEncoderLayers'] if modelConstants.userInputParams['submodel'] == modelConstants.signalEncoderModel else 1
        numSharedLayers = modelConstants.userInputParams[f'{prefix}Layers']

        for paramInd in range(numParams):
            # Create a figure and axes array
            fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex=True, sharey=False)
            numLow, numSpecificHigh, numSharedHigh = 0, 0, 0
            paramName = paramNames[paramInd].lower()

            for layerInd in range(numModuleLayers):
                moduleName = moduleNames[0][layerInd].lower()

                if 'low' in moduleName or 'real' in moduleName: rowInd, colInd = numLow, nCols - 1; numLow += 1
                elif 'high' in moduleName or 'imaginary' in moduleName:
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
                    if 'shared' in moduleName and modelInd != 0: continue
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
            fig.suptitle(f"{plotTitle}: {paramName.lower()} at epoch {numEpochs}", fontsize=24)
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
        nRows, nCols = self.getRowsCols(combineSharedLayers=True, saveFigureLocation=saveFigureLocation)
        numParams = len(paramNames)
        x = np.arange(numEpochs)

        # Get the parameters.
        prefix = "numSharedEncoder" if modelConstants.userInputParams['submodel'] == modelConstants.signalEncoderModel else ("numActivityModel" if "activity" in saveFigureLocation.lower() else "numEmotionModel")
        numSpecificLayers = modelConstants.userInputParams['numSpecificEncoderLayers'] if modelConstants.userInputParams['submodel'] == modelConstants.signalEncoderModel else 1
        numSharedLayers = modelConstants.userInputParams[f'{prefix}Layers']

        for paramInd in range(numParams):
            # Create a figure and axes array
            fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex=True, sharey=True)
            numLow, numSpecificHigh, numSharedHigh = 0, 0, 0
            paramName = paramNames[paramInd]

            for layerInd in range(numModuleLayers):
                moduleName = moduleNames[0][layerInd].lower()

                if 'low' in moduleName or 'real' in moduleName: rowInd, colInd = numLow, nCols - 1; numLow += 1
                elif 'high' in moduleName or 'imaginary' in moduleName:
                    if 'shared' in moduleName: rowInd = numSpecificLayers + (numSharedHigh % numSharedLayers); colInd = numSharedHigh // numSharedLayers; numSharedHigh += 1
                    elif 'specific' in moduleName: rowInd = numSpecificHigh % numSpecificLayers; colInd = numSpecificHigh // numSpecificLayers; numSpecificHigh += 1
                    else: raise ValueError("Module name must contain 'shared' or 'specific'")
                else: raise ValueError("Module name must contain 'low' or 'high'.")
                rowInd = min(rowInd, nRows - 1)
                ax = axes[rowInd, colInd]

                ax.set_xlabel("Training epoch")
                ax.set_title(moduleName, fontsize=16)
                ax.set_xlim((0, numEpochs))
                ax.set_ylim((0.95, 1.05))

                for modelInd in range(numModels):
                    if 'shared' in moduleName and modelInd != 0: continue

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
            fig.suptitle(f"{plotTitle}: {paramName.lower()}", fontsize=24)
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
        nRows, nCols = self.getRowsCols(combineSharedLayers=True, saveFigureLocation=saveFigureLocation)
        numParams = len(paramNames)
        x = np.arange(numEpochs)

        # Get the parameters.
        prefix = "numSharedEncoder" if modelConstants.userInputParams['submodel'] == modelConstants.signalEncoderModel else ("numActivityModel" if "activity" in saveFigureLocation.lower() else "numEmotionModel")
        numSpecificLayers = modelConstants.userInputParams['numSpecificEncoderLayers'] if modelConstants.userInputParams['submodel'] == modelConstants.signalEncoderModel else 1
        numSharedLayers = modelConstants.userInputParams[f'{prefix}Layers']

        for paramInd in range(numParams):
            # Create a figure and axes array
            fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex=True, sharey='col')
            numLow, numSpecificHigh, numSharedHigh = 0, 0, 0
            paramName = paramNames[paramInd]

            fig.suptitle(f"{plotTitle}: {paramName.lower()}", fontsize=24)
            for layerInd in range(numModuleLayers):
                moduleName = moduleNames[0][layerInd].lower()

                if 'low' in moduleName or 'real' in moduleName: rowInd, colInd = numLow, nCols - 1; numLow += 1
                elif 'high' in moduleName or 'imaginary' in moduleName:
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
                    if 'shared' in moduleName and modelInd != 0: continue
                    if 'specific' in moduleName: lineColor = self.darkColors[modelInd]
                    elif 'shared' in moduleName: lineColor = self.blackColor
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
