# General
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from torchviz import make_dot

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols


class generalVisualizations(globalPlottingProtocols):

    def __init__(self, baseSavingFolder, stringID, datasetName):
        super(generalVisualizations, self).__init__()
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
        plt.ylabel("Probability (AU)")
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
        plt.tight_layout()
        plt.suptitle(f"{emotionName}")

        # Save the figure is desired.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{emotionName} epochs{epoch}.pdf", baseSaveFigureName=f"{emotionName}.pdf")
        else: self.clearFigure(fig=None, legend=None, showPlot=True)

    def plotTrainingLosses(self, trainingLosses, testingLosses, lossLabels, saveFigureLocation="", plotTitle="Model Convergence Loss", logY=True):
        # Assert the validity of the input data.
        assert len(trainingLosses) == len(lossLabels), "Number of loss labels must match the number of loss indices."
        if len(trainingLosses[0]) == 0: return None
        numModels, numEpochs = len(trainingLosses), len(trainingLosses[0])

        # Plot the average losses.
        for modelInd in range(numModels):
            modelTrainingLosses = np.asarray(trainingLosses[modelInd])
            # modelTrainingLosses: numEpochs, numSignals

            # Calculate the average and standard deviation of the training losses.
            N = np.sum(~np.isnan(modelTrainingLosses), axis=-1)
            trainingStandardError = np.nanstd(modelTrainingLosses, ddof=1, axis=-1) / np.sqrt(N)
            trainingLoss = np.nanmean(modelTrainingLosses, axis=-1)

            # Plot the training losses.
            plt.errorbar(x=np.arange(len(trainingLoss)), y=trainingLoss, yerr=trainingStandardError, color=self.darkColors[modelInd], linewidth=1)
            plt.plot(modelTrainingLosses, '--', color=self.darkColors[modelInd], linewidth=1, alpha=0.05)

            if testingLosses is not None:
                modelTestingLosses = np.asarray(testingLosses[modelInd])
                # Calculate the average and standard deviation of the testing losses.
                N = np.sum(~np.isnan(modelTestingLosses), axis=-1)
                testingStd = np.nanstd(modelTestingLosses, ddof=1, axis=-1) / np.sqrt(N)
                testingLoss = np.nanmean(modelTestingLosses, axis=-1)
                modelTestingLosses[np.isnan(modelTestingLosses)] = None

                # Plot the testing losses.
                plt.errorbar(x=np.arange(len(testingLoss)), y=testingLoss, yerr=testingStd, color=self.darkColors[modelInd], linewidth=1)
                plt.plot(modelTestingLosses, '-', color=self.darkColors[modelInd], linewidth=1, alpha=0.025)

        # Plot gridlines.
        plt.hlines(y=0.1, xmin=0, xmax=len(trainingLosses[0]), colors=self.blackColor, linestyles='dashed', linewidth=1)
        plt.hlines(y=0.03, xmin=0, xmax=len(trainingLosses[0]), colors=self.blackColor, linestyles='dashed', linewidth=1, alpha=0.25)
        plt.hlines(y=0.02, xmin=0, xmax=len(trainingLosses[0]), colors=self.blackColor, linestyles='dashed', linewidth=1, alpha=0.25)
        plt.hlines(y=0.01, xmin=0, xmax=len(trainingLosses[0]), colors=self.blackColor, linestyles='dashed', linewidth=1)
        plt.xlim((0, max(32, len(trainingLosses[0]) + 1)))
        plt.ylim((0.0025, 1))
        plt.grid(True)

        # Label the plot.
        if logY: plt.yscale('log')
        plt.xlabel("Training Epoch")
        plt.ylabel("Loss Values")
        plt.title(f"{plotTitle}")
        plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1), borderaxespad=0)

        # Save the figure if desired.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{len(trainingLosses[0])}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None, showPlot=True)

    def plotSinglaParameterFlow(self, activationParamsPaths, activationModuleNames, modelLabels, activationParamNames, saveFigureLocation="", plotTitle="Model Convergence Loss"):
        numModels, numEpochs, numActivations, numParams = activationParamsPaths.shape

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=1, ncols=numParams, figsize=(6 * numParams, 4), squeeze=False, sharex=True, sharey=False)  # squeeze=False ensures axes is 2D
        axes = axes.flatten()  # Flatten axes for easy indexing if you prefer

        for paramInd in range(numParams):
            ax = axes[paramInd]  # which subplot to use
            paramName = activationParamNames[paramInd]

            for modelInd in range(numModels):
                for activationInd in range(numActivations):
                    activationParams = activationParamsPaths[modelInd, :, activationInd, paramInd]
                    activationModuleName = activationModuleNames[modelInd, activationInd].lower()
                    if "shared" in activationModuleName and modelInd != 0: continue

                    if "specific" in activationModuleName: lineColor = self.darkColors[modelInd]; alpha = 0.8
                    elif "shared" in activationModuleName: lineColor = self.blackColor; alpha = 0.5
                    else: raise ValueError("Activation module name must contain 'specific' or 'shared'.")

                    if modelInd == 0: modelLabel = modelLabels[modelInd]
                    else: modelLabel = None

                    # Plot the activation parameters.
                    ax.plot(activationParams, color=lineColor, linewidth=0.8, alpha=alpha, label=modelLabel)
                    ax.set_xlabel("Training Epoch")
                    ax.set_ylabel("Values")
                    if 'Infinite' in paramName: ax.set_ylim((0, 1.1))
                    elif 'Linearity' in paramName: ax.set_ylim((0, 10.1))
                    elif 'Convergent' in paramName: ax.set_ylim((0, 2.1))
        plt.xlim((0, numEpochs + 1))
        plt.grid(True)

        # Label the plot.
        plt.title(f"{plotTitle}")

        # Save the figure if desired.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{len(activationParams)}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None, showPlot=True)

    def generalDataPlotting(self, plottingData, plottingLabels, saveFigureLocation, plotTitle="Model Convergence Loss"):
        # Plot the training path.
        for plottingDataInd in range(len(plottingData)):
            plt.plot(plottingData[plottingDataInd], label=f'{plottingLabels[plottingDataInd]}', color=self.darkColors[plottingDataInd], linewidth=2)

        # Label the plot.
        plt.legend(loc="upper right")
        plt.xlabel("Training Epoch")
        plt.ylabel("Data Values")
        plt.title(f"{plotTitle}")

        # Save the figure if desired.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{len(plottingData[0])}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None, showPlot=True)
