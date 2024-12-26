# General
import numpy as np
from torchviz import make_dot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


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
        plt.plot(allPredictedTrainingLabels, allTrainingLabels, 'ko',
                 markersize=6, alpha=0.3, label="Training Points")
        plt.plot(allPredictedTestingLabels, allTestingLabels, '*',
                 color='tab:blue', markersize=6, alpha=0.6, label="Testing Points")
        plt.xlabel("Predicted Emotion Rating")
        plt.ylabel("Emotion Rating")
        plt.title(f"{plotTitle}")
        plt.legend(loc="best")
        plt.xlim((-0.1, numClasses-0.9))
        plt.ylim((-0.1, numClasses-0.9))
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
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=np.asarray([15, 5]), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1]})

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

    def plotTrainingLosses(self, trainingLosses, testingLosses, lossLabels, saveFigureLocation="", plotTitle="Model Convergence Loss", logY=True, offset=0):
        # Assert the validity of the input data.
        assert len(trainingLosses) == len(lossLabels), "Number of loss labels must match the number of loss indices."
        lossEpochOffset = int(not modelConstants.useInitialLoss)
        if len(trainingLosses[0]) == 0: return None

        # Plot the losses
        for modelInd in range(len(trainingLosses)):
            plt.plot(np.nanmean(trainingLosses[modelInd], axis=-1), label=f'{lossLabels[modelInd]} (Train)', color=self.darkColors[modelInd], linewidth=2)
            if testingLosses is not None:
                testingLoss = np.nanmean(testingLosses[modelInd], axis=-1)
                plt.plot(testingLoss, color=self.darkColors[modelInd], linewidth=2, alpha=0.75)

        # Plot the losses
        for modelInd in range(len(trainingLosses)):
            plt.plot(np.asarray(trainingLosses[modelInd]), '--', color=self.darkColors[modelInd], linewidth=1, alpha=0.05)
            if testingLosses is not None:
                testingLoss = np.asarray(testingLosses[modelInd])
                testingLoss[np.isnan(testingLoss)] = None
                plt.plot(testingLoss, '--', color=self.darkColors[modelInd], linewidth=1, alpha=0.025)
        plt.hlines(y=0.01, xmin=0, xmax=len(trainingLosses[0]), colors=self.blackColor, linestyles='dashed', linewidth=2)
        plt.hlines(y=0.1, xmin=0, xmax=len(trainingLosses[0]), colors=self.blackColor, linestyles='dashed', linewidth=2)
        plt.hlines(y=0.07, xmin=0, xmax=len(trainingLosses[0]), colors=self.blackColor, linestyles='dashed', linewidth=2, alpha=0.5)
        plt.hlines(y=0.06, xmin=0, xmax=len(trainingLosses[0]), colors=self.blackColor, linestyles='dashed', linewidth=2, alpha=0.25)
        plt.hlines(y=0.02, xmin=0, xmax=len(trainingLosses[0]), colors=self.blackColor, linestyles='dashed', linewidth=2, alpha=0.5)
        plt.hlines(y=0.03, xmin=0, xmax=len(trainingLosses[0]), colors=self.blackColor, linestyles='dashed', linewidth=2, alpha=0.25)
        plt.xlim((lossEpochOffset, len(trainingLosses[0]) + 1 + lossEpochOffset))
        plt.ylim((0.005, 2))
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
