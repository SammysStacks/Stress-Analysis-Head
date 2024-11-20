# General
import numpy as np
from torchviz import make_dot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols


class generalVisualizations(globalPlottingProtocols):

    def __init__(self, saveDataFolder):
        super(generalVisualizations, self).__init__()
        # General parameter
        self.saveDataFolder = None

        # Set the location for saving the models.
        self.setSavingFolder(saveDataFolder)
                
    def setSavingFolder(self, saveDataFolder):
        self.saveDataFolder = saveDataFolder
        
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
        plt.title(f"{plotTitle.split('/')[-1]}")
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
        plt.title(f"{plotTitle.split('/')[-1]}")
        plt.legend(loc="best")
        plt.show()

    def plotPredictedMatrix(self, allTrainingLabels, allTestingLabels, allPredictedTrainingLabels, allPredictedTestingLabels, numClasses, epoch, emotionName):
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
        fig, axes = plt.subplots(1, 2, figsize=np.asarray([15, 5]), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1]})

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
        if self.saveDataFolder:
            self.displayFigure(self.saveDataFolder + f"{emotionName} epochs = {epoch}.pdf")
        plt.show()

    def plotTrainingPath_timeAnalysis(self, pathParameters, timeLabels, plotTitle="Model Convergence Loss"):
        # Plot the training path.
        for timeWindowInd in range(len(pathParameters)):
            plt.plot(pathParameters[timeWindowInd], label=f'{timeLabels[timeWindowInd]}', color=self.darkColors[timeWindowInd], linewidth=2)

        # Label the plot.
        plt.legend(loc="upper right")
        plt.xlabel("Training Epoch")
        plt.ylabel("Path Values")
        plt.title(f"{plotTitle.split('/')[-1]}")

        # Save the figure if desired.
        if self.saveDataFolder: self.displayFigure(self.saveDataFolder + f"{plotTitle} at epoch {len(pathParameters[0])}.pdf")
        else: plt.show(); plt.close('all')

    def plotTrainingLosses(self, trainingLosses, testingLosses, lossLabels, plotTitle="Model Convergence Loss", logY=True):
        # Assert the validity of the input data.
        assert len(trainingLosses) == len(lossLabels), "Number of loss labels must match the number of loss indices."

        # Base case: there is no data to plot.
        if len(trainingLosses[0]) == 0: return None

        # Plot the losses
        for modelInd in range(len(trainingLosses)):
            plt.plot(np.nanmean(trainingLosses[modelInd], axis=-1), label=f'{lossLabels[modelInd]} (Train)', color=self.darkColors[modelInd], linewidth=2)
            if testingLosses is not None:
                testingLoss = np.nanmean(testingLosses[modelInd], axis=-1)
                testingLoss = np.where(np.isnan(testingLoss), 0, testingLoss)
                plt.plot(testingLoss, '--', color=self.darkColors[modelInd], linewidth=2, alpha=0.75)

        # Plot the losses
        for modelInd in range(len(trainingLosses)):
            plt.plot(np.asarray(trainingLosses[modelInd]), color=self.darkColors[modelInd], linewidth=1, alpha=0.01)
            if testingLosses is not None:
                testingLoss = np.asarray(testingLosses[modelInd])
                if np.isnan(testingLoss).all(): continue
                testingLoss = testingLoss[np.isnan(testingLoss)]
                plt.plot(testingLoss, '--', color=self.darkColors[modelInd], linewidth=1, alpha=0.01)

        # Label the plot.
        if logY: plt.yscale('log')
        plt.xlabel("Training Epoch")
        plt.ylabel("Loss Values")
        plt.title(f"{plotTitle.split('/')[-1]}")
        plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1))  # Move legend off to the right, level with the top

        # Save the figure if desired.
        if self.saveDataFolder: self.displayFigure(self.saveDataFolder + f"{plotTitle} at epoch {len(trainingLosses[0])}.pdf")
        else: plt.show()

    def generalDataPlotting(self, plottingData, plottingLabels, plotTitle="Model Convergence Loss"):
        # Plot the training path.
        for plottingDataInd in range(len(plottingData)):
            plt.plot(plottingData[plottingDataInd], label=f'{plottingLabels[plottingDataInd]}', color=self.darkColors[plottingDataInd], linewidth=2)

        # Label the plot.
        plt.legend(loc="upper right")
        plt.xlabel("Training Epoch")
        plt.ylabel("Data Values")
        plt.title(f"{plotTitle.split('/')[-1]}")

        # Save the figure if desired.
        if self.saveDataFolder: self.displayFigure(self.saveDataFolder + f"{plotTitle} at epoch {len(plottingData[0])}.pdf")
        else: plt.show(); plt.close('all')
