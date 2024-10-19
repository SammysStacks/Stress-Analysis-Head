# General
import matplotlib.pyplot as plt
import matplotlib
import torch
import os

# Import files for machine learning
from ..emotionDataInterface import emotionDataInterface
from ..generalMethods.dataAugmentation import dataAugmentation
from ..modelConstants import modelConstants

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from ._signalEncoderVisualizations import signalEncoderVisualizations
from ._autoencoderVisualizations import autoencoderVisualizations
from ._generalVisualizations import generalVisualizations


class modelVisualizations(globalPlottingProtocols):

    def __init__(self, accelerator, generalTimeWindow, modelSubfolder):
        super(modelVisualizations, self).__init__()
        # General parameters.
        self.generalTimeWindow = generalTimeWindow
        self.accelerator = accelerator
        self.saveDataFolder = None

        # Plotting settings.
        plt.ion()

        # Initialize helper classes.
        self.dataInterface = emotionDataInterface()

        # Initialize visualization protocols.
        self.generalViz = generalVisualizations("")
        self.autoencoderViz = autoencoderVisualizations("")
        self.signalEncoderViz = signalEncoderVisualizations("")
        # Organize the visualization components.
        self.setSavingFolder(modelSubfolder)

    def setSavingFolder(self, modelSubfolder):
        # Compile and shorten the name of the model visualization folder.
        saveDataFolder = os.path.normpath(os.path.dirname(__file__) + f"/../../../dataAnalysis/{modelSubfolder}") + "/"
        self.saveDataFolder = os.path.relpath(os.path.normpath(saveDataFolder), os.getcwd()) + "/"
        # Make the folder to save the data.
        os.makedirs(self.saveDataFolder, exist_ok=True)

        # Initialize visualization protocols.
        self.generalViz.setSavingFolder(self.saveDataFolder)
        self.autoencoderViz.setSavingFolder(self.saveDataFolder)
        self.signalEncoderViz.setSavingFolder(self.saveDataFolder)

    # ---------------------------------------------------------------------- #

    def plotDatasetComparison(self, submodel, allModelPipelines, trainingDate):
        self.accelerator.print(f"\nCalculating loss for model comparison", flush=True)

        # Prepare the model/data for evaluation.
        self.setSavingFolder(f"trainingFigures/{submodel}/{trainingDate}/modelComparison/")  # Label the correct folder to save this analysis.

        # Plot the loss on the primary GPU.
        if self.accelerator.is_local_main_process:
            sharedModels = [modelPipeline.model.sharedSignalEncoderModel for modelPipeline in allModelPipelines]

            # Plot reconstruction loss.
            self.generalViz.plotTrainingLosses([sharedModel.trainingLosses_signalReconstruction for sharedModel in sharedModels],
                                               [sharedModel.testingLosses_signalReconstruction for sharedModel in sharedModels],
                                               lossLabels=[f"{modelPipeline.model.datasetName} Signal Encoding Reconstruction Loss" for modelPipeline in allModelPipelines],
                                               plotTitle="trainingLosses/Signal Encoder Convergence Losses")

    def plotAllTrainingEvents(self, submodel, modelPipeline, lossDataLoader, trainingDate, currentEpoch):
        self.accelerator.print(f"\nPlotting results for the {modelPipeline.model.datasetName} model", flush=True)

        # Prepare the model/data for evaluation.
        modelPipeline.setupTrainingFlags(modelPipeline.model, trainingFlag=False)  # Set all models into evaluation mode.
        model = modelPipeline.model

        # Load in all the data and labels for final predictions.
        # Load in all the data and labels for final predictions and calculate the activity and emotion class weights.
        allData, allLabels, allTrainingMasks, allTestingMasks, allSignalData, allSignalIdentifiers, allMetadata, reconstructionIndex = modelPipeline.prepareInformation(lossDataLoader)
        assert reconstructionIndex is not None
        with torch.no_grad():

            # ---------------------- Time-Specific Plots ----------------------- #

            # Go through all the plots at this specific time window.
            self.plotTrainingEvent(model, currentEpoch, allSignalData, allSignalIdentifiers, allMetadata, allTrainingMasks, allTestingMasks, reconstructionIndex, submodel, trainingDate, model.datasetName)

            # ---------------------- Time-Agnostic Plots ----------------------- #

            # Prepare the model/data for evaluation.
            self.setSavingFolder(f"trainingFigures/{submodel}/{trainingDate}/{model.datasetName}/")  # Label the correct folder to save this analysis.

            # Wait before continuing.
            self.accelerator.wait_for_everyone()

        # ------------------------------------------------------------------ #

    def plotTrainingEvent(self, model, currentEpoch, allSignalData, allSignalIdentifiers, allMetadata, allTrainingMasks, allTestingMasks, reconstructionIndex, submodel, trainingDate, datasetName):
        # Prepare the model/data for evaluation.
        self.setSavingFolder(f"trainingFigures/{submodel}/{trainingDate}/{datasetName}/")  # Label the correct folder to save this analysis.
        model.eval()

        # General plotting parameters.
        numTrainingInstances, numTestingInstances = 6, 6
        _, numSignals, _, _ = allSignalData.shape

        # Get the reconstruction data mask
        reconstructionDataTrainingMask = self.dataInterface.getEmotionColumn(allTrainingMasks, reconstructionIndex)  # Dim: numExperiments
        reconstructionDataTestingMask = self.dataInterface.getEmotionColumn(allTestingMasks, reconstructionIndex)  # Dim: numExperiments

        # Compile the training data to plot
        trainingSignalIdentifiers = allSignalIdentifiers[reconstructionDataTrainingMask][0: numTrainingInstances]
        trainingSignalData = allSignalData[reconstructionDataTrainingMask][0: numTrainingInstances]
        trainingMetadata = allMetadata[reconstructionDataTrainingMask][0: numTrainingInstances]

        # Compile the training data to plot
        testingSignalIdentifiers = allSignalIdentifiers[reconstructionDataTestingMask][0: numTestingInstances]
        testingSignalData = allSignalData[reconstructionDataTestingMask][0: numTestingInstances]
        testingMetadata = allMetadata[reconstructionDataTestingMask][0: numTestingInstances]

        with torch.no_grad():  # Stop gradient tracking
            # Pass all the data through the model and store the emotions, activity, and intermediate variables.
            missingDataTrainingMask, reconstructedSignalTrainingData, finalTrainingManifoldProjectionLoss, fourierTrainingData, physiologicalTrainingProfile, activityTrainingProfile, basicEmotionTrainingProfile, emotionTrainingProfile = model.forward(submodel, trainingSignalData, trainingSignalIdentifiers, trainingMetadata, device=self.accelerator.device, trainingFlag=True)
            missingDataTestingMask, reconstructedSignalTestingData, finalTestingManifoldProjectionLoss, fourierTestingData, physiologicalTestingProfile, activityTestingProfile, basicEmotionTestingProfile, emotionTestingProfile = model.forward(submodel, testingSignalData, testingSignalIdentifiers, testingMetadata, device=self.accelerator.device, trainingFlag=True)

            # Detach the data from the GPU and tensor format.
            reconstructedSignalTrainingData, finalTrainingManifoldProjectionLoss, fourierTrainingData, physiologicalTrainingProfile, activityTrainingProfile, basicEmotionTrainingProfile, emotionTrainingProfile = reconstructedSignalTrainingData.detach().cpu().numpy(), finalTrainingManifoldProjectionLoss.detach().cpu().numpy(), fourierTrainingData.detach().cpu().numpy(), physiologicalTrainingProfile.detach().cpu().numpy(), activityTrainingProfile.detach().cpu().numpy(), basicEmotionTrainingProfile.detach().cpu().numpy(), emotionTrainingProfile.detach().cpu().numpy()
            reconstructedSignalTestingData, finalTestingManifoldProjectionLoss, fourierTestingData, physiologicalTestingProfile, activityTestingProfile, basicEmotionTestingProfile, emotionTestingProfile = reconstructedSignalTestingData.detach().cpu().numpy(), finalTestingManifoldProjectionLoss.detach().cpu().numpy(), fourierTestingData.detach().cpu().numpy(), physiologicalTestingProfile.detach().cpu().numpy(), activityTestingProfile.detach().cpu().numpy(), basicEmotionTestingProfile.detach().cpu().numpy(), emotionTestingProfile.detach().cpu().numpy()
            physiologicalTimes = model.sharedSignalEncoderModel.pseudoEncodedTimes.detach().cpu().numpy()

        # ------------------- Plot the Data on One Device ------------------ # 

        # Plot the loss on the primary GPU.
        if self.accelerator.is_local_main_process:

            # ------------------- Signal Encoding Plots -------------------- # 

            if submodel == modelConstants.signalEncoderModel:
                # Plot the encoding example.
                self.signalEncoderViz.plotSignalEncodingMap(physiologicalTimes, physiologicalTrainingProfile, trainingSignalData, epoch=currentEpoch, plotTitle="signalEncoding/Testing Physiological Profile", numBatchPlots=1, numSignalPlots=1)
                self.signalEncoderViz.plotSignalEncodingMap(physiologicalTimes, physiologicalTestingProfile, testingSignalData, epoch=currentEpoch, plotTitle="signalEncoding/Training Physiological Profile", numBatchPlots=1, numSignalPlots=1)

            # Plot the autoencoder results.
            self.autoencoderViz.plotEncoder(reconstructedSignalTrainingData, fourierTrainingData, epoch=currentEpoch, plotTitle="signalReconstruction/Signal Encoding Training Reconstruction", numSignalPlots=2)
            self.autoencoderViz.plotEncoder(reconstructedSignalTestingData, fourierTestingData, epoch=currentEpoch, plotTitle="signalReconstruction/Signal Encoding Testing Reconstruction", numSignalPlots=2)

            # Dont keep plotting untrained models.
            if submodel == modelConstants.signalEncoderModel: return None

            # ------------------ Emotion Prediction Plots ------------------ # 

            # Organize activity information.
            # activityTestingMask = self.dataInterface.getActivityColumn(allTestingMasks)
            # activityTrainingMask = self.dataInterface.getActivityColumn(allTrainingMasks)
            # activityTestingLabels = self.dataInterface.getActivityLabels(allLabels, allTestingMasks)
            # activityTrainingLabels = self.dataInterface.getActivityLabels(allLabels, allTrainingMasks)

            # Activity plotting.
            # predictedActivityLabels = allActivityDistributions.argmax(dim=1).int()
            # self.plotPredictedMatrix(activityTrainingLabels, activityTestingLabels, predictedActivityLabels[activityTrainingMask], predictedActivityLabels[activityTestingMask], self.numActivities, epoch=currentEpoch, "Activities")
            # self.plotTrainingLosses(self.trainingLosses_activities, self.testingLosses_activities, plotTitle = "Activity Loss (Cross Entropy)")

            # Get the valid emotion indices (ones with training points).
            # emotionTrainingMask = self.dataInterface.getEmotionMasks(allTrainingMasks)
            # validEmotionInds = self.dataInterface.getLabelInds_withPoints(emotionTrainingMask)
            # For each emotion we are predicting that has training data.
            # for validEmotionInd in validEmotionInds:  
            #     testingMask = allTestingMasks[:, validEmotionInd]
            #     trainingMask = allTrainingMasks[:, validEmotionInd]
            #     emotionName = self.emotionNames[validEmotionInd]

            #     # # Organize the emotion's training/testing information.
            #     trainingEmotionLabels = self.dataInterface.getEmotionLabels(validEmotionInd, allLabels, allTrainingMasks)
            #     testingEmotionLabels = self.dataInterface.getEmotionLabels(validEmotionInd, allLabels, allTestingMasks)

            #     # Get the predicted and true emotion distributions.
            #     predictedTrainingEmotions, trueTrainingEmotions = self.dataInterface.getEmotionDistributions(validEmotionInd, allFinalEmotionDistributions, allLabels, allTrainingMasks)
            #     predictedTestingEmotions, trueTestingEmotions = self.dataInterface.getEmotionDistributions(validEmotionInd, allFinalEmotionDistributions, allLabels, allTestingMasks)

            #     # Get the class information from the testing and training data.
            #     allPredictedEmotionClasses = modelHelpers.extractClassIndex(allFinalEmotionDistributions[validEmotionInd], self.allEmotionClasses[validEmotionInd], axisDimension = 1, returnIndex = True)
            #     predictedTrainingEmotionClasses = allPredictedEmotionClasses[trainingMask]
            #     predictedTestingEmotionClasses = allPredictedEmotionClasses[testingMask]

            #     # Scale the emotion if log-softmax was the final layer.
            #     if model.lastEmotionLayer.lastEmotionLayer == "logSoftmax":
            #         predictedTestingEmotions = np.exp(predictedTestingEmotions)
            #         predictedTrainingEmotions = np.exp(predictedTrainingEmotions)

            # Get all the data predictions.
            # self.plotDistributions(trueTestingEmotions, predictedTestingEmotions, self.allEmotionClasses[validEmotionInd], plotTitle = "Testing Emotion Distributions")
            # self.plotDistributions(trueTrainingEmotions, predictedTrainingEmotions, self.allEmotionClasses[validEmotionInd], plotTitle = "Training Emotion Distributions")
            # # self.plotPredictions(trainingEmotionLabels, testingEmotionLabels, predictedTrainingEmotionClasses,
            # #                       predictedTestingEmotionClasses, self.allEmotionClasses[validEmotionInd], emotionName)
            # self.plotPredictedMatrix(trainingEmotionLabels, testingEmotionLabels, predictedTrainingEmotionClasses, predictedTestingEmotionClasses, self.allEmotionClasses[validEmotionInd], epoch=currentEpoch, emotionName)
            # # Plot model convergence curves.
            # self.plotTrainingLosses(self.trainingLosses_emotions[validEmotionInd], self.testingLosses_emotions[validEmotionInd], plotTitle = "Emotion Convergence Loss (KL)")
