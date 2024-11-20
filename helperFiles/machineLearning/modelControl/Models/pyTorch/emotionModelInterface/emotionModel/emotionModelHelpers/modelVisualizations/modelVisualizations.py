# General
import os

import torch

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from ._autoencoderVisualizations import autoencoderVisualizations
from ._generalVisualizations import generalVisualizations
from ._signalEncoderVisualizations import signalEncoderVisualizations
# Import files for machine learning
from ..emotionDataInterface import emotionDataInterface
from ..modelConstants import modelConstants


class modelVisualizations(globalPlottingProtocols):

    def __init__(self, accelerator, modelSubfolder):
        super(modelVisualizations, self).__init__()
        # General parameters.
        self.accelerator = accelerator
        self.saveDataFolder = None

        # Plotting settings.
        # plt.ion()

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
        saveDataFolder = os.path.normpath(os.path.dirname(__file__) + f"/../../../dataAnalysis/{modelSubfolder}") + '/'
        self.saveDataFolder = os.path.relpath(os.path.normpath(saveDataFolder), os.getcwd()) + '/'
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
            specificModels = [modelPipeline.model.specificSignalEncoderModel for modelPipeline in allModelPipelines]
            datasetNames = [modelPipeline.model.datasetName for modelPipeline in allModelPipelines]

            # Plot reconstruction loss.
            self.generalViz.plotTrainingLosses([specificModel.trainingLosses_signalReconstruction for specificModel in specificModels],
                                               [specificModel.testingLosses_signalReconstruction for specificModel in specificModels],
                                               lossLabels=[f"{datasetName} Signal Encoding Reconstruction Loss" for datasetName in datasetNames],
                                               plotTitle="trainingLosses/Signal Encoder Convergence Losses")

    def plotAllTrainingEvents(self, submodel, modelPipeline, lossDataLoader, trainingDate, currentEpoch):
        self.accelerator.print(f"\nPlotting results for the {modelPipeline.model.datasetName} model", flush=True)

        # Prepare the model/data for evaluation.
        self.setSavingFolder(f"trainingFigures/{submodel}/{trainingDate}/{modelPipeline.model.datasetName}/")
        modelPipeline.setupTrainingFlags(modelPipeline.model, trainingFlag=False)  # Set all models into evaluation mode.
        model = modelPipeline.model
        numPlottingPoints = 12

        # Load in all the data and labels for final predictions and calculate the activity and emotion class weights.
        allLabels, allSignalData, allSignalIdentifiers, allMetadata, allTrainingLabelMask, allTrainingSignalMask, allTestingLabelMask, allTestingSignalMask = modelPipeline.prepareInformation(lossDataLoader)
        signalData, signalIdentifiers, metadata = allSignalData[:numPlottingPoints], allSignalIdentifiers[:numPlottingPoints], allMetadata[:numPlottingPoints]

        with torch.no_grad():
            # Pass all the data through the model and store the emotions, activity, and intermediate variables.
            validDataMask, reconstructedSignalData, resampledSignalData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile = model.forward(submodel, signalData, signalIdentifiers, metadata, device=self.accelerator.device, inferenceTraining=False)
            reconstructedPhysiologicalProfile = model.reconstructPhysiologicalProfile(resampledSignalData)

            # Detach the data from the GPU and tensor format.
            validDataMask, reconstructedSignalData, resampledSignalData, physiologicalProfile = validDataMask.detach().cpu().numpy(), reconstructedSignalData.detach().cpu().numpy(), resampledSignalData.detach().cpu().numpy(), physiologicalProfile.detach().cpu().numpy()
            reconstructedPhysiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile = reconstructedPhysiologicalProfile.detach().cpu().numpy(), activityProfile.detach().cpu().numpy(), basicEmotionProfile.detach().cpu().numpy(), emotionProfile.detach().cpu().numpy()
            physiologicalTimes = model.sharedSignalEncoderModel.pseudoEncodedTimes.detach().cpu().numpy()

            # Plot the loss on the primary GPU.
            if self.accelerator.is_local_main_process:

                # ------------------- Signal Encoding Plots -------------------- #

                if submodel == modelConstants.signalEncoderModel:
                    # Plot the encoding example.
                    self.signalEncoderViz.plotPhysiologicalProfile(physiologicalTimes, physiologicalProfile, epoch=currentEpoch, plotTitle="signalEncoding/Physiological Profile")
                    self.signalEncoderViz.plotPhysiologicalReconstruction(physiologicalTimes, physiologicalProfile, reconstructedPhysiologicalProfile, epoch=currentEpoch, plotTitle="signalEncoding/Physiological Reconstruction")

                # Plot the autoencoder results.
                self.signalEncoderViz.plotEncoder(signalData, reconstructedSignalData, physiologicalTimes, resampledSignalData, epoch=currentEpoch, plotTitle="signalReconstruction/Signal Encoding Reconstruction", numSignalPlots=1)

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
