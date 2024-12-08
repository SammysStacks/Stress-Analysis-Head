# General
import os

import numpy as np
import torch

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from ._generalVisualizations import generalVisualizations
from ._signalEncoderVisualizations import signalEncoderVisualizations
# Import files for machine learning
from ..modelConstants import modelConstants


class modelVisualizations(globalPlottingProtocols):

    def __init__(self, accelerator, datasetName):
        super(modelVisualizations, self).__init__()
        self.accelerator = accelerator
        self.datasetName = datasetName

        # Initialize helper classes.
        self.signalEncoderViz = signalEncoderVisualizations(baseSavingFolder="", stringID="", datasetName=datasetName)
        self.generalViz = generalVisualizations(baseSavingFolder="", stringID="", datasetName=datasetName)

    def setModelSavingFolder(self, baseSavingFolder, stringID):
        # Compile and shorten the name of the model visualization folder.
        baseSavingDataFolder = os.path.normpath(os.path.dirname(__file__) + f"/../../../dataAnalysis/{baseSavingFolder}") + '/'
        saveDataFolder = os.path.normpath(baseSavingDataFolder + stringID + '/')

        # Set the saving folder for the model visualizations.
        self.baseSavingDataFolder = os.path.relpath(baseSavingDataFolder, os.getcwd()) + '/'
        self.saveDataFolder = os.path.relpath(saveDataFolder, os.getcwd()) + '/'
        self._createFolder(self.saveDataFolder)

        # Initialize visualization protocols.
        self.signalEncoderViz.setSavingFolder(self.baseSavingDataFolder, stringID, self.datasetName)
        self.generalViz.setSavingFolder(self.baseSavingDataFolder, stringID, self.datasetName)

    # ---------------------------------------------------------------------- #

    def plotDatasetComparison(self, submodel, allModelPipelines, trainingDate):
        self.accelerator.print(f"\nCalculating loss for model comparison")

        # Prepare the model/data for evaluation.
        self.setModelSavingFolder(baseSavingFolder=f"trainingFigures/{submodel}/{trainingDate}/", stringID=f"modelComparison/")  # Label the correct folder to save this analysis.

        # Plot the loss on the primary GPU.
        if self.accelerator.is_local_main_process:
            specificModels = [modelPipeline.model.specificSignalEncoderModel for modelPipeline in allModelPipelines]
            datasetNames = [modelPipeline.model.datasetName for modelPipeline in allModelPipelines]

            # Plot reconstruction loss.
            self.generalViz.plotTrainingLosses([specificModel.trainingLosses_signalReconstruction for specificModel in specificModels],
                                               [specificModel.testingLosses_signalReconstruction for specificModel in specificModels],
                                               lossLabels=[f"{datasetName} Signal Encoding Reconstruction Loss" for datasetName in datasetNames],
                                               saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Convergence Losses")

            # Plot inference loss.
            self.generalViz.plotTrainingLosses([specificModel.trainingLosses_inference for specificModel in specificModels],
                                               [specificModel.testingLosses_inference for specificModel in specificModels],
                                               lossLabels=[f"{datasetName} Signal Encoding Inference Loss" for datasetName in datasetNames],
                                               saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Inference Losses")

            # Plot inference loss.
            self.generalViz.plotTrainingLosses([specificModel.trainingLosses_signalReconstruction for specificModel in specificModels],
                                               [specificModel.trainingLosses_inference for specificModel in specificModels],
                                               lossLabels=[f"{datasetName} Signal Encoding Inference Loss" for datasetName in datasetNames],
                                               saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Training Losses")

    def plotAllTrainingEvents(self, submodel, modelPipeline, lossDataLoader, trainingDate, currentEpoch):
        self.accelerator.print(f"\nPlotting results for the {modelPipeline.model.datasetName} model")

        # Prepare the model/data for evaluation.
        self.setModelSavingFolder(baseSavingFolder=f"trainingFigures/{submodel}/{trainingDate}/", stringID=f"{modelPipeline.model.datasetName}/")
        modelPipeline.setupTrainingFlags(modelPipeline.model, trainingFlag=False)  # Set all models into evaluation mode.
        model = modelPipeline.model
        numPlottingPoints = 12

        # Load in all the data and labels for final predictions and calculate the activity and emotion class weights.
        allLabels, allSignalData, allSignalIdentifiers, allMetadata, allTrainingLabelMask, allTrainingSignalMask, allTestingLabelMask, allTestingSignalMask = modelPipeline.prepareInformation(lossDataLoader)
        signalData, signalIdentifiers, metadata = allSignalData[:numPlottingPoints], allSignalIdentifiers[:numPlottingPoints], allMetadata[:numPlottingPoints]

        with torch.no_grad():
            # Pass all the data through the model and store the emotions, activity, and intermediate variables.
            validDataMaskInference, reconstructedSignalDataInference, resampledSignalDataInference, physiologicalProfileInference, activityProfileInference, basicEmotionProfileInference, emotionProfileInference = model.forward(submodel, signalData, signalIdentifiers, metadata, device=self.accelerator.device, inferenceTraining=True, trainingFlag=False)
            validDataMask, reconstructedSignalData, resampledSignalData, physiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile = model.forward(submodel, signalData, signalIdentifiers, metadata, device=self.accelerator.device, inferenceTraining=False, trainingFlag=False)
            reconstructedPhysiologicalProfile, compiledSignalEncoderLayerStates = model.reconstructPhysiologicalProfile(resampledSignalData)

            # Detach the data from the GPU and tensor format.
            validDataMaskInference, reconstructedSignalDataInference, resampledSignalDataInference, physiologicalProfileInference = validDataMaskInference.detach().cpu().numpy(), reconstructedSignalDataInference.detach().cpu().numpy(), resampledSignalDataInference.detach().cpu().numpy(), physiologicalProfileInference.detach().cpu().numpy()
            reconstructedPhysiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile = reconstructedPhysiologicalProfile.detach().cpu().numpy(), activityProfile.detach().cpu().numpy(), basicEmotionProfile.detach().cpu().numpy(), emotionProfile.detach().cpu().numpy()
            validDataMask, reconstructedSignalData, resampledSignalData, physiologicalProfile = validDataMask.detach().cpu().numpy(), reconstructedSignalData.detach().cpu().numpy(), resampledSignalData.detach().cpu().numpy(), physiologicalProfile.detach().cpu().numpy()
            activityProfileInference, basicEmotionProfileInference, emotionProfileInference = activityProfileInference.detach().cpu().numpy(), basicEmotionProfileInference.detach().cpu().numpy(), emotionProfileInference.detach().cpu().numpy()
            physiologicalTimes = model.sharedSignalEncoderModel.pseudoEncodedTimes.detach().cpu().numpy()  # pseudoEncodedTimes: numTimePoints
            compiledSignalEncoderLayerStates = np.asarray(compiledSignalEncoderLayerStates)  # numLayers, numExperiments, numSignals, encodedDimension
            inferenceStatePath = np.asarray(model.inferenceModel.inferenceStatePath)  # numInferenceSteps, numExperiments, encodedDimension

            # Plot the loss on the primary GPU.
            if self.accelerator.is_local_main_process:

                # ------------------- Signal Encoding Plots -------------------- #

                if submodel == modelConstants.signalEncoderModel:
                    # Plot the physiological profile training information.
                    self.signalEncoderViz.plotPhysiologicalProfile(physiologicalTimes, physiologicalProfile, physiologicalProfileInference, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Physiological Profile")
                    self.signalEncoderViz.plotPhysiologicalReconstruction(physiologicalTimes, physiologicalProfile, reconstructedPhysiologicalProfile, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Physiological Reconstruction")
                    self.signalEncoderViz.plotPhysiologicalError(physiologicalTimes, physiologicalProfile, reconstructedPhysiologicalProfile, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Physiological Reconstruction Error")
                    if inferenceStatePath.shape[0] != 0: self.signalEncoderViz.plotInferencePath(physiologicalTimes, physiologicalProfile, inferenceStatePath, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Physiological Inference State Path")

                    # Plot the signal encoding training information.
                    self.signalEncoderViz.plotSignalEncodingStatePath(physiologicalTimes, compiledSignalEncoderLayerStates, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Signal Encoding State Path")

                # Plot the autoencoder results.
                self.signalEncoderViz.plotEncoder(signalData, reconstructedSignalDataInference, physiologicalTimes, resampledSignalDataInference, epoch=currentEpoch, saveFigureLocation="signalReconstruction/", plotTitle="Signal Encoding Inference Reconstruction", numSignalPlots=1)
                self.signalEncoderViz.plotEncoder(signalData, reconstructedSignalData, physiologicalTimes, resampledSignalData, epoch=currentEpoch, saveFigureLocation="signalReconstruction/", plotTitle="Signal Encoding Reconstruction", numSignalPlots=1)

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
