import numpy as np
import torch
import os

from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from ._generalVisualizations import generalVisualizations
from ._signalEncoderVisualizations import signalEncoderVisualizations
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

        with torch.no_grad():
            if self.accelerator.is_local_main_process:
                specificModels = [modelPipeline.model.specificSignalEncoderModel for modelPipeline in allModelPipelines]
                # sharedModels = [modelPipeline.model.sharedSignalEncoderModel for modelPipeline in allModelPipelines]
                datasetNames = [modelPipeline.model.datasetName for modelPipeline in allModelPipelines]
                if allModelPipelines[0].getTrainingEpoch(submodel) == 0: return None

                # Plot reconstruction loss for the signal encoder.
                self.generalViz.plotTrainingLosses(trainingLosses=[specificModel.trainingLosses_signalReconstruction for specificModel in specificModels],
                                                   testingLosses=[specificModel.testingLosses_signalReconstruction for specificModel in specificModels],
                                                   lossLabels=[f"{datasetName}" for datasetName in datasetNames],
                                                   saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Convergence Losses")

                # Plot the losses during few-shot retraining the profile.
                self.generalViz.plotTrainingLosses(trainingLosses=[specificModel.profileModel.retrainingProfileLosses for specificModel in specificModels], testingLosses=None,
                                                   lossLabels=[f"{datasetName}" for datasetName in datasetNames],
                                                   saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Profile Losses")

                # Plot the shared and specific jacobian convergences.
                self.generalViz.plotSinglaParameterFlow(trainingValues=[specificModel.specificJacobianFlow for specificModel in specificModels],
                                                        testingValues=[specificModel.sharedJacobianFlow for specificModel in specificModels],
                                                        labels=[f"{datasetName}" for datasetName in datasetNames],
                                                        saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Health Jacobian Convergences")

    def plotAllTrainingEvents(self, submodel, modelPipeline, lossDataLoader, trainingDate, currentEpoch):
        self.accelerator.print(f"\nPlotting results for the {modelPipeline.model.datasetName} model")

        # Prepare the model/data for evaluation.
        self.setModelSavingFolder(baseSavingFolder=f"trainingFigures/{submodel}/{trainingDate}/", stringID=f"{modelPipeline.model.datasetName}/")
        modelPipeline.setupTrainingFlags(modelPipeline.model, trainingFlag=False)  # Set all models into evaluation mode.
        model = modelPipeline.model
        numPlottingPoints = 8

        # Load in all the data and labels for final predictions and calculate the activity and emotion class weights.
        allLabels, allSignalData, allSignalIdentifiers, allMetadata, allTrainingLabelMask, allTrainingSignalMask, allTestingLabelMask, allTestingSignalMask = modelPipeline.prepareInformation(lossDataLoader)
        signalData, signalIdentifiers, metadata = allSignalData[:numPlottingPoints], allSignalIdentifiers[:numPlottingPoints], allMetadata[:numPlottingPoints]
        # allSignalData: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
        # allTrainingLabelMask, allTestingLabelMask: batchSize, numEmotions + 1 (activity)
        # allTrainingSignalMask, allTestingSignalMask: batchSize, numSignals
        # allSignalIdentifiers: batchSize, numSignals, numSignalIdentifiers
        # allLabels: batchSize, numEmotions + 1 (activity) + numSignals
        # allMetadata: batchSize, numMetadata

        with torch.no_grad():
            # Pass all the data through the model and store the emotions, activity, and intermediate variables.
            validDataMask, reconstructedSignalData, resampledSignalData, _, healthProfile, activityProfile, basicEmotionProfile, emotionProfile = model.forward(submodel, signalData, signalIdentifiers, metadata, device=self.accelerator.device, onlyProfileTraining=False)
            reconstructedPhysiologicalProfile, compiledSignalEncoderLayerStates = model.reconstructPhysiologicalProfile(resampledSignalData)
            compiledSignalEncoderLayerStates = np.flip(compiledSignalEncoderLayerStates, axis=0)

            # Detach the data from the GPU and tensor format.
            reconstructedPhysiologicalProfile, activityProfile, basicEmotionProfile, emotionProfile = reconstructedPhysiologicalProfile.detach().cpu().numpy(), activityProfile.detach().cpu().numpy(), basicEmotionProfile.detach().cpu().numpy(), emotionProfile.detach().cpu().numpy()
            validDataMask, reconstructedSignalData, resampledSignalData, healthProfile = validDataMask.detach().cpu().numpy(), reconstructedSignalData.detach().cpu().numpy(), resampledSignalData.detach().cpu().numpy(), healthProfile.detach().cpu().numpy()
            sharedEigenvalues = np.asarray([modelLayer.getAllEigenvalues(device=self.accelerator.device) for modelLayer in model.sharedSignalEncoderModel.processingLayers])  # numProcessingLayers, numLayers=1, numSignals=1, encodedDimension
            specificEigenvalues = np.asarray([modelLayer.getAllEigenvalues(device=self.accelerator.device) for modelLayer in model.specificSignalEncoderModel.processingLayers])  # numProcessingLayers, numLayers=1, numSignals, encodedDimension
            signalEncoderLayerTransforms = np.asarray(model.specificSignalEncoderModel.profileModel.signalEncoderLayerTransforms)  # numProfileShots, 2*numSpecific + numShared + 1, numExperiments, numSignals, encodedDimension
            embeddedProfile = model.specificSignalEncoderModel.profileModel.embeddedHealthProfiles.detach().cpu().numpy()  # numProfileShots, numExperiments, numEncodedWeights
            retrainingEmbeddedProfilePath = np.asarray(model.specificSignalEncoderModel.profileModel.retrainingEmbeddedProfilePath)  # numProfileShots, numExperiments, numEncodedWeights
            resampledBiomarkerTimes = model.sharedSignalEncoderModel.hyperSampledTimes.detach().cpu().numpy()  # numTimePoints
            globalPlottingProtocols.clearFigure(fig=None, legend=None, showPlot=False)
            batchInd, signalInd = -1, -1

            # Plot the loss on the primary GPU.
            if self.accelerator.is_local_main_process:

                # ------------------- Signal Encoding Plots -------------------- #

                if submodel == modelConstants.signalEncoderModel:
                    # Plot the health profile training information.
                    self.signalEncoderViz.plotPhysiologicalReconstruction(resampledBiomarkerTimes, healthProfile, reconstructedPhysiologicalProfile, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Physiological Reconstruction")
                    self.signalEncoderViz.plotPhysiologicalError(resampledBiomarkerTimes, healthProfile, reconstructedPhysiologicalProfile, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Physiological Reconstruction Error")
                    if embeddedProfile.shape[0] != 0: self.signalEncoderViz.plotProfilePath(relativeTimes=None, healthProfile=embeddedProfile, retrainingProfilePath=retrainingEmbeddedProfilePath, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Embedded Profile Generation")
                    if signalEncoderLayerTransforms.shape[0] != 0: self.signalEncoderViz.plotProfilePath(relativeTimes=resampledBiomarkerTimes, healthProfile=healthProfile, retrainingProfilePath=signalEncoderLayerTransforms[:, 0, :, signalInd, :], epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Health Profile Generation")

                    # Plot the signal encoding training information.
                    self.signalEncoderViz.plotEigenValueLocations(specificEigenvalues[:, 0, allTrainingSignalMask[batchInd], :], testingEigenValues=specificEigenvalues[:, 0, allTestingSignalMask[batchInd], :], epoch=currentEpoch, signalInd=0, saveFigureLocation="signalEncoding/", plotTitle="Specific Eigenvalues")
                    self.signalEncoderViz.plotEigenValueLocations(sharedEigenvalues[:, 0, :, :], testingEigenValues=None, epoch=currentEpoch, signalInd=0, saveFigureLocation="signalEncoding/", plotTitle="Shared Eigenvalues")
                    self.signalEncoderViz.plotEigenvalueAngles(specificEigenvalues[:, 0, allTrainingSignalMask[batchInd], :], testingEigenValues=specificEigenvalues[:, 0, allTestingSignalMask[batchInd], :], epoch=currentEpoch, signalInd=0, saveFigureLocation="signalEncoding/", plotTitle="Specific Eigenvalue Angles")
                    self.signalEncoderViz.plotEigenvalueAngles(sharedEigenvalues[:, 0, :, :], testingEigenValues=None, epoch=currentEpoch, signalInd=0, saveFigureLocation="signalEncoding/", plotTitle="Shared Eigenvalue Angles")

                    self.signalEncoderViz.plotSignalEncodingStatePath(resampledBiomarkerTimes, compiledSignalEncoderLayerStates, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Final Model Flow")
                    if signalEncoderLayerTransforms.shape[0] != 0: self.signalEncoderViz.plotProfilePath(relativeTimes=resampledBiomarkerTimes, healthProfile=healthProfile, retrainingProfilePath=signalEncoderLayerTransforms[:, -1, :, signalInd, :], epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Reconstructing Signal")
                    if signalEncoderLayerTransforms.shape[0] != 0: self.signalEncoderViz.plotProfilePath(relativeTimes=resampledBiomarkerTimes, healthProfile=healthProfile, retrainingProfilePath=signalEncoderLayerTransforms[-1, :, :, signalInd, :], epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Model Flow Curves")

                # Plot the autoencoder results.
                self.signalEncoderViz.plotEncoder(signalData, reconstructedSignalData, resampledBiomarkerTimes, resampledSignalData, epoch=currentEpoch, saveFigureLocation="signalReconstruction/", plotTitle="Signal Reconstruction", numSignalPlots=1)

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
