import os

import numpy as np
import torch
from matplotlib import pyplot as plt

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
        plt.ioff()  # Turn off interactive mode

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
                sharedModels = [modelPipeline.model.sharedSignalEncoderModel for modelPipeline in allModelPipelines]
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

                # # Plot the shared and specific jacobian convergences.
                # self.generalViz.plotSinglaParameterFlow(trainingValues=[specificModel.specificJacobianFlow for specificModel in specificModels],
                #                                         testingValues=[specificModel.sharedJacobianFlow for specificModel in specificModels],
                #                                         labels=[f"{datasetName}" for datasetName in datasetNames],
                #                                         saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Health Jacobian Convergences")

                # Plot the shared and specific jacobian convergences.
                activationParamsPaths = np.asarray([specificModel.activationParamsPath for specificModel in specificModels])
                self.generalViz.plotSinglaParameterFlow(trainingValues=activationParamsPaths[:, :, :, 0, 0], testingValues=activationParamsPaths[:, :, :, 0, 0], labels=[f"{datasetName}" for datasetName in datasetNames],
                                                        saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Health Jacobian Convergences")

    def plotAllTrainingEvents(self, submodel, modelPipeline, lossDataLoader, trainingDate, currentEpoch):
        self.accelerator.print(f"\nPlotting results for the {modelPipeline.model.datasetName} model")

        # Prepare the model/data for evaluation.
        self.setModelSavingFolder(baseSavingFolder=f"trainingFigures/{submodel}/{trainingDate}/", stringID=f"{modelPipeline.model.datasetName}/")
        modelPipeline.setupTrainingFlags(modelPipeline.model, trainingFlag=False)  # Set all models into evaluation mode.
        model = modelPipeline.model
        numPlottingPoints = 6

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
            reconstructedHealthProfile, compiledSignalEncoderLayerStates = model.reconstructPhysiologicalProfile(resampledSignalData)  # reconstructedHealthProfile: batchSize, encodedDimension

            # Extract the model's internal variables.
            signalEncoderLayerTransforms = np.asarray(model.specificSignalEncoderModel.profileModel.signalEncoderLayerTransforms)  # numProfileShots, numProcessingLayers, numExperiments, numSignals=1***, encodedDimension
            retrainingHealthProfilePath = np.asarray(model.specificSignalEncoderModel.profileModel.retrainingHealthProfilePath)  # numProfileShots, numExperiments, numEncodedWeights
            resampledBiomarkerTimes = model.sharedSignalEncoderModel.hyperSampledTimes.detach().cpu().numpy()  # numTimePoints
            compiledSignalEncoderLayerStates = np.flip(compiledSignalEncoderLayerStates, axis=0)
            # compiledSignalEncoderLayerStates: numProcessingLayers, numLayers=1, numSignals, encodedDimension

            # Detach the data from the GPU and tensor format.
            reconstructedHealthProfile, activityProfile, basicEmotionProfile, emotionProfile = reconstructedHealthProfile.detach().cpu().numpy(), activityProfile.detach().cpu().numpy(), basicEmotionProfile.detach().cpu().numpy(), emotionProfile.detach().cpu().numpy()
            validDataMask, reconstructedSignalData, resampledSignalData, healthProfile = validDataMask.detach().cpu().numpy(), reconstructedSignalData.detach().cpu().numpy(), resampledSignalData.detach().cpu().numpy(), healthProfile.detach().cpu().numpy()
            signalData = signalData.detach().cpu().numpy()
            
            # Compile additional information for the model.getActivationParamsFullPassPath
            activationCurvePath, activationModuleNames = model.getActivationCurvesFullPassPath()  # numProcessingLayers, 2, numPoints
            eigenvaluesPath, eigenvaluesModuleNames = model.getEigenvalueFullPassPath()  # numProcessingLayers, numSignals, encodedDimension
            globalPlottingProtocols.clearFigure(fig=None, legend=None, showPlot=False)
            batchInd, signalInd = 1, 0

            # Plot the loss on the primary process.
            if self.accelerator.is_local_main_process:

                # ------------------- Signal Encoding Plots -------------------- #

                if submodel == modelConstants.signalEncoderModel:
                    # Plot the signal reconstruction training information.
                    if signalEncoderLayerTransforms.shape[0] != 0: self.signalEncoderViz.plotProfilePath(relativeTimes=resampledBiomarkerTimes, healthProfile=healthProfile, retrainingProfilePath=signalEncoderLayerTransforms[:, -1, :, signalInd, :], epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Generating Initial Signal")

                    # Plotting model flows.
                    if signalEncoderLayerTransforms.shape[0] != 0: self.signalEncoderViz.plotProfilePath(relativeTimes=resampledBiomarkerTimes, healthProfile=healthProfile, retrainingProfilePath=signalEncoderLayerTransforms[-1, :, :, signalInd, :], epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Data Flow Curves within Model")
                    self.signalEncoderViz.plotSignalEncodingStatePath(relativeTimes=resampledBiomarkerTimes, compiledSignalEncoderLayerStates=compiledSignalEncoderLayerStates, vMin=1.5, epoch=currentEpoch, hiddenLayers=1, saveFigureLocation="signalEncoding/", plotTitle="2D Model Flow by Layer")
                    self.signalEncoderViz.modelFlow(dataTimes=resampledBiomarkerTimes, dataStates=compiledSignalEncoderLayerStates[:, 0], epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="3D Data Flow by Layer", batchInd=0, signalInd=0)

                    # Plot the health profile training information.
                    self.signalEncoderViz.plotProfilePath(relativeTimes=resampledBiomarkerTimes, healthProfile=healthProfile, retrainingProfilePath=retrainingHealthProfilePath, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Health Profile Generation")
                    self.signalEncoderViz.plotProfileReconstructionError(resampledBiomarkerTimes, healthProfile, reconstructedHealthProfile, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Health Profile Reconstruction Error")
                    self.signalEncoderViz.plotProfileReconstruction(resampledBiomarkerTimes, healthProfile, reconstructedHealthProfile, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Health Profile Reconstruction")

                    # # Plot the eigenvalue information.
                    # self.signalEncoderViz.plotEigenvalueAngles(specificSpatialRotationsPath[:, allTrainingSignalMask[batchInd], :], testingEigenValues=specificSpatialRotationsPath[:, allTestingSignalMask[batchInd], :], epoch=currentEpoch, degreesFlag=False, signalInd=0, saveFigureLocation="signalEncoding/", plotTitle="Specific Spatial Eigenvalues Angles")
                    # self.signalEncoderViz.plotEigenvalueAngles(specificNeuralRotationsPath[:, allTrainingSignalMask[batchInd], :], testingEigenValues=specificNeuralRotationsPath[:, allTestingSignalMask[batchInd], :], epoch=currentEpoch, degreesFlag=False, signalInd=0, saveFigureLocation="signalEncoding/", plotTitle="Specific Neural Eigenvalues Angles")
                    # self.signalEncoderViz.plotEigenvalueAngles(sharedSpatialRotationsPath, testingEigenValues=sharedSpatialRotationsPath, epoch=currentEpoch, degreesFlag=False, signalInd=0, saveFigureLocation="signalEncoding/", plotTitle="Shared Spatial Eigenvalues Angles")
                    # self.signalEncoderViz.plotEigenvalueAngles(sharedNeuralRotationsPath, testingEigenValues=sharedNeuralRotationsPath, epoch=currentEpoch, degreesFlag=False, signalInd=0, saveFigureLocation="signalEncoding/", plotTitle="Shared Neural Eigenvalues Angles")
                    #
                    # # Plot the eigenvalue information.
                    # self.signalEncoderViz.plotEigenValueLocations(specificSpatialRotationsPath[:, allTrainingSignalMask[batchInd], :], testingEigenValues=specificSpatialRotationsPath[:, allTestingSignalMask[batchInd], :], epoch=currentEpoch, signalInd=0, saveFigureLocation="signalEncoding/", plotTitle="Specific Spatial Eigenvalues on Circle")
                    # self.signalEncoderViz.plotEigenValueLocations(specificNeuralRotationsPath[:, allTrainingSignalMask[batchInd], :], testingEigenValues=specificNeuralRotationsPath[:, allTestingSignalMask[batchInd], :], epoch=currentEpoch, signalInd=0, saveFigureLocation="signalEncoding/", plotTitle="Specific Neural Eigenvalues on Circle")
                    # self.signalEncoderViz.plotEigenValueLocations(sharedSpatialRotationsPath, testingEigenValues=sharedSpatialRotationsPath, epoch=currentEpoch, signalInd=0, saveFigureLocation="signalEncoding/", plotTitle="Shared Spatial Eigenvalues on Circle")
                    # self.signalEncoderViz.plotEigenValueLocations(sharedNeuralRotationsPath, testingEigenValues=sharedNeuralRotationsPath, epoch=currentEpoch, signalInd=0, saveFigureLocation="signalEncoding/", plotTitle="Shared Neural Eigenvalues on Circle")
                    #
                    # # Plot the eigenvalue information.
                    # self.signalEncoderViz.plotSignalEncodingStatePath(relativeTimes=None, compiledSignalEncoderLayerStates=specificSpatialRotationsPath[:, None, :, :], vMin=180, epoch=currentEpoch, hiddenLayers=0, saveFigureLocation="signalEncoding/", plotTitle="2D Specific Spatial Angles by Layer")
                    # self.signalEncoderViz.plotSignalEncodingStatePath(relativeTimes=None, compiledSignalEncoderLayerStates=specificNeuralRotationsPath[:, None, :, :], vMin=180, epoch=currentEpoch, hiddenLayers=0, saveFigureLocation="signalEncoding/", plotTitle="2D Specific Neural Angles by Layer")
                    # self.signalEncoderViz.plotSignalEncodingStatePath(relativeTimes=None, compiledSignalEncoderLayerStates=sharedSpatialRotationsPath[:, None, :, :], vMin=180, epoch=currentEpoch, hiddenLayers=0, saveFigureLocation="signalEncoding/", plotTitle="2D Shared Spatial Angles by Layer")
                    # self.signalEncoderViz.plotSignalEncodingStatePath(relativeTimes=None, compiledSignalEncoderLayerStates=sharedNeuralRotationsPath[:, None, :, :], vMin=180, epoch=currentEpoch, hiddenLayers=0, saveFigureLocation="signalEncoding/", plotTitle="2D Shared Neural Angles by Layer")
                    #
                    # # Plot the eigenvalue information.
                    # self.signalEncoderViz.modelPropagation3D(neuralEigenvalues=specificSpatialRotationsPath, epoch=currentEpoch, degreesFlag=False, saveFigureLocation="signalEncoding/", plotTitle="3D Spatial Specific Eigenvalues by Layer", batchInd=0, signalInd=0)
                    # self.signalEncoderViz.modelPropagation3D(neuralEigenvalues=sharedSpatialRotationsPath, epoch=currentEpoch, degreesFlag=False, saveFigureLocation="signalEncoding/", plotTitle="3D Spatial Shared Eigenvalues by Layer", batchInd=0, signalInd=0)
                    # self.signalEncoderViz.modelPropagation3D(neuralEigenvalues=specificNeuralRotationsPath, epoch=currentEpoch, degreesFlag=False, saveFigureLocation="signalEncoding/", plotTitle="3D Specific Neural Eigenvalues by Layer", batchInd=0, signalInd=0)
                    # self.signalEncoderViz.modelPropagation3D(neuralEigenvalues=sharedNeuralRotationsPath, epoch=currentEpoch, degreesFlag=False, saveFigureLocation="signalEncoding/", plotTitle="3D Shared Neural Eigenvalues by Layer", batchInd=0, signalInd=0)

                    # Plot the activation information.
                    self.signalEncoderViz.plotActivationParams(activationCurvePath, activationModuleNames, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Specific Spatial Activation Parameters")

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
