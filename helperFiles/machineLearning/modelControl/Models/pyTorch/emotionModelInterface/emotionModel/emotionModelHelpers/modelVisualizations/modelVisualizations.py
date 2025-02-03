import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from ._generalVisualizations import generalVisualizations
from ._signalEncoderVisualizations import signalEncoderVisualizations
from ..emotionDataInterface import emotionDataInterface
from ..modelConstants import modelConstants


class modelVisualizations(globalPlottingProtocols):

    def __init__(self, accelerator, datasetName):
        super(modelVisualizations, self).__init__()
        self.accelerator = accelerator
        self.datasetName = datasetName

        # Initialize helper classes.
        self.signalEncoderViz = signalEncoderVisualizations(baseSavingFolder="", stringID="", datasetName=datasetName)
        self.generalViz = generalVisualizations(baseSavingFolder="", stringID="", datasetName="_comparison")
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
        self.generalViz.setSavingFolder(self.baseSavingDataFolder, stringID, datasetName="_comparison")

    # ---------------------------------------------------------------------- #

    def plotDatasetComparison(self, submodel, allModelPipelines, trainingDate, showMinimumPlots):
        self.accelerator.print(f"\nCalculating loss for model comparison")

        # Prepare the model/data for evaluation.
        self.setModelSavingFolder(baseSavingFolder=f"trainingFigures/{submodel}/{trainingDate}/", stringID=f"modelComparison/")  # Label the correct folder to save this analysis.

        with torch.no_grad():
            if self.accelerator.is_local_main_process:
                specificModels = [modelPipeline.model.specificSignalEncoderModel for modelPipeline in allModelPipelines]  # Dim: numModels
                datasetNames = [modelPipeline.model.datasetName for modelPipeline in allModelPipelines]  # Dim: numModels

                # Plot reconstruction loss for the signal encoder.
                self.generalViz.plotTrainingLosses(trainingLosses=[specificModel.trainingLosses_signalReconstruction for specificModel in specificModels],
                                                   testingLosses=[specificModel.testingLosses_signalReconstruction for specificModel in specificModels],
                                                   lossLabels=datasetNames, saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Convergence Losses")

                # Plot the losses during few-shot retraining the profile.
                self.generalViz.plotTrainingLosses(trainingLosses=[np.nanmean(specificModel.profileModel.retrainingProfileLosses, axis=1) for specificModel in specificModels], testingLosses=None,
                                                   lossLabels=datasetNames, saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Profile Convergence Losses")
                if showMinimumPlots: return None

                freeParamInformation = np.asarray([modelPipeline.model.getFreeParamsFullPassPath()[1:] for modelPipeline in allModelPipelines])
                moduleNames, maxFreeParamsPath = freeParamInformation[:, 0], freeParamInformation[:, 1].astype(int)  # numFreeParamsPath: numModuleLayers, numSignals, numParams=1
                numFreeModelParams = [specificModel.numFreeParams for specificModel in specificModels]  # numModels, numEpochs, numModuleLayers, numSignals, numParams=1
                self.generalViz.plotFreeParamFlow(numFreeModelParams, maxFreeParamsPath, fullView=True, paramNames=["Free Params"], moduleNames=moduleNames, saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Free Parameters Path")
                self.generalViz.plotFreeParamFlow(numFreeModelParams, maxFreeParamsPath, fullView=False, paramNames=["Free Params"], moduleNames=moduleNames, saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Free Parameters Path Zoomed")

                # Plot the activation parameters for the signal encoder.
                paramNames = ["Infinite Bound", "Linearity Factor", "Convergent Point"]
                activationParamsPaths = np.asarray([specificModel.activationParamsPath for specificModel in specificModels])  # numModels, numEpochs, numActivations, numActivationParams=3
                self.generalViz.plotActivationFlowCompressed(activationParamsPaths=activationParamsPaths, moduleNames=moduleNames, modelLabels=datasetNames, paramNames=paramNames, saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Activation Parameter Compressed Path")
                self.generalViz.plotActivationFlow(activationParamsPaths=activationParamsPaths, moduleNames=moduleNames, paramNames=paramNames, saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Activation Parameter Path")

                # Plot the angle features for the signal encoder.
                hasSpecificLayers = len(allModelPipelines[0].model.specificSignalEncoderModel.processingLayers) != 0
                if hasSpecificLayers: givensAnglesFeatureNames = allModelPipelines[0].model.specificSignalEncoderModel.processingLayers[0].getFeatureParams(layerInd=0)[0]
                else: givensAnglesFeatureNames = allModelPipelines[0].model.sharedSignalEncoderModel.processingLayers[0].getFeatureParams(layerInd=0)[0]
                givensAnglesFeaturesPaths = [specificModel.givensAnglesFeaturesPath for specificModel in specificModels]  # numModels, numEpochs, numModuleLayers, numFeatures, numValues
                self.generalViz.plotGivensFeaturesPath(givensAnglesFeaturesPaths=givensAnglesFeaturesPaths, paramNames=givensAnglesFeatureNames, moduleNames=moduleNames, saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Angular Features Path")

                # Plot the scaling factors for the signal encoder.
                scalingFactorsPaths = [specificModel.scalingFactorsPath for specificModel in specificModels]  # numModels, numEpochs, numModuleLayers, numSignals, numParams=1
                self.generalViz.plotScaleFactorFlow(scalingFactorsPaths, paramNames=["Scalar"], moduleNames=moduleNames, saveFigureLocation="trainingLosses/", plotTitle="Signal Encoder Scalar Path")

    def plotAllTrainingEvents(self, submodel, modelPipeline, lossDataLoader, trainingDate, currentEpoch, showMinimumPlots):
        self.accelerator.print(f"\nPlotting results for the {modelPipeline.model.datasetName} model")

        # Prepare the model/data for evaluation.
        self.setModelSavingFolder(baseSavingFolder=f"trainingFigures/{submodel}/{trainingDate}/", stringID=f"{modelPipeline.model.datasetName}/")
        modelPipeline.setupTrainingFlags(modelPipeline.model, trainingFlag=False)  # Set all models into evaluation mode.
        model = modelPipeline.model
        numPlottingPoints = 4

        # Load in all the data and labels for final predictions and calculate the activity and emotion class weights.
        allLabels, allSignalData, allSignalIdentifiers, allMetadata, allTrainingLabelMask, allTrainingSignalMask, allTestingLabelMask, allTestingSignalMask = modelPipeline.prepareInformation(lossDataLoader)
        validDataMask = emotionDataInterface.getValidDataMask(allSignalData)  # validDataMask: batchSize, numSignals, maxSequenceLength
        validBatchMask = 10 < torch.any(validDataMask, dim=-1).sum(dim=-1)  # validBatchMask: batchSize
        # allSignalData: batchSize, numSignals, maxSequenceLength, [timeChannel, signalChannel]
        # allTrainingLabelMask, allTestingLabelMask: batchSize, numEmotions + 1 (activity)
        # allTrainingSignalMask, allTestingSignalMask: batchSize, numSignals
        # allSignalIdentifiers: batchSize, numSignals, numSignalIdentifiers
        # allLabels: batchSize, numEmotions + 1 (activity) + numSignals
        # allMetadata: batchSize, numMetadata

        with torch.no_grad():
            # Pass all the data through the model and store the emotions, activity, and intermediate variables.
            signalData, signalIdentifiers, metadata = allSignalData[validBatchMask][:numPlottingPoints], allSignalIdentifiers[validBatchMask][:numPlottingPoints], allMetadata[validBatchMask][:numPlottingPoints]
            validDataMask, reconstructedSignalData, resampledSignalData, a, healthProfile, activityProfile, basicEmotionProfile, emotionProfile = model.forward(submodel, signalData, signalIdentifiers, metadata, device=self.accelerator.device, onlyProfileTraining=False)
            reconstructedHealthProfile, forwardModelPassSignals = model.reconstructHealthProfile(resampledSignalData)  # reconstructedHealthProfile: batchSize, encodedDimension
            # forwardModelPassSignals: numProcessingLayers, batchSize, numSignals, encodedDimension

            # Extract the model's internal variables.
            retrainingHealthProfilePath = np.asarray(model.specificSignalEncoderModel.profileModel.retrainingHealthProfilePath)  # numProfileShots, numExperiments, profileDimension
            generatingBiometricSignals = np.asarray(model.specificSignalEncoderModel.profileModel.generatingBiometricSignals)  # numProfileShots, numProcessingLayers, numExperiments, numSignals=1***, encodedDimension
            resampledBiomarkerTimes = model.sharedSignalEncoderModel.hyperSampledTimes.detach().cpu().numpy()  # numTimePoints
            backwardModelPassSignals = np.flip(forwardModelPassSignals, axis=0)

            # Detach the data from the GPU and tensor format.
            reconstructedHealthProfile, activityProfile, basicEmotionProfile, emotionProfile = reconstructedHealthProfile.detach().cpu().numpy(), activityProfile.detach().cpu().numpy(), basicEmotionProfile.detach().cpu().numpy(), emotionProfile.detach().cpu().numpy()
            validDataMask, reconstructedSignalData, resampledSignalData, healthProfile = validDataMask.detach().cpu().numpy(), reconstructedSignalData.detach().cpu().numpy(), resampledSignalData.detach().cpu().numpy(), healthProfile.detach().cpu().numpy()
            signalData = signalData.detach().cpu().numpy()
            
            # Compile additional information for the model.getActivationParamsFullPassPath
            givensAnglesPath, scalingFactorsPath, _, reversibleModuleNames, givensAnglesFeatureNames = model.getLearnableParams()
            activationCurvePath, activationModuleNames = model.getActivationCurvesFullPassPath()  # numModuleLayers, 2=(x, y), numPoints=100
            _, _, maxFreeParamsPath = model.getFreeParamsFullPassPath()
            # givensAnglesPath: numModuleLayers, numSignals, numParams
            # scalingFactorsPath: numModuleLayers, numSignals, numParam=1
            signalNames = model.featureNames
            batchInd, signalInd = 0, 0

            # Plot the loss on the primary process.
            if self.accelerator.is_local_main_process:

                # ------------------- Signal Encoding Plots -------------------- #
                globalPlottingProtocols.clearFigure(fig=None, legend=None, showPlot=False)

                if submodel == modelConstants.signalEncoderModel:
                    # Plot the angular information.
                    if not showMinimumPlots:
                        # Plot information collected across profile training.
                        self.signalEncoderViz.plotProfilePath(relativeTimes=resampledBiomarkerTimes, healthProfile=healthProfile, retrainingProfilePath=generatingBiometricSignals[:, :, signalInd, :], epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Reconstructing Biometric Feature Signal")

                        # Plotting the data flow within the model.
                        self.signalEncoderViz.plotProfilePath(relativeTimes=resampledBiomarkerTimes, healthProfile=healthProfile, retrainingProfilePath=backwardModelPassSignals[:, :, signalInd, :], epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Backwards Transformations (HP to Feature)")
                        self.signalEncoderViz.plotProfilePath(relativeTimes=resampledBiomarkerTimes, healthProfile=healthProfile, retrainingProfilePath=forwardModelPassSignals[:, :, signalInd, :], epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Forwards Transformations (Feature to HP)")
                        self.signalEncoderViz.plotSignalEncodingStatePath(relativeTimes=resampledBiomarkerTimes, compiledSignalEncoderLayerStates=backwardModelPassSignals, vMin=1.1, batchInd=batchInd, signalInd=signalInd, signalNames=signalNames, epoch=currentEpoch, hiddenLayers=1, saveFigureLocation="signalEncoding/", plotTitle="Signal Transformations by Layer Heatmap")
                        self.signalEncoderViz.modelFlow(dataTimes=resampledBiomarkerTimes, dataStates=backwardModelPassSignals[:, :, signalInd, :], signalNames=signalNames, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Signal Transformations by Layer 3D", batchInd=batchInd, signalInd=signalInd)

                        # Plot the health profile training information.
                        self.signalEncoderViz.plotProfilePath(relativeTimes=resampledBiomarkerTimes, healthProfile=healthProfile, retrainingProfilePath=retrainingHealthProfilePath, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Health Profile Generation")
                        self.signalEncoderViz.plotProfileReconstructionError(resampledBiomarkerTimes, healthProfile, reconstructedHealthProfile, epoch=currentEpoch, batchInd=batchInd, saveFigureLocation="signalEncoding/", plotTitle="Health Profile Reconstruction Error")
                        self.signalEncoderViz.plotProfileReconstruction(resampledBiomarkerTimes, healthProfile, reconstructedHealthProfile, epoch=currentEpoch, batchInd=batchInd, saveFigureLocation="signalEncoding/", plotTitle="Health Profile Reconstruction")

                        # Plot the angle information.
                        self.signalEncoderViz.plotsGivensAnglesLine(givensAnglesPath, reversibleModuleNames, epoch=currentEpoch, signalInd=signalInd, degreesFlag=False, saveFigureLocation="signalEncoding/", plotTitle="Rotation Angles Line")
                        self.signalEncoderViz.plotsGivensAnglesHist(givensAnglesPath, reversibleModuleNames, epoch=currentEpoch, signalInd=signalInd, degreesFlag=False, saveFigureLocation="signalEncoding/", plotTitle="Rotation Angles Hist")
                        self.signalEncoderViz.plotAngleLocations(givensAnglesPath, reversibleModuleNames, signalNames=signalNames, epoch=currentEpoch, signalInd=signalInd, saveFigureLocation="signalEncoding/", plotTitle="Givens transformations")
                        self.signalEncoderViz.plotsGivensAnglesHeatmap(givensAnglesPath, reversibleModuleNames, signalInd=signalInd, epoch=currentEpoch, degreesFlag=False, saveFigureLocation="signalEncoding/", plotTitle="Rotation Weight Matrix (S)")

                        # Plot the scale factor information.
                        self.signalEncoderViz.plotScaleFactorHist(scalingFactorsPath, reversibleModuleNames, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Scale Factors Hist")
                        self.signalEncoderViz.plotScaleFactorLines(scalingFactorsPath, reversibleModuleNames, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Scale Factors Line")

                        # Plot the activation information.
                        self.signalEncoderViz.plotActivationCurvesCompressed(activationCurvePath, activationModuleNames, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Activation Parameters (Compressed)")
                        self.signalEncoderViz.plotActivationCurves(activationCurvePath, activationModuleNames, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Activation Parameters")

                # Plot the autoencoder results.
                self.signalEncoderViz.plotEncoder(signalData, reconstructedSignalData, resampledBiomarkerTimes, resampledSignalData, signalNames=signalNames, epoch=currentEpoch, batchInd=batchInd, saveFigureLocation="signalReconstruction/", plotTitle="Signal Reconstruction")

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

                # get the valid emotion indices (ones with training points).
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
