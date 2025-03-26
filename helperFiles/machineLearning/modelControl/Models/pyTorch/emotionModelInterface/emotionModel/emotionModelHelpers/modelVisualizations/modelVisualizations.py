import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from ._generalVisualizations import generalVisualizations
from ._signalEncoderVisualizations import signalEncoderVisualizations
from ..emotionDataInterface import emotionDataInterface
from ..modelConstants import modelConstants
from ..submodels.modelComponents.reversibleComponents.reversibleLieLayer import reversibleLieLayer


class modelVisualizations(globalPlottingProtocols):

    def __init__(self, accelerator, datasetName):
        super(modelVisualizations, self).__init__(interactivePlots=False)
        self.accelerator = accelerator
        self.datasetName = datasetName
        plt.ioff()  # Turn off interactive mode

        # Initialize helper classes.
        self.signalEncoderViz = signalEncoderVisualizations(baseSavingFolder="", stringID="", datasetName=datasetName)
        self.generalViz = generalVisualizations(baseSavingFolder="", stringID="", datasetName="_comparison")

    def setModelSavingFolder(self, baseSavingFolder, stringID, epoch=None):
        # Compile and shorten the name of the model visualization folder.
        baseSavingDataFolder = os.path.normpath(os.path.dirname(__file__) + f"/../../../dataAnalysis/{baseSavingFolder}") + '/'
        saveDataFolder = os.path.normpath(baseSavingDataFolder + stringID + '/')

        # Set the saving folder for the model visualizations.
        self.baseSavingDataFolder = os.path.relpath(baseSavingDataFolder, os.getcwd()) + '/'
        self.saveDataFolder = os.path.relpath(saveDataFolder, os.getcwd()) + '/'
        if self.hpcFlag and epoch == 0 and os.path.exists(self.saveDataFolder): raise Exception(f"Folder already exists: {self.saveDataFolder}")
        self._createFolder(self.saveDataFolder)

        # Initialize visualization protocols.
        self.signalEncoderViz.setSavingFolder(self.baseSavingDataFolder, stringID, self.datasetName)
        self.generalViz.setSavingFolder(self.baseSavingDataFolder, stringID, datasetName="_comparison")

    # ---------------------------------------------------------------------- #

    def plotDatasetComparison(self, submodel, allModelPipelines, trainingDate, showMinimumPlots):
        self.accelerator.print(f"\nCalculating loss for model comparison")

        # Prepare the model/data for evaluation.
        self.setModelSavingFolder(baseSavingFolder=f"trainingFigures/{submodel}/{trainingDate}/", stringID=f"modelComparison/", epoch=-1)  # Label the correct folder to save this analysis.

        with torch.no_grad():
            if self.accelerator.is_local_main_process:
                specificModels = [modelPipeline.model.specificSignalEncoderModel for modelPipeline in allModelPipelines]  # Dim: numModels
                datasetNames = [modelPipeline.model.datasetName for modelPipeline in allModelPipelines]  # Dim: numModels

                # Plot reconstruction loss for the signal encoder.
                self.generalViz.plotTrainingLosses(trainingLosses=[specificModel.trainingLosses_signalReconstruction for specificModel in specificModels],
                                                   testingLosses=[specificModel.testingLosses_signalReconstruction for specificModel in specificModels],
                                                   lossLabels=datasetNames, saveFigureLocation="trainingLosses/", plotTitle="Signal encoder convergence losses")

                # Plot the losses during few-shot retraining the profile.
                self.generalViz.plotTrainingLosses(trainingLosses=[np.nanmean(specificModel.profileModel.retrainingProfileLosses, axis=1) for specificModel in specificModels], testingLosses=None,
                                                   lossLabels=datasetNames, saveFigureLocation="trainingLosses/", plotTitle="Signal encoder profile convergence losses")

                freeParamInformation = np.asarray([modelPipeline.model.getFreeParamsFullPassPath()[1:] for modelPipeline in allModelPipelines])
                moduleNames, maxFreeParamsPath = freeParamInformation[:, 0], freeParamInformation[:, 1].astype(int)  # numFreeParamsPath: numModuleLayers, numSignals, numParams=1
                numFreeModelParams = [specificModel.numFreeParams for specificModel in specificModels]  # numModels, numEpochs, numModuleLayers, numSignals, numParams=1
                self.generalViz.plotFreeParamFlow(numFreeModelParams, maxFreeParamsPath, fullView=False, paramNames=["Free params"], moduleNames=moduleNames, saveFigureLocation="trainingLosses/", plotTitle="Signal encoder free parameters path zoomed")
                self.generalViz.plotFreeParamFlow(numFreeModelParams, maxFreeParamsPath, fullView=True, paramNames=["Free params"], moduleNames=moduleNames, saveFigureLocation="trainingLosses/", plotTitle="Signal encoder free parameters path")
                for modelInd in range(len(numFreeModelParams)): print('numFreeModelParams:', numFreeModelParams[modelInd][-1][0].mean(), numFreeModelParams[modelInd][-1][1].mean(),  numFreeModelParams[modelInd][-1][2].mean(), numFreeModelParams[modelInd][-1][3].mean(), numFreeModelParams[modelInd][-1][4].mean())
                if showMinimumPlots: return None

                # Plot the activation parameters for the signal encoder.
                paramNames = ["Infinite Bound", "Linearity Factor", "Convergent Point"]
                activationParamsPaths = np.asarray([specificModel.activationParamsPath for specificModel in specificModels])  # numModels, numEpochs, numActivations, numActivationParams=3
                self.generalViz.plotActivationFlowCompressed(activationParamsPaths=activationParamsPaths, moduleNames=moduleNames, modelLabels=datasetNames, paramNames=paramNames, saveFigureLocation="trainingLosses/", plotTitle="Signal encoder activation parameter compressed path")
                self.generalViz.plotActivationFlow(activationParamsPaths=activationParamsPaths, moduleNames=moduleNames, paramNames=paramNames, saveFigureLocation="trainingLosses/", plotTitle="Signal encoder activation parameter path")

                # Plot the angle features for the signal encoder.
                givensAnglesFeatureNames = reversibleLieLayer.getFeatureNames()
                givensAnglesFeaturesPaths = [specificModel.givensAnglesFeaturesPath for specificModel in specificModels]  # numModels, numEpochs, numModuleLayers, numFeatures, numValues
                self.generalViz.plotGivensFeaturesPath(givensAnglesFeaturesPaths=givensAnglesFeaturesPaths, paramNames=givensAnglesFeatureNames, moduleNames=moduleNames, saveFigureLocation="trainingLosses/", plotTitle="Signal encoder angular features path")

                # Plot the scaling factors for the signal encoder.
                scalingFactorsPaths = [specificModel.scalingFactorsPath for specificModel in specificModels]  # numModels, numEpochs, numModuleLayers, numSignals, numParams=1
                self.generalViz.plotScaleFactorFlow(scalingFactorsPaths, paramNames=["Scalar"], moduleNames=moduleNames, saveFigureLocation="trainingLosses/", plotTitle="Signal encoder scalar path")

    def plotAllTrainingEvents(self, submodel, modelPipeline, lossDataLoader, trainingDate, currentEpoch, showMinimumPlots):
        self.accelerator.print(f"\nPlotting results for the {modelPipeline.model.datasetName} model")

        # Prepare the model/data for evaluation.
        self.setModelSavingFolder(baseSavingFolder=f"trainingFigures/{submodel}/{trainingDate}/", stringID=f"{modelPipeline.model.datasetName}/", epoch=currentEpoch)
        modelPipeline.setupTrainingFlags(modelPipeline.model, trainingFlag=False)  # Set all models into evaluation mode.
        model = modelPipeline.model
        numPlottingPoints = 3

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
            validDataMask, reconstructedSignalData, resampledSignalData, healthProfile, activityProfile, basicEmotionProfile, emotionProfile = model.forward(submodel, signalData, signalIdentifiers, metadata, device=self.accelerator.device, onlyProfileTraining=False)
            reconstructedHealthProfile = model.reconstructHealthProfile(resampledSignalData)  # reconstructedHealthProfile: batchSize, encodedDimension
            forwardModelPassSignals = model.specificSignalEncoderModel.profileModel.compiledLayerStates
            # forwardModelPassSignals: numModuleLayers, batchSize, numSignals, encodedDimension

            # Extract the model's internal variables.
            retrainingHealthProfilePath = np.asarray(model.specificSignalEncoderModel.profileModel.retrainingHealthProfilePath)  # numProfileShots, numExperiments, encodedDimension
            generatingBiometricSignals = np.asarray(model.specificSignalEncoderModel.profileModel.generatingBiometricSignals)  # numProfileShots, numModuleLayers, numExperiments, numSignals=1***, encodedDimension
            resampledBiomarkerTimes = model.sharedSignalEncoderModel.hyperSampledTimes.detach().cpu().numpy().astype(np.float16)  # numTimePoints
            backwardModelPassSignals = np.flip(forwardModelPassSignals, axis=0)

            # Detach the data from the GPU and tensor format.
            reconstructedHealthProfile, activityProfile, basicEmotionProfile, emotionProfile = reconstructedHealthProfile.detach().cpu().numpy(), activityProfile.detach().cpu().numpy().astype(np.float16), basicEmotionProfile.detach().cpu().numpy().astype(np.float16), emotionProfile.detach().cpu().numpy().astype(np.float16)
            validDataMask, reconstructedSignalData, resampledSignalData, healthProfile = validDataMask.detach().cpu().numpy().astype(np.float16), reconstructedSignalData.detach().cpu().numpy().astype(np.float16), resampledSignalData.detach().cpu().numpy().astype(np.float16), healthProfile.detach().cpu().numpy()
            signalData = signalData.detach().cpu().numpy().astype(np.float16)
            
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

                if submodel == modelConstants.signalEncoderModel:
                    # Plot the health profile training information.
                    self.signalEncoderViz.plotProfilePath(relativeTimes=resampledBiomarkerTimes, retrainingProfilePath=retrainingHealthProfilePath[:, :, None, :], epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Health profile generation")
                    self.signalEncoderViz.plotProfileReconstructionError(resampledBiomarkerTimes, healthProfile, reconstructedHealthProfile, epoch=currentEpoch, batchInd=batchInd, saveFigureLocation="signalEncoding/", plotTitle="Health profile reconstruction error")
                    self.signalEncoderViz.plotProfileReconstruction(resampledBiomarkerTimes, healthProfile, reconstructedHealthProfile, epoch=currentEpoch, batchInd=batchInd, saveFigureLocation="signalEncoding/", plotTitle="Health profile reconstruction")

                    # Plot the scale factor information.
                    self.signalEncoderViz.plotScaleFactorLines(scalingFactorsPath, reversibleModuleNames, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Normalization factors")

                    # Plot the angular information.
                    if not showMinimumPlots:
                        # Plot information collected across profile training.
                        self.signalEncoderViz.plotProfilePath(relativeTimes=resampledBiomarkerTimes, retrainingProfilePath=generatingBiometricSignals, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Reconstructing biometric feature signal")

                        # Plotting the data flow within the model.
                        self.signalEncoderViz.plotProfilePath(relativeTimes=resampledBiomarkerTimes, retrainingProfilePath=backwardModelPassSignals, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Backwards transformations (HP to feature)")
                        self.signalEncoderViz.modelFlow(dataTimes=resampledBiomarkerTimes, dataStatesAll=backwardModelPassSignals, signalNames=signalNames, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Signal transformations by layer 3D", batchInd=batchInd)
                        self.signalEncoderViz.plotSignalEncodingStatePath(relativeTimes=resampledBiomarkerTimes, compiledSignalEncoderLayerStates=backwardModelPassSignals, batchInd=batchInd, signalNames=signalNames, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Signal transformations by layer heatmap")

                        # Plot the angle information.
                        if currentEpoch % 10 == 0: self.signalEncoderViz.plotsGivensAnglesHeatmap(givensAnglesPath, reversibleModuleNames, signalInd=signalInd, epoch=currentEpoch, degreesFlag=False, saveFigureLocation="signalEncoding/", plotTitle="Rotation weight matrix (S)")
                        if currentEpoch % 10 == 0: self.signalEncoderViz.plotAngleLocations(givensAnglesPath, reversibleModuleNames, signalNames=signalNames, epoch=currentEpoch, signalInd=signalInd, saveFigureLocation="signalEncoding/", plotTitle="Givens transformations")
                        self.signalEncoderViz.plotsGivensAnglesHist(givensAnglesPath, reversibleModuleNames, epoch=currentEpoch, signalInd=signalInd, degreesFlag=False, saveFigureLocation="signalEncoding/", plotTitle="Rotation angles hist")

                        # Plot the activation information.
                        self.signalEncoderViz.plotActivationCurvesCompressed(activationCurvePath, activationModuleNames, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Activation forward and inverse curves (compressed)")
                        self.signalEncoderViz.plotActivationCurves(activationCurvePath, activationModuleNames, epoch=currentEpoch, saveFigureLocation="signalEncoding/", plotTitle="Activation forward and inverse curves")

                        # Plot the autoencoder results.
                        self.signalEncoderViz.plotEncoder(signalData, reconstructedSignalData, resampledBiomarkerTimes, resampledSignalData, signalNames=signalNames, epoch=currentEpoch, batchInd=batchInd, saveFigureLocation="signalReconstruction/", plotTitle="Signal reconstruction")

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
