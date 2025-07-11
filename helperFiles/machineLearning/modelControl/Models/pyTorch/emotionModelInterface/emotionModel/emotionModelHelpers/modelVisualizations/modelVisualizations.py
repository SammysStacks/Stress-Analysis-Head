from matplotlib import pyplot as plt
import numpy as np
import torch
import os

from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelVisualizations._emotionModelVisualizations import emotionModelVisualizations
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelVisualizations._generalVisualizations import generalVisualizations
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelVisualizations._signalEncoderVisualizations import signalEncoderVisualizations
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleLieLayer import reversibleLieLayer


class modelVisualizations(globalPlottingProtocols):

    def __init__(self, accelerator, datasetName, activityLabelInd):
        super(modelVisualizations, self).__init__(interactivePlots=False)
        self.activityLabelInd = activityLabelInd
        self.accelerator = accelerator
        self.datasetName = datasetName
        plt.ioff()  # Turn off interactive mode

        # Initialize helper classes.
        self.signalEncoderViz = signalEncoderVisualizations(baseSavingFolder="", stringID="", datasetName=datasetName)
        self.emotionModelViz = emotionModelVisualizations(baseSavingFolder="", stringID="", datasetName=datasetName)
        self.generalViz = generalVisualizations(baseSavingFolder="", stringID="", datasetName="_comparison")

    def setModelSavingFolder(self, baseSavingFolder, stringID, epoch=None):
        # Compile and shorten the name of the model visualization folder.
        baseSavingDataFolder = os.path.normpath(os.path.dirname(__file__) + f"/../../../dataAnalysis/{baseSavingFolder}") + '/'
        saveDataFolder = os.path.normpath(baseSavingDataFolder + stringID + '/') if stringID else baseSavingDataFolder

        # Set the saving folder for the model visualizations.
        self.baseSavingDataFolder = os.path.relpath(baseSavingDataFolder, os.getcwd()) + '/'
        self.saveDataFolder = os.path.relpath(saveDataFolder, os.getcwd()) + '/'
        if self.hpcFlag and epoch == 0 and os.path.exists(self.saveDataFolder): raise Exception(f"Folder already exists: {self.saveDataFolder}")
        self._createFolder(self.saveDataFolder)

        # Initialize visualization protocols.
        self.emotionModelViz.setSavingFolder(self.baseSavingDataFolder, stringID, self.datasetName)
        self.signalEncoderViz.setSavingFolder(self.baseSavingDataFolder, stringID, self.datasetName)
        self.generalViz.setSavingFolder(self.baseSavingDataFolder, stringID, datasetName="_comparison")

    # ---------------------------------------------------------------------- #

    def plotDatasetComparison(self, submodel, allModelPipelines, trainingModelName, showMinimumPlots):
        self.accelerator.print(f"\nCalculating loss for model comparison")

        # Prepare the model/data for evaluation.
        self.setModelSavingFolder(baseSavingFolder=f"trainingFigures/{submodel}/{trainingModelName}/", stringID="", epoch=-1)  # Label the correct folder to save this analysis.

        with torch.no_grad():
            if self.accelerator.is_local_main_process:

                if submodel == modelConstants.signalEncoderModel:
                    # Compile and plot the signal encoder model.
                    specificSignalEncoderModels = [modelPipeline.model.specificSignalEncoderModel for modelPipeline in allModelPipelines]  # Dim: numModels
                    self.generalPlotting(submodel, allModelPipelines, specificSignalEncoderModels, showMinimumPlots=showMinimumPlots, modelIdentifier="SignalEncoder", submodelString="SignalEncoderModel")
                elif submodel == modelConstants.emotionModel:
                    # Compile and plot the activity model.
                    specificActivityModels = [modelPipeline.model.specificActivityModel for modelPipeline in allModelPipelines]  # Dim: numModels
                    self.generalPlotting(submodel, allModelPipelines, specificActivityModels, showMinimumPlots=showMinimumPlots, modelIdentifier="Activity", submodelString="ActivityModel")

                    # Compile and plot the emotion model.
                    specificEmotionModels = [modelPipeline.model.specificEmotionModel for modelPipeline in allModelPipelines]  # Dim: numModels
                    self.generalPlotting(submodel, allModelPipelines, specificEmotionModels, showMinimumPlots=showMinimumPlots, modelIdentifier="Emotion", submodelString="EmotionModel")

    def generalPlotting(self, submodel, allModelPipelines, specificModels, showMinimumPlots, modelIdentifier, submodelString):
        datasetNames = [modelPipeline.model.datasetName for modelPipeline in allModelPipelines]  # Dim: numModels
        numEpochs = allModelPipelines[0].getTrainingEpoch(submodel)  # Dim: numModels

        # Plot reconstruction loss for the signal encoder.
        self.generalViz.plotTrainingLosses(trainingLosses=[specificModel.trainingLosses_signalReconstruction for specificModel in specificModels],
                                           testingLosses=[specificModel.testingLosses_signalReconstruction for specificModel in specificModels], numEpochs=numEpochs,
                                           saveFigureLocation=f"trainingLosses{modelIdentifier}/", plotTitle=f"{modelIdentifier} convergence losses")

        if submodel == modelConstants.signalEncoderModel:
            # Plot the losses during few-shot retraining the profile.
            self.generalViz.plotTrainingLosses(trainingLosses=[np.nanmean(specificModel.profileModel.retrainingProfileLosses, axis=1) for specificModel in specificModels], testingLosses=None,
                                               numEpochs=numEpochs, saveFigureLocation=f"trainingLosses{modelIdentifier}/", plotTitle=f"{modelIdentifier} profile convergence losses")

        freeParamInformation = np.asarray([modelPipeline.model.getFreeParamsFullPassPath(submodelString=submodelString)[1:] for modelPipeline in allModelPipelines])
        moduleNames, maxFreeParamsPath = freeParamInformation[:, 0], freeParamInformation[:, 1].astype(int)  # numFreeParamsPath: numModuleLayers, numSignals, numParams=1
        numFreeModelParams = [specificModel.numFreeParams for specificModel in specificModels]  # numModels, loadSubmodelEpochs, numModuleLayers, numSignals, numParams=1
        self.generalViz.plotFreeParamFlow(numFreeModelParams, maxFreeParamsPath, paramNames=["Free params"], moduleNames=moduleNames, saveFigureLocation=f"trainingLosses{modelIdentifier}/", plotTitle=f"{modelIdentifier} free parameters path zoomed")
        for modelInd in range(len(numFreeModelParams)): print('numFreeModelParams:', numFreeModelParams[modelInd][-1][0].mean(), numFreeModelParams[modelInd][-1][1].mean())
        if showMinimumPlots: return None

        # Plot the activation parameters for the signal encoder.
        paramNames = ["Infinite Bound", "Non-Linear Coefficient", "Convergent Point"]
        activationParamsPaths = np.asarray([specificModel.activationParamsPath for specificModel in specificModels])  # numModels, loadSubmodelEpochs, numActivations, numActivationParams=3
        self.generalViz.plotActivationFlowCompressed(activationParamsPaths=activationParamsPaths, moduleNames=moduleNames, modelLabels=datasetNames, paramNames=paramNames, saveFigureLocation=f"trainingLosses{modelIdentifier}/", plotTitle=f"{modelIdentifier} activation parameter compressed path")
        self.generalViz.plotActivationFlow(activationParamsPaths=activationParamsPaths, moduleNames=moduleNames, paramNames=paramNames, saveFigureLocation=f"trainingLosses{modelIdentifier}/", plotTitle=f"{modelIdentifier} activation parameter path")

        # Plot the angle features for the signal encoder.
        givensAnglesFeatureNames = reversibleLieLayer.getFeatureNames()
        givensAnglesFeaturesPaths = [specificModel.givensAnglesFeaturesPath for specificModel in specificModels]  # numModels, loadSubmodelEpochs, numModuleLayers, numFeatures, numValues
        self.generalViz.plotGivensFeaturesPath(givensAnglesFeaturesPaths=givensAnglesFeaturesPaths, paramNames=givensAnglesFeatureNames, moduleNames=moduleNames, saveFigureLocation=f"trainingLosses{modelIdentifier}/", plotTitle=f"{modelIdentifier} angular features path")

        # Plot the scaling factors for the signal encoder.
        normalizationFactorsPaths = [specificModel.normalizationFactorsPath for specificModel in specificModels]  # numModels, loadSubmodelEpochs, numModuleLayers, numSignals, numParams=1
        self.generalViz.plotNormalizationFactorFlow(normalizationFactorsPaths, paramNames=["normalization factors"], moduleNames=moduleNames, saveFigureLocation=f"trainingLosses{modelIdentifier}/", plotTitle=f"{modelIdentifier} normalization factors path")
        return None

    def plotAllTrainingEvents(self, submodel, modelPipeline, lossDataLoader, trainingModelName, currentEpoch, showMinimumPlots):
        self.accelerator.print(f"\nPlotting results for the {modelPipeline.model.datasetName} model")

        # Prepare the model/data for evaluation.
        self.setModelSavingFolder(baseSavingFolder=f"trainingFigures/{submodel}/{trainingModelName}/", stringID=f"{modelPipeline.model.datasetName}/", epoch=currentEpoch)
        modelPipeline.setupTrainingFlags(modelPipeline.model, trainingFlag=False)  # Set all models into evaluation mode.

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

        # Prepare the model/data for evaluation.
        numPlottingPoints = 3 if submodel == modelConstants.signalEncoderModel else validBatchMask.sum()
        model = modelPipeline.model

        with torch.no_grad():
            # Pass all the data through the model and store the emotions, activity, and intermediate variables.
            allLabels, allTrainingLabelMask, allTestingLabelMask, allTrainingSignalMask, allTestingSignalMask = allLabels[validBatchMask], allTrainingLabelMask[validBatchMask], allTestingLabelMask[validBatchMask], allTrainingSignalMask[validBatchMask], allTestingSignalMask[validBatchMask]
            signalData, signalIdentifiers, metadata = allSignalData[validBatchMask][:numPlottingPoints], allSignalIdentifiers[validBatchMask][:numPlottingPoints], allMetadata[validBatchMask][:numPlottingPoints]

            # Pass all the data through the model and store the emotions, activity, and intermediate variables.
            validDataMask, reconstructedSignalData, resampledSignalData, healthProfile, activityProfile, basicEmotionProfile, emotionProfile = model.forward(submodel, signalData, signalIdentifiers, metadata, device=self.accelerator.device, compiledLayerStates=submodel == modelConstants.signalEncoderModel)

            # Detach the data from the GPU and tensor format.
            validDataMask, reconstructedSignalData, healthProfile = validDataMask.detach().cpu().numpy().astype(np.float16), reconstructedSignalData.detach().cpu().numpy().astype(np.float16), healthProfile.detach().cpu().numpy()
            signalData = signalData.detach().cpu().numpy().astype(np.float16)
            signalNames, emotionNames = model.featureNames, model.emotionNames
            batchInd, signalInd, emotionInd = 0, 0, 0

            # Plot the loss on the primary process.
            if self.accelerator.is_local_main_process:

                # ------------------- Signal Encoding Plots -------------------- #

                if submodel == modelConstants.signalEncoderModel:
                    # Perform the backward pass through the model to get the reconstructed health profile.
                    reconstructedHealthProfile = model.reconstructHealthProfile(resampledSignalData).detach().cpu().numpy()  # reconstructedHealthProfile: batchSize, encodedDimension
                    forwardModelPassSignals = model.specificSignalEncoderModel.profileModel.compiledLayerStates  # numModuleLayers, batchSize, numSignals, encodedDimension
                    resampledSignalData = resampledSignalData.detach().cpu().numpy()  # resampledSignalData: batchSize, encodedDimension

                    # Extract the model's internal variables.
                    retrainingHealthProfilePath = np.asarray(model.specificSignalEncoderModel.profileModel.retrainingHealthProfilePath)  # numProfileShots, numExperiments, encodedDimension
                    generatingBiometricSignals = np.asarray(model.specificSignalEncoderModel.profileModel.generatingBiometricSignals)  # numProfileShots, numModuleLayers, numExperiments, numSignals=1***, encodedDimension
                    resampledBiomarkerTimes = model.sharedSignalEncoderModel.hyperSampledTimes.detach().cpu().numpy().astype(np.float16)  # numTimePoints
                    backwardModelPassSignals = np.flip(forwardModelPassSignals, axis=0)  # numModuleLayers, batchSize, numSignals, encodedDimension

                    # Compile additional information for the model.getActivationParamsFullPassPath
                    givensAnglesPath, normalizationFactorsPath, _, reversibleModuleNames, givensAnglesFeatureNames = model.getLearnableParams(submodelString="SignalEncoderModel")
                    activationCurvePath, activationModuleNames = model.getActivationCurvesFullPassPath(submodelString="SignalEncoderModel")  # numModuleLayers, 2=(x, y), numPoints=100
                    _, _, maxFreeParamsPath = model.getFreeParamsFullPassPath(submodelString="SignalEncoderModel")
                    # normalizationFactorsPath: numModuleLayers, numSignals, numParam=1
                    # givensAnglesPath: numModuleLayers, numSignals, numParams

                    # Plot the health profile training information.
                    self.signalEncoderViz.plotProfilePath(initialSignalData=None, relativeTimes=resampledBiomarkerTimes, retrainingProfilePath=retrainingHealthProfilePath[:, :, None, :], batchInd=batchInd, signalNames=signalNames, epoch=currentEpoch, saveFigureLocation="SignalEncoderModel/", plotTitle="Health profile generation")
                    self.signalEncoderViz.plotProfileReconstructionError(resampledBiomarkerTimes, healthProfile, reconstructedHealthProfile, epoch=currentEpoch, batchInd=batchInd, saveFigureLocation="SignalEncoderModel/", plotTitle="Health profile reconstruction error")
                    self.signalEncoderViz.plotProfileReconstruction(resampledBiomarkerTimes, healthProfile, reconstructedHealthProfile, epoch=currentEpoch, batchInd=batchInd, saveFigureLocation="SignalEncoderModel/", plotTitle="Health profile reconstruction")

                    # Plot the scale factor information.
                    self.signalEncoderViz.plotNormalizationFactors(normalizationFactorsPath, reversibleModuleNames, epoch=currentEpoch, saveFigureLocation="SignalEncoderModel/", plotTitle="Normalization factors")

                    # Plot the angular information.
                    if not showMinimumPlots:
                        # Plot information collected across profile training.
                        self.signalEncoderViz.plotProfilePath(initialSignalData=signalData, relativeTimes=resampledBiomarkerTimes, retrainingProfilePath=generatingBiometricSignals, batchInd=batchInd, signalNames=signalNames, epoch=currentEpoch, saveFigureLocation="SignalEncoderModel/", plotTitle="Reconstructing biometric feature signal")

                        # Plotting the data flow within the model.
                        self.signalEncoderViz.plotProfilePath(initialSignalData=None, relativeTimes=resampledBiomarkerTimes, retrainingProfilePath=backwardModelPassSignals, batchInd=batchInd, signalNames=signalNames, epoch=currentEpoch, saveFigureLocation="SignalEncoderModel/", plotTitle="Backwards transformations (HP to feature)")
                        self.signalEncoderViz.modelFlow(dataTimes=resampledBiomarkerTimes, dataStatesAll=backwardModelPassSignals, batchInd=batchInd, signalNames=signalNames, epoch=currentEpoch, saveFigureLocation="SignalEncoderModel/", plotTitle="Signal transformations by layer 3D")
                        self.signalEncoderViz.plotSignalEncodingStatePath(relativeTimes=resampledBiomarkerTimes, compiledSignalEncoderLayerStates=backwardModelPassSignals, batchInd=batchInd, signalNames=signalNames, epoch=currentEpoch, saveFigureLocation="SignalEncoderModel/", plotTitle="Signal transformations by layer heatmap")

                        # Plot the angle information.
                        self.signalEncoderViz.plotsGivensAnglesHeatmap(givensAnglesPath, reversibleModuleNames, signalInd=signalInd, epoch=currentEpoch, saveFigureLocation="SignalEncoderModel/", plotTitle="Rotation weight matrix (S)")
                        self.signalEncoderViz.plotsGivensAnglesHist(givensAnglesPath, reversibleModuleNames, epoch=currentEpoch, signalInd=signalInd, saveFigureLocation="SignalEncoderModel/", plotTitle="Rotation angles hist")

                        # Plot the activation information.
                        self.signalEncoderViz.plotActivationCurvesCompressed(activationCurvePath, activationModuleNames, epoch=currentEpoch, saveFigureLocation="SignalEncoderModel/", plotTitle="Activation forward and inverse curves (compressed)")
                        self.signalEncoderViz.plotActivationCurves(activationCurvePath, activationModuleNames, epoch=currentEpoch, saveFigureLocation="SignalEncoderModel/", plotTitle="Activation forward and inverse curves")

                        # Plot the autoencoder results.
                        self.signalEncoderViz.plotEncoder(signalData, reconstructedSignalData, resampledBiomarkerTimes, resampledSignalData, signalNames=signalNames, epoch=currentEpoch, batchInd=batchInd, saveFigureLocation="signalReconstruction/", plotTitle="Signal reconstruction")

                # Dont keep plotting untrained models.
                if submodel == modelConstants.signalEncoderModel: return None

                # ------------------ Activity Prediction Plots ------------------ #
                # activityProfile: batchSize, encodedDimension

                # Compile additional information for the model.getActivationParamsFullPassPath
                activityGivensAnglesPath, activityNormalizationFactorsPath, _, reversibleModuleNames, givensAnglesFeatureNames = model.getLearnableParams(submodelString="ActivityModel")
                activityActivationCurvePath, activityActivationModuleNames = model.getActivationCurvesFullPassPath(submodelString="ActivityModel")  # numModuleLayers, 2=(x, y), numPoints=100
                # normalizationFactorsPath: numModuleLayers, numSignals, numParam=1
                # givensAnglesPath: numModuleLayers, numSignals, numParams

                # Get the activity classes.
                activityClassDistribution = emotionDataInterface.getActivityClassProfile(activityProfile, model.numActivities)
                activityClasses = activityClassDistribution.argmax(dim=-1)  # activityClasses: batchSize
                # activityClassDistribution: batchSize, numActivities

                # Get the training and testing masks.
                activityTrainingMask = emotionDataInterface.getActivityColumn(allTrainingLabelMask, self.activityLabelInd)
                activityTestingMask = emotionDataInterface.getActivityColumn(allTestingLabelMask, self.activityLabelInd)

                # Separate the training and testing data.
                predictedActivityTrainingClasses = activityClasses[activityTrainingMask]
                predictedActivityTestingClasses = activityClasses[activityTestingMask]

                # Separate the training and testing data.
                trueActivityTrainingClasses = emotionDataInterface.getActivityLabels(allLabels, allTrainingLabelMask, self.activityLabelInd)  # trueActivityClasses: batchSize
                trueActivityTestingClasses = emotionDataInterface.getActivityLabels(allLabels, allTestingLabelMask, self.activityLabelInd)  # trueActivityClasses: batchSize

                # Detach the tensors.
                predictedActivityTrainingClasses, predictedActivityTestingClasses = predictedActivityTrainingClasses.detach().cpu().numpy(), predictedActivityTestingClasses.detach().cpu().numpy()
                trueActivityTrainingClasses, trueActivityTestingClasses = trueActivityTrainingClasses.detach().cpu().numpy(), trueActivityTestingClasses.detach().cpu().numpy()
                activityProfile = activityProfile.detach().cpu().numpy()

                # Plot the activity profile.
                self.emotionModelViz.plotDistributions(activityProfile[:, None, :], distributionNames=['Activity'], epoch=currentEpoch, batchInd=batchInd, showMinimumPlots=showMinimumPlots, saveFigureLocation="ActivityModel/", plotTitle="Activity profile")
                self.emotionModelViz.plotPredictedMatrix(trueActivityTrainingClasses, trueActivityTestingClasses, predictedActivityTrainingClasses, predictedActivityTestingClasses, numClasses=model.numActivities, epoch=currentEpoch, saveFigureLocation="ActivityModel/", plotTitle="Activity confusion matrix")

                if not showMinimumPlots:
                    # Plot the scale factor information.
                    self.signalEncoderViz.plotNormalizationFactors(activityNormalizationFactorsPath, reversibleModuleNames, epoch=currentEpoch, saveFigureLocation="ActivityModel/", plotTitle="Normalization factors")

                    # Plot the angle information.
                    self.signalEncoderViz.plotsGivensAnglesHeatmap(activityGivensAnglesPath, reversibleModuleNames, signalInd=signalInd, epoch=currentEpoch, saveFigureLocation="ActivityModel/", plotTitle="Rotation weight matrix (S)")
                    self.signalEncoderViz.plotsGivensAnglesHist(activityGivensAnglesPath, reversibleModuleNames, epoch=currentEpoch, signalInd=signalInd, saveFigureLocation="ActivityModel/", plotTitle="Rotation angles hist")

                    # Plot the activation information.
                    self.signalEncoderViz.plotActivationCurvesCompressed(activityActivationCurvePath, activityActivationModuleNames, epoch=currentEpoch, saveFigureLocation="ActivityModel/", plotTitle="Activation forward and inverse curves (compressed)")
                    self.signalEncoderViz.plotActivationCurves(activityActivationCurvePath, activityActivationModuleNames, epoch=currentEpoch, saveFigureLocation="ActivityModel/", plotTitle="Activation forward and inverse curves")

                # ------------------ Emotion Prediction Plots ------------------ #
                # basicEmotionProfile: batchSize, numEmotions, numBasicEmotions, encodedDimension
                # emotionProfile: batchSize, numEmotions, encodedDimension
                numBasicEmotions = model.numBasicEmotions
                emotionNames = model.emotionNames

                # Get the emotion classes.
                allEmotionClassPredictions = emotionDataInterface.getEmotionClassPredictions(emotionProfile, model.allEmotionClasses, emotionProfile.device)
                # allEmotionClassPredictions: numEmotions, batchSize, numEmotionClasses

                # Detach the tensors.
                allLabels, allTrainingLabelMask, allTestingLabelMask = allLabels.detach().cpu().numpy(), allTrainingLabelMask.detach().cpu().numpy(), allTestingLabelMask.detach().cpu().numpy()
                basicEmotionProfile, emotionProfile = basicEmotionProfile.detach().cpu().numpy(), emotionProfile.detach().cpu().numpy()

                # Plot the activity profile.
                self.emotionModelViz.plotDistributions(emotionProfile, distributionNames=emotionNames, epoch=currentEpoch, batchInd=batchInd, showMinimumPlots=showMinimumPlots, saveFigureLocation="EmotionModel/", plotTitle="Emotion profile")

                for emotionInd in range(model.numEmotions):
                    emotionClassDistributions = allEmotionClassPredictions[emotionInd].detach().cpu().numpy()  # batchSize, numEmotionClasses
                    emotionClasses = emotionClassDistributions.argmax(axis=-1)  # batchSize
                    numClasses = model.allEmotionClasses[emotionInd]
                    emotionName = model.emotionNames[emotionInd]

                    # Get the training and testing data.
                    trueEmotionTrainingClasses = emotionDataInterface.getEmotionLabels(emotionInd, allLabels, allTrainingLabelMask)
                    trueEmotionTestingClasses = emotionDataInterface.getEmotionLabels(emotionInd, allLabels, allTestingLabelMask)

                    # Get the training and testing masks.
                    emotionTrainingMask = emotionDataInterface.getEmotionColumn(allTrainingLabelMask, emotionInd)
                    emotionTestingMask = emotionDataInterface.getEmotionColumn(allTestingLabelMask, emotionInd)

                    # Separate the training and testing predictions.
                    predictedEmotionTrainingClasses = emotionClasses[emotionTrainingMask]
                    predictedEmotionTestingClasses = emotionClasses[emotionTestingMask]

                    self.emotionModelViz.plotDistributions(basicEmotionProfile[:, emotionInd], distributionNames=[f"Basic{i}" for i in range(numBasicEmotions)], epoch=currentEpoch, batchInd=batchInd, showMinimumPlots=False, saveFigureLocation="EmotionModel/", plotTitle=f"Basic emotion profile {emotionName}")
                    self.emotionModelViz.plotPredictedMatrix(trueEmotionTrainingClasses, trueEmotionTestingClasses, predictedEmotionTrainingClasses, predictedEmotionTestingClasses, numClasses=numClasses, epoch=currentEpoch, saveFigureLocation="EmotionModel/", plotTitle=f"{emotionName} confusion matrix")
                    if showMinimumPlots: break

        return None
