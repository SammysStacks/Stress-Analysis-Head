import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.lossInformation.lossCalculations import lossCalculations
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelParameters import modelParameters
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleLieLayer import reversibleLieLayer
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleInterface import reversibleInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.sharedActivityModel import sharedActivityModel
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.sharedEmotionModel import sharedEmotionModel
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.sharedSignalEncoderModel import sharedSignalEncoderModel
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.specificActivityModel import specificActivityModel
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.specificEmotionModel import specificEmotionModel
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.specificSignalEncoderModel import specificSignalEncoderModel


class emotionModelHead(nn.Module):
    def __init__(self, submodel, emotionNames, activityNames, featureNames, numSubjects, datasetName, numExperiments):
        super(emotionModelHead, self).__init__()
        # General model parameters.
        self.numActivities = len(activityNames)  # The number of activities to predict.
        self.numEmotions = len(emotionNames)  # The number of emotions to predict.
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.featureNames = featureNames  # The names of each feature/signal in the model. Dim: numSignals
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.numSubjects = numSubjects  # The maximum number of subjects the model is training on.
        self.datasetName = datasetName  # The name of the dataset the model is training on.
        self.numSignals = len(featureNames)  # The number of signals in the model.
        self.hpcFlag = 'HPC' in modelConstants.userInputParams['deviceListed']  # Flag to determine if the model is running on an HPC.

        # General parameters.
        self.encodedDimension = modelConstants.userInputParams['encodedDimension']  # The dimension of the encoded signal.
        self.operatorType = modelConstants.userInputParams['operatorType']  # The type of operator to use for the neural operator.
        self.debugging = True

        # Signal encoder parameters.
        self.reversibleLearningProtocol = modelConstants.userInputParams['reversibleLearningProtocol']   # The learning protocol for the model.
        self.neuralOperatorParameters = modelConstants.userInputParams['neuralOperatorParameters']   # The parameters for the neural operator.
        self.numSpecificEncoderLayers = modelConstants.userInputParams['numSpecificEncoderLayers']  # The number of specific layers.
        self.numSharedEncoderLayers = modelConstants.userInputParams['numSharedEncoderLayers']  # The number of shared layers.

        # Emotion and activity parameters.
        self.irreversibleLearningProtocol = modelConstants.userInputParams['irreversibleLearningProtocol']  # The learning protocol for the model.
        self.numActivityModelLayers = modelConstants.userInputParams['numActivityModelLayers']  # The number of basic emotions (basis states of emotions).
        self.numEmotionModelLayers = modelConstants.userInputParams['numEmotionModelLayers']  # The number of basic emotions (basis states of emotions).
        self.numActivityChannels = modelConstants.userInputParams['numActivityChannels']  # The number of activity channels to predict.
        self.numBasicEmotions = modelConstants.userInputParams['numBasicEmotions']  # The number of basic emotions (basis states of emotions).

        # Setup holder for the model's training information
        self.calculateModelLosses = lossCalculations(accelerator=None, allEmotionClasses=None, activityLabelInd=None)

        # ------------------------ Data Compression ------------------------ #

        # The signal encoder model to find a common feature vector across all signals.
        self.specificSignalEncoderModel = specificSignalEncoderModel(
            neuralOperatorParameters=self.neuralOperatorParameters,
            numSpecificEncoderLayers=self.numSpecificEncoderLayers,
            learningProtocol=self.reversibleLearningProtocol,
            encodedDimension=self.encodedDimension,
            featureNames=self.featureNames,
            operatorType=self.operatorType,
            numExperiments=numExperiments,
        )

        # The autoencoder model reduces the incoming signal's dimension.
        self.sharedSignalEncoderModel = sharedSignalEncoderModel(
            neuralOperatorParameters=self.neuralOperatorParameters,
            numSharedEncoderLayers=self.numSharedEncoderLayers,
            learningProtocol=self.reversibleLearningProtocol,
            encodedDimension=self.encodedDimension,
            operatorType=self.operatorType,
        )

        # -------------------- Final Emotion Prediction -------------------- #

        if submodel == modelConstants.emotionModel:
            self.specificEmotionModel = specificEmotionModel(
                neuralOperatorParameters=self.neuralOperatorParameters,
                learningProtocol=self.irreversibleLearningProtocol,
                encodedDimension=self.encodedDimension,
                numBasicEmotions=self.numBasicEmotions,
                numModelLayers=self.numEmotionModelLayers,
                operatorType=self.operatorType,
                numSpecificEncoderLayers=self.numSpecificEncoderLayers,
                numEmotions=self.numEmotions,
                numSubjects=self.numSubjects,

            )

            self.sharedEmotionModel = sharedEmotionModel(
                neuralOperatorParameters=self.neuralOperatorParameters,
                learningProtocol=self.irreversibleLearningProtocol,
                encodedDimension=self.encodedDimension,
                numBasicEmotions=self.numBasicEmotions,
                numModelLayers=self.numEmotionModelLayers,
                operatorType=self.operatorType,
                numEmotions=self.numEmotions,
            )

            self.specificActivityModel = specificActivityModel(
                neuralOperatorParameters=self.neuralOperatorParameters,
                learningProtocol=self.irreversibleLearningProtocol,
                numActivityChannels=self.numActivityChannels,
                encodedDimension=self.encodedDimension,
                numModelLayers=self.numActivityModelLayers,
                numActivities=self.numActivities,
                operatorType=self.operatorType,
                numSpecificEncoderLayers=self.numSpecificEncoderLayers,
            )

            self.sharedActivityModel = sharedActivityModel(
                neuralOperatorParameters=self.neuralOperatorParameters,
                learningProtocol=self.irreversibleLearningProtocol,
                numActivityChannels=self.numActivityChannels,
                encodedDimension=self.encodedDimension,
                numModelLayers=self.numActivityModelLayers,
                operatorType=self.operatorType,
            )

    # ------------------------- Full Forward Calls ------------------------- #

    def forward(self, submodel, signalData, signalIdentifiers, metadata, device, onlyProfileTraining=False):
        signalData, signalIdentifiers, metadata = (tensor.to(device) for tensor in (signalData, signalIdentifiers, metadata))
        signalIdentifiers, signalData, metadata = signalIdentifiers.int(), signalData.double(), metadata.int()
        batchSize, numSignals, maxSequenceLength, numChannels = signalData.size()
        assert numChannels == len(modelConstants.signalChannelNames)
        # timepoints: [further away from survey (300) -> closest to survey (0)]
        # signalData: [batchSize, numSignals, maxSequenceLength, numChannels]
        # signalIdentifiers: [batchSize, numSignals, numSignalIdentifiers]
        # metadata: [batchSize, numMetadata]

        # Initialize default output tensors.
        basicEmotionProfile = torch.zeros((batchSize, self.numBasicEmotions, self.encodedDimension), device=device)
        emotionProfile = torch.zeros((batchSize, self.numEmotions, self.encodedDimension), device=device)
        activityProfile = torch.zeros((batchSize, self.encodedDimension), device=device)

        # ------------------- Organize the Incoming Data ------------------- #

        # Check which points were missing in the data.
        batchInds = emotionDataInterface.getSignalIdentifierData(signalIdentifiers, channelName=modelConstants.batchIndexSI)[:, 0]  # Dim: batchSize
        validDataMask = emotionDataInterface.getValidDataMask(signalData)
        # validDataMask: batchSize, numSignals, maxSequenceLength

        # ------------------- Estimated Health Profile ------------------- #

        # Get the estimated profile weights.
        embeddedProfile = self.specificSignalEncoderModel.profileModel.getHealthEmbedding(batchInds)
        healthProfile = self.sharedSignalEncoderModel.generateHealthProfile(embeddedProfile)
        # embeddedProfile: batchSize, profileDimension
        # healthProfile: batchSize, encodedDimension

        # ------------------- Learned Signal Mapping ------------------- #

        # Perform the backward pass: health profile -> signal data.
        resampledSignalData = healthProfile.unsqueeze(1).repeat(repeats=(1, numSignals, 1))
        resampledSignalData = self.signalEncoderPass(metaLearningData=resampledSignalData, forwardPass=False, compileLayerStates=onlyProfileTraining)
        reconstructedSignalData = self.sharedSignalEncoderModel.interpolateOriginalSignals(signalData, resampledSignalData)
        # reconstructedSignalData: batchSize, numSignals, maxSequenceLength
        # resampledSignalData: batchSize, numSignals, encodedDimension

        # Visualize the data transformations within signal encoding.
        if submodel == modelConstants.signalEncoderModel and not onlyProfileTraining and random.random() < 0.1 and not self.hpcFlag:
            with torch.no_grad(): self.visualizeSignalEncoding(embeddedProfile, healthProfile, resampledSignalData, reconstructedSignalData, signalData, validDataMask)

        # ------------------- Learned Emotion Mapping ------------------- #

        if submodel == modelConstants.emotionModel:
            # Perform the backward pass: physiologically -> emotion data.
            reversibleInterface.changeDirections(forwardDirection=False)
            basicEmotionProfile = healthProfile.unsqueeze(1).repeat(repeats=(1, self.numBasicEmotions, 1))
            basicEmotionProfile = self.coreModelPass(self.numEmotionModelLayers, metaLearningData=basicEmotionProfile, specificModel=self.specificEmotionModel, sharedModel=self.sharedEmotionModel)
            # metaLearningData: batchSize, numEmotions*numBasicEmotions, encodedDimension

            # Reconstruct the emotion data.
            basicEmotionProfile = basicEmotionProfile.repeat(repeats=(1, self.numEmotions, 1, 1))
            subjectInds = emotionDataInterface.getMetaDataChannel(metadata, channelName=modelConstants.subjectIndexMD)  # Dim: batchSize
            emotionProfile = self.specificEmotionModel.calculateEmotionProfile(basicEmotionProfile, subjectInds)

        # ------------------- Learned Activity Mapping ------------------- #

            # Perform the backward pass: physiologically -> activity data.
            reversibleInterface.changeDirections(forwardDirection=False)
            resampledActivityData = healthProfile.unsqueeze(1).repeat(repeats=(1, self.numActivityChannels, 1))
            activityProfile = self.coreModelPass(self.numActivityModelLayers, metaLearningData=resampledActivityData, specificModel=self.specificActivityModel, sharedModel=self.sharedActivityModel)
            # metaLearningData: batchSize, numEmotions*numBasicEmotions, encodedDimension

        # --------------------------------------------------------------- #

        return validDataMask, reconstructedSignalData, resampledSignalData, healthProfile, activityProfile, basicEmotionProfile, emotionProfile

    # ------------------------- Model Components ------------------------- #

    @staticmethod
    def compileModuleName(name):
        # Initialize the model name.
        compiledName = ""; name = name.lower()

        # Add the model type information.
        if 'specific' in name: compiledName = 'Specific' + compiledName; assert 'shared' not in name
        elif 'shared' in name: compiledName = 'Shared' + compiledName; assert 'specific' not in name
        else: raise Exception("Invalid name:", name)

        # Add the neural layer information.
        if 'high' in name: compiledName = compiledName + ' neural high frequency'; assert 'low' not in name
        elif 'low' in name: compiledName = compiledName + ' neural low frequency'; assert 'high' not in name
        else: raise Exception("Invalid name:", name)

        return compiledName

    def getLearnableParams(self):
        # Initialize the learnable parameters.
        givensAnglesPath, scalingFactorsPath, givensAnglesFeaturesPath,  = [], [], []
        givensAnglesFeatureNames = reversibleLieLayer.getFeatureNames()
        numGivensFeatures = len(givensAnglesFeatureNames)
        reversibleModuleNames = []

        # For each module.
        for name, module in self.named_modules():
            if isinstance(module, reversibleLieLayer):
                _, allGivensAnglesFeatures = module.getFeatureParams()
                allGivensAngles, allScaleFactors = module.getAllLinearParams()
                if allScaleFactors.shape[0] == 0: continue

                for givensAngles in allGivensAngles: givensAnglesPath.append(givensAngles)  # givensAnglesPath: numModuleLayers, numSignals, numParams
                for scalingFactors in allScaleFactors: scalingFactorsPath.append(scalingFactors)  # scalingFactorsPath: numModuleLayers, numSignals, numParams=1
                for givensAnglesFeatures in allGivensAnglesFeatures: givensAnglesFeaturesPath.append(givensAnglesFeatures)  # givensAnglesFeaturesPath: numModuleLayers, numFeatures, numValues
                for _ in allGivensAngles: reversibleModuleNames.append(self.compileModuleName(name))

            elif isinstance(module, nn.Identity) and 'highFrequenciesWeights' in name:
                decompositionLevel = int(name.split('highFrequenciesWeights.')[-1]) + 1 if 'highFrequenciesWeights' in name else 0
                numLayers = modelConstants.userInputParams['numSharedEncoderLayers' if 'shared' in name else 'numSpecificEncoderLayers']
                sequenceLength = self.encodedDimension // 2**decompositionLevel
                numSignals = self.numSignals if 'specific' in name else 1

                for _ in range(numLayers): givensAnglesPath.append(np.zeros((numSignals, int(sequenceLength * (sequenceLength - 1) / 2))))  # givensAnglesPath: numModuleLayers, numSignals, numParams
                for _ in range(numLayers): scalingFactorsPath.append(np.ones((numSignals, 1)))  # scalingFactorsPath: numModuleLayers, numSignals, numParams=1
                for _ in range(numLayers): givensAnglesFeaturesPath.append(np.zeros((numGivensFeatures, numSignals)))  # givensAnglesFeaturesPath: numModuleLayers, numFeatures, numValues
                for _ in range(numLayers): reversibleModuleNames.append(self.compileModuleName(name))

        return givensAnglesPath, scalingFactorsPath, givensAnglesFeaturesPath, reversibleModuleNames, givensAnglesFeatureNames

    def getActivationCurvesFullPassPath(self):
        activationCurvePath, moduleNames = [], []

        for name, module in self.named_modules():
            if isinstance(module, reversibleLieLayer):
                xs, ys = module.geAllActivationCurves(x_min=-1.5, x_max=1.5, num_points=100)
                for ind in range(len(xs)): activationCurvePath.append([xs[ind], ys[ind]])
                for _ in range(len(xs)): moduleNames.append(self.compileModuleName(name))

            elif isinstance(module, nn.Identity) and 'highFrequenciesWeights' in name:
                numLayers = modelConstants.userInputParams['numSharedEncoderLayers' if 'shared' in name else 'numSpecificEncoderLayers']
                x = np.linspace(-1.5, stop=1.5, num=100); y = x

                for _ in range(numLayers): activationCurvePath.append([x, y])
                for _ in range(numLayers): moduleNames.append(self.compileModuleName(name))
        activationCurvePath = np.asarray(activationCurvePath)
        moduleNames = np.asarray(moduleNames)

        return activationCurvePath, moduleNames

    def cullAngles(self, epoch):
        for name, module in self.named_modules():
            sharedLayer = 'shared' in name.lower()
            if isinstance(module, reversibleLieLayer):
                module.angularThresholding(epoch=epoch, sharedLayer=sharedLayer)

    def getActivationParamsFullPassPath(self):
        activationParamsPath, moduleNames = [], []
        for name, module in self.named_modules():
            if isinstance(module, reversibleLieLayer):
                allActivationParams = module.getAllActivationParams()
                activationParamsPath.extend(allActivationParams)
                for _ in allActivationParams: moduleNames.append(self.compileModuleName(name))

            elif isinstance(module, nn.Identity) and 'highFrequenciesWeights' in name:
                numLayers = modelConstants.userInputParams['numSharedEncoderLayers' if 'shared' in name else 'numSpecificEncoderLayers']
                for _ in range(numLayers): moduleNames.append(self.compileModuleName(name))
                for _ in range(numLayers): activationParamsPath.append([0.5, 1, 1])
        assert len(activationParamsPath) != 0
        activationParamsPath = np.asarray(activationParamsPath)
        return activationParamsPath, moduleNames

    def getFreeParamsFullPassPath(self):
        numFreeParamsPath, moduleNames, maxFreeParamsPath = [], [], []

        for name, module in self.named_modules():
            if isinstance(module, reversibleLieLayer):
                allNumFreeParams = module.getNumFreeParams()

                # Compile the free parameters.
                for _ in range(len(allNumFreeParams)): moduleNames.append(self.compileModuleName(name))
                for _ in range(len(allNumFreeParams)): maxFreeParamsPath.append(module.numParams)  # maxFreeParamsPath: numModuleLayers
                numFreeParamsPath.extend(allNumFreeParams)  # numFreeParamsPath: numModuleLayers, numSignals, numParams=1

            elif isinstance(module, nn.Identity) and 'highFrequenciesWeights' in name:
                numLayers = modelConstants.userInputParams['numSharedEncoderLayers' if 'shared' in name else 'numSpecificEncoderLayers']
                numSignals = self.numSignals if 'specific' in name else 1

                # Compile the free parameters.
                for _ in range(numLayers): maxFreeParamsPath.append(0)  # maxFreeParamsPath: numModuleLayers
                for _ in range(numLayers): numFreeParamsPath.append(np.zeros((numSignals, 1)))  # numFreeParamsPath: numModuleLayers, numSignals, numParams=1
                for _ in range(numLayers): moduleNames.append(self.compileModuleName(name))
        assert len(numFreeParamsPath) != 0

        return numFreeParamsPath, moduleNames, maxFreeParamsPath

    def signalEncoderPass(self, metaLearningData, forwardPass, compileLayerStates=False):
        if compileLayerStates: self.specificSignalEncoderModel.profileModel.resetModelStates(metaLearningData)  # Add the initial state.
        compilingFunction = self.specificSignalEncoderModel.profileModel.addModelState if compileLayerStates else None
        reversibleInterface.changeDirections(forwardDirection=forwardPass)

        if forwardPass:
            # Signal encoder layers.
            metaLearningData = self.sharedSignalEncoderModel.learningInterface(signalData=metaLearningData, compilingFunction=compilingFunction)
            metaLearningData = self.specificSignalEncoderModel.learningInterface(signalData=metaLearningData, compilingFunction=compilingFunction)
            # metaLearningData: batchSize, numSignals, encodedDimension
        else:
            # Signal encoder layers.
            metaLearningData = self.specificSignalEncoderModel.learningInterface(signalData=metaLearningData, compilingFunction=compilingFunction)
            metaLearningData = self.sharedSignalEncoderModel.learningInterface(signalData=metaLearningData, compilingFunction=compilingFunction)
            # metaLearningData: batchSize, numSignals, encodedDimension

        return metaLearningData

    def reconstructHealthProfile(self, resampledSignalData):
        return self.signalEncoderPass(metaLearningData=resampledSignalData, forwardPass=True, compileLayerStates=True)

    def fullPass(self, submodel, signalData, signalIdentifiers, metadata, device, profileEpoch=None):
        # Preallocate the output tensors.
        numExperiments, numSignals, maxSequenceLength, numChannels = signalData.size()
        testingBatchSize = modelParameters.getInferenceBatchSize(submodel, device)
        onlyProfileTraining = profileEpoch is not None

        # Initialize the output tensors.
        basicEmotionProfile = torch.zeros((numExperiments, self.numBasicEmotions, self.encodedDimension), device='cpu')
        validDataMask = torch.zeros((numExperiments, numSignals, maxSequenceLength), device='cpu', dtype=torch.bool)
        emotionProfile = torch.zeros((numExperiments, self.numEmotions, self.encodedDimension), device='cpu')
        reconstructedSignalData = torch.zeros((numExperiments, numSignals, maxSequenceLength), device='cpu')
        resampledSignalData = torch.zeros((numExperiments, numSignals, self.encodedDimension), device='cpu')
        activityProfile = torch.zeros((numExperiments, self.encodedDimension), device='cpu')
        healthProfile = torch.zeros((numExperiments, self.encodedDimension), device='cpu')

        startBatchInd = 0
        while startBatchInd < numExperiments:
            endBatchInd = startBatchInd + testingBatchSize

            # Perform a full pass of the model.
            validDataMask[startBatchInd:endBatchInd], reconstructedSignalData[startBatchInd:endBatchInd], resampledSignalData[startBatchInd:endBatchInd], \
                healthProfile[startBatchInd:endBatchInd], activityProfile[startBatchInd:endBatchInd], basicEmotionProfile[startBatchInd:endBatchInd], emotionProfile[startBatchInd:endBatchInd] \
                = (element.cpu() if isinstance(element, torch.Tensor) else element for element in self.forward(submodel=submodel, signalData=signalData[startBatchInd:endBatchInd], signalIdentifiers=signalIdentifiers[startBatchInd:endBatchInd],
                                                                                                               metadata=metadata[startBatchInd:endBatchInd], device=device, onlyProfileTraining=onlyProfileTraining))
            startBatchInd = endBatchInd  # Update the batch index.

        if onlyProfileTraining:
            with torch.no_grad():
                batchInds = emotionDataInterface.getSignalIdentifierData(signalIdentifiers, channelName=modelConstants.batchIndexSI)[:, 0].long()  # Dim: batchSize
                batchLossValues = self.calculateModelLosses.calculateSignalEncodingLoss(signalData.cpu(), reconstructedSignalData, validDataMask, allSignalMask=None, averageBatches=False)
                self.specificSignalEncoderModel.profileModel.populateProfileState(profileEpoch, batchInds, batchLossValues, resampledSignalData, healthProfile)

        return validDataMask, reconstructedSignalData, resampledSignalData, healthProfile, activityProfile, basicEmotionProfile, emotionProfile

    # ------------------------- Model Visualizations ------------------------- #

    def visualizeSignalEncoding(self, embeddedProfile, healthProfile, resampledSignalData, reconstructedSignalData, signalData, validDataMask):
        # Find the first valid signal.
        validSignalMask = torch.any(validDataMask, dim=-1)
        firstBatchInd, firstSignalInd = validSignalMask.nonzero(as_tuple=False)[0, :]
        validPointMask = validDataMask[firstBatchInd, firstSignalInd]
        plt.close('all')

        # Optionally, plot the health profile for visual comparison
        resampledBiomarkerTimes = self.sharedSignalEncoderModel.hyperSampledTimes.clone().detach().cpu().numpy()
        plt.plot(resampledBiomarkerTimes, healthProfile[firstBatchInd].clone().detach().cpu().numpy(), 'o-', color='tab:red', linewidth=0.25, label='Health Profile', alpha=2/3, markersize=2)
        plt.plot(torch.linspace(start=resampledBiomarkerTimes[0], end=resampledBiomarkerTimes[-1], steps=embeddedProfile.size(-1)).clone().detach().cpu().numpy().astype(np.float16), embeddedProfile[firstBatchInd].clone().detach().cpu().numpy(), 'ok', linewidth=1, markersize=3,  label='Embedded Profile', alpha=0.75)
        plt.title(f"batchInd{firstBatchInd}")
        plt.ylim((-1.75, 1.75))
        plt.show()
        plt.close()

        # Get the first valid signal points.
        validReconstructedPoints = reconstructedSignalData[firstBatchInd, firstSignalInd, validPointMask].clone().detach().cpu().numpy()
        datapoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.signalChannel)
        timepoints = emotionDataInterface.getChannelData(signalData, channelName=modelConstants.timeChannel)
        validTimepoints = timepoints[firstBatchInd, firstSignalInd, validPointMask].clone().detach().cpu().numpy()
        validDatapoints = datapoints[firstBatchInd, firstSignalInd, validPointMask].clone().detach().cpu().numpy()

        # Optionally, plot the original and reconstructed signals for visual comparison
        plt.plot(validTimepoints, validDatapoints, 'ok', markersize=3, label='Initial Signal', alpha=0.75)
        plt.plot(validTimepoints, validReconstructedPoints, 'o-', color='tab:red', markersize=3, label='Reconstructed Signal', alpha=0.75, linewidth=0.25)
        plt.plot(resampledBiomarkerTimes, resampledSignalData[firstBatchInd, firstSignalInd, :].clone().detach().cpu().numpy(), 'tab:blue', linewidth=1, label='Resampled Signal', alpha=0.75)
        plt.title(f"batchInd{firstBatchInd} signalInd{firstSignalInd} numPoints{len(validTimepoints)}")
        plt.ylim((-1.75, 1.75))
        plt.legend()
        plt.show()
        plt.close()
