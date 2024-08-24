# General
import time

# PyTorch
import torch

# Import helper modules
from .emotionModelHelpers.emotionDataInterface import emotionDataInterface
from .emotionModelHelpers.generalMethods.dataAugmentation import dataAugmentation
from .emotionModelHelpers.modelConstants import modelConstants
from .emotionModelHelpers.submodels.autoencoderModel import autoencoderModel  # An autoencoder pipeline for compressing individual signals.
from .emotionModelHelpers.submodels.sharedEmotionModel import sharedEmotionModel
from .emotionModelHelpers.submodels.signalEncoderModel import signalEncoderModel  # A signal encoder pipeline to make a universal feature vector.
from .emotionModelHelpers.submodels.signalMappingModel import signalMappingModel
# Import submodels
from .emotionModelHelpers.submodels.specificEmotionModel import specificEmotionModel
from .emotionModelHelpers.submodels.trainingInformation import trainingInformation
from ..._globalPytorchModel import globalModel


class emotionModelHead(globalModel):
    def __init__(self, submodel, accelerator, finalDistributionLength, signalMinMaxScale, maxNumSignals, metadata, userInputParams,
                 timeWindows, emotionNames, activityNames, featureNames, numSubjects, datasetName, useFinalParams, debuggingResults=False):
        super(emotionModelHead, self).__init__()
        # General model parameters.
        self.sequenceBounds = (timeWindows[0], timeWindows[-1])  # The minimum and maximum sequence length for the model.
        self.finalDistributionLength = finalDistributionLength  # The final length of the signal distribution.
        self.metadata = metadata  # The subject identifiers for the model (e.g., subjectIndex, datasetIndex, etc.)
        self.signalMinMaxScale = signalMinMaxScale  # The minimum and maximum values for the signals.
        self.debuggingResults = debuggingResults  # Whether to print debugging results. Type: bool
        self.numActivities = len(activityNames)  # The number of activities to predict.
        self.numEmotions = len(emotionNames)  # The number of emotions to predict.
        self.numSignals = len(featureNames)  # The number of signals going into the model.
        self.maxNumSignals = maxNumSignals  # The maximum number of signals to consider.
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.featureNames = featureNames  # The names of each feature/signal in the model. Dim: numSignals
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.device = accelerator.device  # The device the model is running on.
        self.useFinalParams = useFinalParams  # Whether to use the HPC parameters.
        self.numSubjects = numSubjects  # The maximum number of subjects the model is training on.
        self.accelerator = accelerator  # Hugging face model optimizations.
        self.datasetName = datasetName  # The name of the dataset the model is training on.

        # Signal encoder parameters.
        self.signalEncoderWaveletType = userInputParams['signalEncoderWaveletType']         # The number of signals to group when you begin compression or finish expansion.
        self.numSigLiftedChannels = userInputParams['numSigLiftedChannels']     # The number of channels to lift to during signal encoding.
        self.numSigEncodingLayers = userInputParams['numSigEncodingLayers']     # The number of transformer layers during signal encoding.
        self.numExpandedSignals = userInputParams['numExpandedSignals']         # The number of signals to group when you begin compression or finish expansion.

        # Autoencoder parameters.
        self.compressionFactor = userInputParams['compressionFactor']       # The expansion factor of the autoencoder
        self.expansionFactor = userInputParams['expansionFactor']           # The expansion factor of the autoencoder
        # Emotion parameters.
        self.numInterpreterHeads = userInputParams['numInterpreterHeads']   # The number of ways to interpret a set of physiological signals.
        self.numBasicEmotions = userInputParams['numBasicEmotions']         # The number of basic emotions (basis states of emotions).

        # Tunable encoding parameters.
        self.numEncodedSignals = 1  # The final number of signals to accept, encoding all signal information.
        self.compressedLength = 64  # The final length of the compressed signal after the autoencoder.
        # Feature parameters (code changes required if you change these!!!)
        self.numCommonSignals = 8    # The number of features from considering all the signals.
        self.numEmotionSignals = 8   # The number of common activity features to extract.
        self.numActivitySignals = 8  # The number of common activity features to extract.

        # Setup holder for the model's training information
        self.trainingInformation = trainingInformation()

        # Initialize all the models.
        self.specificEmotionModel = None
        self.signalEncoderModel = None
        self.signalMappingModel = None
        self.sharedEmotionModel = None
        self.autoencoderModel = None

        # Populate the current models.
        self.initializeSubmodels(submodel)

    def setDebuggingResults(self, debuggingResults, submodels):
        # Set the debugging results for the emotion model.
        self.debuggingResults = debuggingResults

        # For each submodel.
        for submodel in submodels:
            # Set the debugging results for the submodel.
            submodel.setDebuggingResults(debuggingResults)

    def initializeSubmodels(self, submodel):

        # ------------------------ Data Compression ------------------------ # 

        # The signal encoder model to find a common feature vector across all signals.
        self.signalEncoderModel = signalEncoderModel(
            numSigLiftedChannels=self.numSigLiftedChannels,
            numSigEncodingLayers=self.numSigEncodingLayers,
            numExpandedSignals=self.numExpandedSignals,
            waveletType=self.signalEncoderWaveletType,
            numEncodedSignals=self.numEncodedSignals,
            signalMinMaxScale=self.signalMinMaxScale,
            debuggingResults=self.debuggingResults,
            useFinalParams=self.useFinalParams,
            sequenceBounds=self.sequenceBounds,
            maxNumSignals=self.maxNumSignals,
            accelerator=self.accelerator,
        )

        if submodel == modelConstants.signalEncoderModel: return None

        # The autoencoder model reduces the incoming signal's dimension.
        self.autoencoderModel = autoencoderModel(
            compressionFactor=self.compressionFactor,
            compressedLength=self.compressedLength,
            debuggingResults=self.debuggingResults,
            expansionFactor=self.expansionFactor,
            useFinalParams=self.useFinalParams,
            accelerator=self.accelerator,
        )

        if submodel == modelConstants.autoencoderModel: return None

        # -------------------- Final Emotion Prediction -------------------- #

        # The manifold projection model maps each signal to a common dimension.
        self.signalMappingModel = signalMappingModel(
            numEncodedSignals=self.numEncodedSignals,
            compressedLength=self.compressedLength,
            featureNames=self.featureNames,
        )

        self.specificEmotionModel = specificEmotionModel(
            numInterpreterHeads=self.numInterpreterHeads,
            numActivityFeatures=self.numCommonSignals,
            numCommonSignals=self.numCommonSignals,
            numBasicEmotions=self.numBasicEmotions,
            activityNames=self.activityNames,
            emotionNames=self.emotionNames,
            featureNames=self.featureNames,
            numSubjects=self.numSubjects,
        )

        self.sharedEmotionModel = sharedEmotionModel(
            numActivityFeatures=self.numCommonSignals,
            numInterpreterHeads=self.numInterpreterHeads,
            numCommonSignals=self.numCommonSignals,
            numBasicEmotions=self.numBasicEmotions,
            numEncodedSignals=self.numEncodedSignals,
            compressedLength=self.compressedLength,
        )

    # ---------------------------------------------------------------------- #  
    # -------------------------- Model Components -------------------------- #

    def signalEncoding(self, signalData, startTimeIndices, signalIdentifiers, metadata, decodeSignals=True, calculateLoss=True, trainingFlag=False):
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        signalData, signalIdentifiers, metadata = signalData.to(self.device), signalIdentifiers.to(self.device), metadata.to(self.device)

        t1 = time.time()
        # Forward pass through the signal encoder to reduce to a common signal number.
        encodedData, reconstructedData, predictedIndexProbabilities, decodedPredictedIndexProbabilities, signalEncodingLayerLoss = self.signalEncoderModel(signalData, startTimeIndices, signalIdentifiers, metadata, decodeSignals, calculateLoss, trainingFlag)
        # decodedPredictedIndexProbabilities dimension: batchSize, numSignals
        # encodedData dimension: batchSize, numEncodedSignals, finalDistributionLength
        # reconstructedData dimension: batchSize, numSignals, finalDistributionLength
        # predictedIndexProbabilities dimension: batchSize, numSignals
        # signalEncodingLayerLoss dimension: batchSize
        t2 = time.time(); print("\tSignal Encoder:", t2 - t1)

        return encodedData, reconstructedData, predictedIndexProbabilities, decodedPredictedIndexProbabilities, signalEncodingLayerLoss

    def compressData(self, signalData, initialSignalData, reconstructSignals=True, calculateLoss=True, compileVariables=False, compileLosses=False, fullReconstruction=False, trainingFlag=False):
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        signalData, initialSignalData = signalData.to(self.device), initialSignalData.to(self.device)
        denoisedDoubleReconstructedData = 0

        with torch.no_grad():
            # Compile the variables from signal encoding.
            encodedData, reconstructedData, predictedIndexProbabilities, decodedPredictedIndexProbabilities, signalEncodingLayerLoss = self.signalEncoding(signalData, startTimeIndices, signalIdentifiers, metadata, decodeSignals=compileVariables, calculateLoss=calculateLoss and compileLosses, trainingFlag=False)
            signalEncodingOutputs = encodedData, reconstructedData, predictedIndexProbabilities, decodedPredictedIndexProbabilities, signalEncodingLayerLoss

        t1 = time.time()
        # Forward pass through the autoencoder for data compression.
        compressedData, reconstructedEncodedData, autoencoderLayerLoss = self.autoencoderModel(encodedData, reconstructSignals, calculateLoss, trainingFlag)
        # compressedData dimension: batchSize, numSignals, compressedLength
        # reconstructedEncodedData dimension: batchSize, numSignals, finalDistributionLength
        # signalChannel dimension: batchSize, numSignals, finalDistributionLength
        # autoencoderLayerLoss dimension: batchSize
        t2 = time.time(); print("\tAutoencoder:", t2 - t1)

        if fullReconstruction:
            # Denoise the final signals.
            numSignalForwardPath = self.signalEncoderModel.encodeSignals.simulateSignalPath(initialSignalData.size(1), encodedData.size(1))[0]
            doubleReconstructedData = self.signalEncoderModel.reconstructEncodedData(reconstructedEncodedData, numSignalForwardPath, signalEncodingLayerLoss=None, calculateLoss=False)[2]
            denoisedDoubleReconstructedData = self.autoencoderModel.generalAutoencoder.applyDenoiserLast(doubleReconstructedData)
            # denoisedDoubleReconstructedData dimension: batchSize, numSignals, finalDistributionLength

        return signalEncodingOutputs, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss

    # ----------------------- Emotion Classification ----------------------- #  

    def mapSignals(self, signalData, initialSignalData, remapSignals=True, compileVariables=False, trainingFlag=False):
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        signalData, initialSignalData = signalData.to(self.device), initialSignalData.to(self.device)

        with torch.no_grad():
            # Compile the variables from autoencoder.
            signalEncodingOutputs, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss = self.compressData(signalData, initialSignalData, reconstructSignals=compileVariables, calculateLoss=compileVariables, compileVariables=compileVariables, compileLosses=compileVariables, fullReconstruction=False, trainingFlag=False)
            autoencodingOutputs = compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss

        t1 = time.time()
        # Forward pass through the manifold projector to find a common manifold space.
        mappedSignalData, reconstructedCompressedData = self.signalMappingModel(compressedData, remapSignals, trainingFlag)
        # reconstructedCompressedData dimension: batchSize, numEncodedSignals, compressedLength
        # mappedSignalData dimension: batchSize, numEncodedSignals, compressedLength
        t2 = time.time(); print("\tManifold Projection:", t2 - t1)

        return signalEncodingOutputs, autoencodingOutputs, mappedSignalData, reconstructedCompressedData

    def emotionPrediction(self, signalData, initialSignalData, metadata, remapSignals=True, compileVariables=False, trainingFlag=False):
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        signalData, metadata, initialSignalData = signalData.to(self.device), metadata.to(self.device), initialSignalData.to(self.device)

        # Compile the manifold projection data.
        signalEncodingOutputs, autoencodingOutputs, mappedSignalData, reconstructedCompressedData = self.mapSignals(signalData, initialSignalData, remapSignals, compileVariables, trainingFlag=False)

        t1 = time.time()
        # Forward pass through the emotion model to predict complex emotions.
        featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions = self.sharedEmotionModel(mappedSignalData, metadata, self.specificEmotionModel, trainingFlag)
        t2 = time.time(); print("\tEmotion Prediction:", t2 - t1)

        return signalEncodingOutputs, autoencodingOutputs, mappedSignalData, reconstructedCompressedData, featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions

    # ------------------------- Full Forward Calls ------------------------- #  

    def forward(self, signalData, metadata, initialSignalData, reconstructSignals=True, compileVariables=False, submodel=None, trainingFlag=False):
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        signalData, metadata, initialSignalData = signalData.to(self.device), metadata.to(self.device), initialSignalData.to(self.device)

        # Initialize the output tensors.
        autoencodingOutputs = (torch.tensor(data=0, device=self.device) for _ in range(4))
        emotionModelOutputs = (torch.tensor(data=0, device=self.device) for _ in range(6))

        if submodel == modelConstants.signalEncoderModel:
            with self.accelerator.autocast():
                # Only look at the signal encoder.
                encodedData, reconstructedData, predictedIndexProbabilities, decodedPredictedIndexProbabilities, signalEncodingLayerLoss = self.signalEncoding(signalData, startTimeIndices, signalIdentifiers, metadata, decodeSignals=reconstructSignals, calculateLoss=compileVariables, trainingFlag=trainingFlag)
                signalEncodingOutputs = encodedData.to('cpu'), reconstructedData.to('cpu'), predictedIndexProbabilities.to('cpu'), decodedPredictedIndexProbabilities.to('cpu'), signalEncodingLayerLoss.to('cpu')

        elif submodel == modelConstants.autoencoderModel:
            # Only look at the autoencoder.
            signalEncodingOutputs, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss \
                = self.compressData(signalData, initialSignalData, reconstructSignals=reconstructSignals, calculateLoss=compileVariables, compileVariables=compileVariables, compileLosses=compileVariables, fullReconstruction=True, trainingFlag=trainingFlag)
            autoencodingOutputs = compressedData.to('cpu'), reconstructedEncodedData.to('cpu'), denoisedDoubleReconstructedData.to('cpu'), autoencoderLayerLoss.to('cpu')

        else:
            # Analyze the full model.
            signalEncodingOutputs, autoencodingOutputs, mappedSignalData, reconstructedCompressedData, featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions = self.emotionPrediction(signalData, metadata, compileVariables, compileVariables)
            emotionModelOutputs = mappedSignalData.to('cpu'), reconstructedCompressedData.to('cpu'), mappedSignalData.to('cpu'), featureData.to('cpu'), activityDistribution.to('cpu'), eachBasicEmotionDistribution.to('cpu'), finalEmotionDistributions.to('cpu')

        return signalEncodingOutputs, autoencodingOutputs, emotionModelOutputs

    def fullDataPass(self, submodel, dataLoader, timeWindow, reconstructSignals=True, compileVariables=False, trainingFlag=False):
        # Initialize variables
        batch_size, numSignals, _ = dataLoader.dataset.features.size()

        # Preallocate tensors for the signal encoder.
        decodedPredictedIndexProbabilities = torch.zeros((batch_size, numSignals), device=dataLoader.dataset.features.device)
        predictedIndexProbabilities = torch.zeros((batch_size, numSignals), device=dataLoader.dataset.features.device)
        encodedData = torch.zeros((batch_size, self.numEncodedSignals, timeWindow), device=dataLoader.dataset.features.device)
        reconstructedData = torch.zeros((batch_size, numSignals, timeWindow), device=dataLoader.dataset.features.device)
        # Preallocate tensors for the autoencoder.
        compressedData = torch.zeros((batch_size, self.numEncodedSignals, self.compressedLength), device=dataLoader.dataset.features.device)
        reconstructedEncodedData = torch.zeros((batch_size, self.numEncodedSignals, timeWindow), device=dataLoader.dataset.features.device)
        denoisedDoubleReconstructedData = torch.zeros((batch_size, numSignals, timeWindow), device=dataLoader.dataset.features.device)
        # Preallocate tensors for the manifold projection.
        mappedSignalData = torch.zeros((batch_size, self.numEncodedSignals, self.compressedLength), device=dataLoader.dataset.features.device)
        reconstructedCompressedData = torch.zeros((batch_size, self.numEncodedSignals, self.compressedLength), device=dataLoader.dataset.features.device)
        # Preallocate tensors for emotion and activity prediction.
        featureData = torch.zeros((batch_size, self.numCommonSignals), device=dataLoader.dataset.features.device)
        activityDistribution = torch.zeros((batch_size, self.numActivities), device=dataLoader.dataset.features.device)
        eachBasicEmotionDistribution = torch.zeros((batch_size, self.numInterpreterHeads, self.numBasicEmotions, self.compressedLength), device=dataLoader.dataset.features.device)
        finalEmotionDistributions = torch.zeros((self.numEmotions, batch_size, self.compressedLength), device=dataLoader.dataset.features.device)

        # Set up the loss calculation.
        signalEncodingLayerLoss = torch.zeros(batch_size, device=dataLoader.dataset.features.device)
        autoencoderLayerLoss = torch.zeros(batch_size, device=dataLoader.dataset.features.device)

        startIdx = 0
        # For each minibatch.
        for batchIdx, data in enumerate(dataLoader):
            # Extract the data, labels, and testing/training indices.
            batchData, trueBatchLabels, batchTrainingMask, batchTestingMask = data
            endIdx = startIdx + batchData.size(0)

            # Separate out the data.
            allSignalData, allSignalIdentifiers, allMetadata = emotionDataInterface.separateData(batchData)
            segmentedSignalData = dataAugmentation.getRecentSignalPoints(allSignalData, timeWindow)  # Segment the data into its time window.

            # Forward pass for the current batch
            signalEncodingOutputs, autoencodingOutputs, emotionModelOutputs = self.forward(segmentedSignalData, allMetadata, segmentedSignalData, reconstructSignals, compileVariables, submodel, trainingFlag=trainingFlag)
            # Unpack all the parameters.
            encodedBatchData, reconstructedBatchData, batchPredictedIndexProbabilities, batchDecodedPredictedIndexProbabilities, batchSignalEncodingLayerLoss = signalEncodingOutputs
            compressedBatchData, reconstructedEncodedBatchData, batchDenoisedDoubleReconstructedData, batchAutoencoderLayerLoss = autoencodingOutputs
            mappedBatchData, reconstructedCompressedBatchData, featureBatchData, batchActivityDistribution, batchBasicEmotionDistribution, batchEmotionDistributions = emotionModelOutputs

            # Signal encoding: assign the results to the preallocated tensors
            encodedData[startIdx:endIdx] = encodedBatchData
            reconstructedData[startIdx:endIdx] = reconstructedBatchData
            predictedIndexProbabilities[startIdx:endIdx] = batchPredictedIndexProbabilities
            decodedPredictedIndexProbabilities[startIdx:endIdx] = batchDecodedPredictedIndexProbabilities
            # Autoencoder: assign the results to the preallocated tensors
            compressedData[startIdx:endIdx] = compressedBatchData
            reconstructedEncodedData[startIdx:endIdx] = reconstructedEncodedBatchData
            denoisedDoubleReconstructedData[startIdx:endIdx] = batchDenoisedDoubleReconstructedData
            # Signal mapping: assign the results to the preallocated tensors
            mappedSignalData[startIdx:endIdx] = mappedBatchData
            reconstructedCompressedData[startIdx:endIdx] = reconstructedCompressedBatchData

            # Emotion/activity model: assign the results to the preallocated tensors
            featureData[startIdx:endIdx] = featureBatchData
            activityDistribution[startIdx:endIdx] = batchActivityDistribution
            eachBasicEmotionDistribution[startIdx:endIdx] = batchBasicEmotionDistribution
            finalEmotionDistributions[:, startIdx:endIdx, :] = batchEmotionDistributions

            # Add up the losses.
            signalEncodingLayerLoss[startIdx:endIdx] = batchSignalEncodingLayerLoss
            autoencoderLayerLoss[startIdx:endIdx] = batchAutoencoderLayerLoss

            # Wrap up the loop.
            startIdx = endIdx

        # Repack the data into the expected format.
        signalEncodingOutputs = encodedData, reconstructedData, predictedIndexProbabilities, decodedPredictedIndexProbabilities, signalEncodingLayerLoss
        autoencodingOutputs = compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss
        emotionModelOutputs = mappedSignalData, reconstructedCompressedData, featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions

        return signalEncodingOutputs, autoencodingOutputs, emotionModelOutputs

    # --------------------------- SHAP Analysis ---------------------------- #  

    # def shapInterface(self, reshapedSignalFeatures):
    #     # Extract the incoming data's dimension and ensure a proper data format.
    #     batchSize, numFeatures = reshapedSignalFeatures.shape
    #     reshapedSignalFeatures = torch.tensor(reshapedSignalFeatures.tolist(), device=reshapedSignalFeatures.device)
    #     assert numFeatures == self.numSignals * self.compressedLength, f"{numFeatures} {self.numSignals} {self.compressedLength}"
    #
    #     # Reshape the inputs to integrate into the model's expected format.
    #     signalFeatures = reshapedSignalFeatures.view((batchSize, self.numSignals, self.compressedLength))
    #
    #     # predict the activities.
    #     activityDistribution = self.forward(signalFeatures)
    #
    #     return activityDistribution.detach().cpu().numpy()
