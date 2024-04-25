# General
import time

# PyTorch
import torch

# Import submodels
from .emotionModelHelpers.submodels.specificEmotionModel import specificEmotionModel
from .emotionModelHelpers.submodels.trainingInformation import trainingInformation
from .emotionModelHelpers.submodels.sharedEmotionModel import sharedEmotionModel
from .emotionModelHelpers.submodels.signalMappingModel import signalMappingModel
from .emotionModelHelpers.submodels.signalEncoderModel import signalEncoderModel  # A signal encoder pipeline to make a universal feature vector.
from .emotionModelHelpers.submodels.autoencoderModel import autoencoderModel  # An autoencoder pipeline for compressing individual signals.

# Import helper modules
from .emotionModelHelpers.emotionDataInterface import emotionDataInterface
from .._globalPytorchModel import globalModel


class emotionModelHead(globalModel):
    def __init__(self, submodel, accelerator, sequenceLength, maxNumSignals, numSubjectIdentifiers, demographicLength, userInputParams,
                 emotionNames, activityNames, featureNames, numSubjects, datasetName, metaTraining=False):
        super(emotionModelHead, self).__init__()
        # General model parameters.
        self.numSubjectIdentifiers = numSubjectIdentifiers
        self.demographicLength = demographicLength  # The amount of demographic information (age, weight, etc.). Subject index is not included.
        self.numActivities = len(activityNames)  # The number of activities to predict.
        self.numEmotions = len(emotionNames)  # The number of emotions to predict.
        self.sequenceLength = sequenceLength  # The length of each incoming signal: features used in the model.
        self.numSignals = len(featureNames)  # The number of signals going into the model.
        self.maxNumSignals = maxNumSignals  # The maximum number of signals to consider.
        self.activityNames = activityNames  # The names of each activity we are predicting. Dim: numActivities
        self.featureNames = featureNames  # The names of each feature/signal in the model. Dim: numSignals
        self.metaTraining = metaTraining  # A flag representing if this is meta-training data.
        self.emotionNames = emotionNames  # The names of each emotion we are predicting. Dim: numEmotions
        self.device = accelerator.device
        self.numSubjects = numSubjects  # The maximum number of subjects the model is training on.
        self.accelerator = accelerator  # Hugging face model optimizations.
        self.datasetName = datasetName

        # Signal encoder parameters.
        self.numExpandedSignals = userInputParams['numExpandedSignals']     # The number of signals to group when you begin compression or finish expansion.
        self.numEncodingLayers = userInputParams['numEncodingLayers']       # The number of transformer layers during signal encoding.
        self.numLiftedChannels = userInputParams['numLiftedChannels']       # The number of channels to lift the signal to.
        # Autoencoder parameters.
        self.compressionFactor = userInputParams['compressionFactor']       # The expansion factor of the autoencoder
        self.expansionFactor = userInputParams['expansionFactor']           # The expansion factor of the autoencoder
        # Emotion parameters.
        self.numInterpreterHeads = userInputParams['numInterpreterHeads']   # The number of ways to interpret a set of physiological signals.
        self.numBasicEmotions = userInputParams['numBasicEmotions']         # The number of basic emotions (basis states of emotions).

        # Tunable encoding parameters.
        self.numEncodedSignals = 32  # The final number of signals to accept, encoding all signal information.
        self.compressedLength = 32  # The final length of the compressed signal after the autoencoder.
        # Feature parameters (code changes required if you change these!!!)
        self.numCommonSignals = 8    # The number of features from considering all the signals.
        self.numEmotionSignals = 8   # The number of common activity features to extract.
        self.numActivitySignals = 8  # The number of common activity features to extract.

        # Specify the parameters for the time analysis.
        self.timeWindows = [90, 120, 150, 180, 210, 240]  # A list of all time windows to consider for the encoding.
        self.sequenceBounds = (self.timeWindows[0], self.timeWindows[-1])  # The minimum and maximum sequence length for the model.

        # Setup holder for the model's training information
        self.trainingInformation = trainingInformation()

        # Initialize all the models.
        self.signalEncoderModel = None
        self.autoencoderModel = None
        self.signalMappingModel = None
        self.specificEmotionModel = None
        self.sharedEmotionModel = None

        # Populate the current models.
        self.initializeSubmodels(submodel)

    def initializeSubmodels(self, submodel):

        # ------------------------ Data Compression ------------------------ # 

        # The signal encoder model to find a common feature vector across all signals.
        self.signalEncoderModel = signalEncoderModel(
            numExpandedSignals=self.numExpandedSignals,
            numEncodedSignals=self.numEncodedSignals,
            numEncodingLayers=self.numEncodingLayers,
            numLiftedChannels=self.numLiftedChannels,
            sequenceBounds=self.sequenceBounds,
            maxNumSignals=self.maxNumSignals,
            timeWindows=self.timeWindows,
            accelerator=self.accelerator,
        )

        if submodel == "signalEncoder": return None

        # The autoencoder model reduces the incoming signal's dimension.
        self.autoencoderModel = autoencoderModel(
            compressionFactor=self.compressionFactor,
            compressedLength=self.compressedLength,
            expansionFactor=self.expansionFactor,
            timeWindows=self.timeWindows,
            accelerator=self.accelerator,
        )

        if submodel == "autoencoder": return None

        # -------------------- Final Emotion Prediction -------------------- #

        # The manifold projection model maps each signal to a common dimension.
        self.signalMappingModel = signalMappingModel(
            numEncodedSignals=self.numEncodedSignals,
            compressedLength=self.compressedLength,
            featureNames=self.featureNames,
            timeWindows=self.timeWindows,
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

    def signalEncoding(self, signalData, initialSignalData, decodeSignals=True, calculateLoss=True, trainingFlag=False):
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        signalData, initialSignalData = signalData.to(self.device), initialSignalData.to(self.device)

        t1 = time.time()
        # Forward pass through the signal encoder to reduce to a common signal number.
        encodedData, reconstructedData, signalEncodingLayerLoss = self.signalEncoderModel(signalData, initialSignalData, decodeSignals, calculateLoss, trainingFlag)
        # encodedData dimension: batchSize, numEncodedSignals, sequenceLength
        # reconstructedData dimension: batchSize, numSignals, sequenceLength
        # signalEncodingLayerLoss dimension: batchSize
        t2 = time.time()
        print("\tSignal Encoder:", t2 - t1)

        return encodedData, reconstructedData, signalEncodingLayerLoss

    def compressData(self, signalData, initialSignalData, reconstructSignals=True, calculateLoss=True, compileVariables=False, compileLosses=False, fullReconstruction=False, trainingFlag=False):
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        signalData, initialSignalData = signalData.to(self.device), initialSignalData.to(self.device)
        denoisedDoubleReconstructedData = 0

        with torch.no_grad():
            # Compile the variables from signal encoding.
            encodedData, reconstructedData, signalEncodingLayerLoss = self.signalEncoding(signalData, initialSignalData, decodeSignals=compileVariables, calculateLoss=calculateLoss and compileLosses, trainingFlag=False)

        t1 = time.time()
        # Forward pass through the autoencoder for data compression.
        compressedData, reconstructedEncodedData, autoencoderLayerLoss = self.autoencoderModel(encodedData, reconstructSignals, calculateLoss, trainingFlag)
        # compressedData dimension: batchSize, numSignals, compressedLength
        # reconstructedEncodedData dimension: batchSize, numSignals, sequenceLength
        # signalData dimension: batchSize, numSignals, sequenceLength
        # autoencoderLayerLoss dimension: batchSize
        t2 = time.time()
        print("\tAutoencoder:", t2 - t1)

        if fullReconstruction:
            # Denoise the final signals.
            numSignalForwardPath = self.signalEncoderModel.encodeSignals.simulateSignalPath(initialSignalData.size(1), encodedData.size(1))[0]
            doubleReconstructedData = self.signalEncoderModel.reconstructEncodedData(reconstructedEncodedData, numSignalForwardPath, signalEncodingLayerLoss=None, calculateLoss=False, trainingFlag=False)[3]
            denoisedDoubleReconstructedData = self.autoencoderModel.generalAutoencoder.applyDenoiserLast(doubleReconstructedData)
            # denoisedDoubleReconstructedData dimension: batchSize, numSignals, sequenceLength

        return encodedData, reconstructedData, signalEncodingLayerLoss, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss

    # ----------------------- Emotion Classification ----------------------- #  

    def mapSignals(self, signalData, initialSignalData, remapSignals=True, compileVariables=False, trainingFlag=False):
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        signalData, initialSignalData = signalData.to(self.device), initialSignalData.to(self.device)

        with torch.no_grad():
            # Compile the variables from autoencoder.
            encodedData, reconstructedData, signalEncodingLayerLoss, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss \
                = self.compressData(signalData, initialSignalData, reconstructSignals=compileVariables, calculateLoss=compileVariables, compileVariables=compileVariables, compileLosses=compileVariables, fullReconstruction=False, trainingFlag=False)

        t1 = time.time()
        # Forward pass through the manifold projector to find a common manifold space.
        mappedSignalData, reconstructedCompressedData = self.signalMappingModel(compressedData, remapSignals, trainingFlag)
        # reconstructedCompressedData dimension: batchSize, numEncodedSignals, compressedLength
        # mappedSignalData dimension: batchSize, numEncodedSignals, compressedLength
        t2 = time.time()
        print("\tManifold Projection:", t2 - t1)

        return encodedData, reconstructedData, signalEncodingLayerLoss, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss, mappedSignalData, reconstructedCompressedData

    def emotionPrediction(self, signalData, initialSignalData, subjectIdentifiers, remapSignals=True, compileVariables=False, trainingFlag=False):
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        signalData, subjectIdentifiers, initialSignalData = signalData.to(self.device), subjectIdentifiers.to(self.device), initialSignalData.to(self.device)

        # Compile the manifold projection data.
        encodedData, reconstructedData, signalEncodingLayerLoss, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss, mappedSignalData, \
            reconstructedCompressedData = self.mapSignals(signalData, initialSignalData, remapSignals, compileVariables, trainingFlag=False)

        t1 = time.time()
        # Forward pass through the emotion model to predict complex emotions.
        featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions = self.sharedEmotionModel(mappedSignalData, subjectIdentifiers, self.specificEmotionModel, trainingFlag)
        t2 = time.time()
        print("\tEmotion Prediction:", t2 - t1)

        return encodedData, reconstructedData, signalEncodingLayerLoss, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss, mappedSignalData, \
            reconstructedCompressedData, featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions

    # ------------------------- Full Forward Calls ------------------------- #  

    def forward(self, signalData, subjectIdentifiers, initialSignalData, compileVariables=False, submodel=None, trainingFlag=False):
        # Add the data, labels, and training/testing indices to the device (GPU/CPU)
        signalData, subjectIdentifiers, initialSignalData = signalData.to(self.device), subjectIdentifiers.to(self.device), initialSignalData.to(self.device)

        encodedData, reconstructedData, signalEncodingLayerLoss, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss, mappedSignalData, reconstructedCompressedData, \
            featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions = (torch.tensor(0, device=self.device) for _ in range(13))

        if submodel == "signalEncoder":
            with self.accelerator.autocast():
                # Only look at the signal encoder.
                encodedData, reconstructedData, signalEncodingLayerLoss = self.signalEncoding(signalData, initialSignalData, decodeSignals=compileVariables, calculateLoss=compileVariables, trainingFlag=trainingFlag)

        elif submodel == "autoencoder":
            # Only look at the autoencoder.
            encodedData, reconstructedData, signalEncodingLayerLoss, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss \
                = self.compressData(signalData, initialSignalData, reconstructSignals=compileVariables, calculateLoss=compileVariables, compileVariables=compileVariables, compileLosses=compileVariables, fullReconstruction=True, trainingFlag=trainingFlag)

        else:
            # Analyze the full model.
            encodedData, reconstructedData, signalEncodingLayerLoss, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss, mappedSignalData, \
                reconstructedCompressedData, featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions \
                = self.emotionPrediction(signalData, subjectIdentifiers, compileVariables, compileVariables)

        return encodedData.to('cpu'), reconstructedData.to('cpu'), signalEncodingLayerLoss.to('cpu'), compressedData.to('cpu'), reconstructedEncodedData.to('cpu'), denoisedDoubleReconstructedData.to('cpu'), autoencoderLayerLoss.to('cpu'), \
            mappedSignalData.to('cpu'), reconstructedCompressedData.to('cpu'), featureData.to('cpu'), activityDistribution.to('cpu'), eachBasicEmotionDistribution.to('cpu'), finalEmotionDistributions.to('cpu')

    def fullDataPass(self, submodel, dataLoader, timeWindow, compileVariables=False, trainingFlag=False):
        # Initialize variables
        batch_size, numSignals, _ = dataLoader.dataset.features.size()

        # Preallocate tensors for the signal encoder.
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
            allSignalData, allDemographicData, allSubjectIdentifiers = emotionDataInterface.separateData(batchData, self.sequenceLength, self.numSubjectIdentifiers, self.demographicLength)
            segmentedSignalData = emotionDataInterface.getRecentSignalPoints(allSignalData, timeWindow)  # Segment the data into its time window.

            # Forward pass for the current batch
            encodedBatchData, reconstructedBatchData, batchSignalEncodingLayerLoss, compressedBatchData, reconstructedEncodedBatchData, batchDenoisedDoubleReconstructedData, batchAutoencoderLayerLoss, \
                mappedBatchData, reconstructedCompressedBatchData, featureBatchData, batchActivityDistribution, batchBasicEmotionDistribution, batchEmotionDistributions \
                = self.forward(segmentedSignalData, allSubjectIdentifiers, segmentedSignalData, compileVariables, submodel, trainingFlag=trainingFlag)

            # Signal encoding: assign the results to the preallocated tensors
            encodedData[startIdx:endIdx] = encodedBatchData
            reconstructedData[startIdx:endIdx] = reconstructedBatchData
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

        # Return preallocated tensors
        return encodedData, reconstructedData, signalEncodingLayerLoss, compressedData, reconstructedEncodedData, denoisedDoubleReconstructedData, autoencoderLayerLoss, mappedSignalData, \
            reconstructedCompressedData, featureData, activityDistribution, eachBasicEmotionDistribution, finalEmotionDistributions

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
