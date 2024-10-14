# General
import matplotlib.pyplot as plt
import matplotlib
import torch
import os

# Import files for machine learning
from ..emotionDataInterface import emotionDataInterface
from ..generalMethods.dataAugmentation import dataAugmentation
from ..modelConstants import modelConstants

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from ._signalEncoderVisualizations import signalEncoderVisualizations
from ._autoencoderVisualizations import autoencoderVisualizations
from ._generalVisualizations import generalVisualizations


class modelVisualizations(globalPlottingProtocols):

    def __init__(self, accelerator, generalTimeWindow, modelSubfolder):
        super(modelVisualizations, self).__init__()
        # General parameters.
        self.generalTimeWindow = generalTimeWindow
        self.accelerator = accelerator
        self.saveDataFolder = None

        # Plotting settings.
        plt.ion()

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
        saveDataFolder = os.path.normpath(os.path.dirname(__file__) + f"/../../../dataAnalysis/{modelSubfolder}") + "/"
        self.saveDataFolder = os.path.relpath(os.path.normpath(saveDataFolder), os.getcwd()) + "/"
        # Make the folder to save the data.
        os.makedirs(self.saveDataFolder, exist_ok=True)

        # Initialize visualization protocols.
        self.generalViz.setSavingFolder(self.saveDataFolder)
        self.autoencoderViz.setSavingFolder(self.saveDataFolder)
        self.signalEncoderViz.setSavingFolder(self.saveDataFolder)

    # ---------------------------------------------------------------------- #

    def plotDatasetComparison(self, submodel, allModelPipelines, trainingDate, fastPass):
        self.accelerator.print(f"\nCalculating loss for model comparison", flush=True)
        fastPass = True

        # Prepare the model/data for evaluation.
        self.setSavingFolder(f"trainingFigures/{submodel}/{trainingDate}/modelComparison/")  # Label the correct folder to save this analysis.

        # Plot the loss on the primary GPU.
        if self.accelerator.is_local_main_process:

            if submodel in [modelConstants.signalEncoderModel, modelConstants.autoencoderModel]:
                if submodel == modelConstants.signalEncoderModel:
                    specificModels = [modelPipeline.model.specificSignalEncoderModel for modelPipeline in allModelPipelines]
                else:
                    specificModels = [modelPipeline.model.autoencoderModel for modelPipeline in allModelPipelines]

                # For every time window.
                for timeWindowInd in range(len(modelConstants.timeWindows)):
                    timeWindow = modelConstants.timeWindows[timeWindowInd]

                    # If we are testing, only plot one time.
                    if fastPass and timeWindow != self.generalTimeWindow: continue

                    # Plot reconstruction loss.
                    self.generalViz.plotTrainingLosses([specificModel.trainingLosses_timeReconstructionAnalysis[timeWindowInd] for specificModel in specificModels],
                                                       [specificModel.testingLosses_timeReconstructionAnalysis[timeWindowInd] for specificModel in specificModels],
                                                       lossLabels=[f"{modelPipeline.model.datasetName} Signal Encoding Reconstruction Loss"[timeWindowInd] for modelPipeline in allModelPipelines],
                                                       plotTitle="trainingLosses/All Signal Encoder Convergence Losses")

                    # Plot mean loss.
                    # self.generalViz.plotTrainingLosses([specificModel.trainingLosses_timeMeanAnalysis[timeWindowInd] for specificModel in specificModels],
                    #                                    [specificModel.testingLosses_timeMeanAnalysis[timeWindowInd] for specificModel in specificModels],
                    #                                    lossLabels=[f"{modelPipeline.model.datasetName} Signal Encoding Mean Loss"[timeWindowInd] for modelPipeline in allModelPipelines],
                    #                                    plotTitle="trainingLosses/All Signal Encoder Mean Losses")
                    #
                    # # Plot MinMax loss.
                    # self.generalViz.plotTrainingLosses([specificModel.trainingLosses_timeMinMaxAnalysis[timeWindowInd] for specificModel in specificModels],
                    #                                    [specificModel.testingLosses_timeMinMaxAnalysis[timeWindowInd] for specificModel in specificModels],
                    #                                    lossLabels=[f"{modelPipeline.model.datasetName} Signal Encoding MinMax Loss"[timeWindowInd] for modelPipeline in allModelPipelines],
                    #                                    plotTitle="trainingLosses/All Signal Encoder MinMax Losses")

    def plotAllTrainingEvents(self, submodel, modelPipeline, lossDataLoader, trainingDate, currentEpoch, fastPass):
        self.accelerator.print(f"\nCalculating loss for {modelPipeline.model.datasetName} model", flush=True)
        fastPass = True

        # Prepare the model/data for evaluation.
        modelPipeline.setupTrainingFlags(modelPipeline.model, trainingFlag=False)  # Set all models into evaluation mode.
        model = modelPipeline.model

        # Load in all the data and labels for final predictions.
        allData, allLabels, allTrainingMasks, allTestingMasks = lossDataLoader.dataset.getAll()
        allSignalData, allSignalIdentifiers, allMetadata = self.dataInterface.separateData(allData)
        reconstructionIndex = self.dataInterface.getReconstructionIndex(allTrainingMasks)
        assert reconstructionIndex is not None

        with torch.no_grad():
            # ---------------------- Time-Specific Plots ----------------------- #

            # For each time window we are analyzing.
            for timeAnalysisInd in range(len(modelConstants.timeWindows)):
                print(f"\tPlotting figures for the {modelConstants.timeWindows[timeAnalysisInd]} second time window")
                timeWindow = modelConstants.timeWindows[timeAnalysisInd]

                # If we are testing, only plot one time.
                if fastPass and timeWindow != modelConstants.timeWindows[5]: continue

                # Go through all the plots at this specific time window.
                self.plotTrainingEvent(model, currentEpoch, allSignalData, allMetadata, allTrainingMasks, allTestingMasks, reconstructionIndex, submodel, trainingDate, timeWindow, model.datasetName)

            # ---------------------- Time-Agnostic Plots ----------------------- #

            # Prepare the model/data for evaluation.
            self.setSavingFolder(f"trainingFigures/{submodel}/{trainingDate}/{model.datasetName}/")  # Label the correct folder to save this analysis.

            # Plot the loss on the primary GPU.
            if self.accelerator.is_local_main_process:

                if submodel == modelConstants.signalEncoderModel:
                    # Plot autoencoder loss for each time.
                    self.generalViz.plotTrainingLosses(model.specificSignalEncoderModel.trainingLosses_timeReconstructionAnalysis, model.specificSignalEncoderModel.testingLosses_timeReconstructionAnalysis, lossLabels=modelConstants.timeWindows, plotTitle="trainingLosses/Signal Encoder Convergence Loss")
                    self.generalViz.plotTrainingLosses(model.specificSignalEncoderModel.trainingLosses_timePosEncAnalysis, model.specificSignalEncoderModel.testingLosses_timePosEncAnalysis, lossLabels=modelConstants.timeWindows, plotTitle="trainingLosses/Positional Encoder Convergence Loss")
                    self.generalViz.plotTrainingLosses(model.specificSignalEncoderModel.trainingLosses_timeMeanAnalysis, model.specificSignalEncoderModel.testingLosses_timeMeanAnalysis, lossLabels=modelConstants.timeWindows, plotTitle="trainingLosses/Signal Encoder Mean Loss")
                    self.generalViz.plotTrainingLosses(model.specificSignalEncoderModel.trainingLosses_timeMinMaxAnalysis, model.specificSignalEncoderModel.testingLosses_timeMinMaxAnalysis, lossLabels=modelConstants.timeWindows, plotTitle="trainingLosses/Signal Encoder MinMax Loss")

                if submodel == modelConstants.autoencoderModel:
                    # Plot autoencoder loss for each time.
                    self.generalViz.plotTrainingLosses(model.autoencoderModel.trainingLosses_timeReconstructionAnalysis, model.autoencoderModel.testingLosses_timeReconstructionAnalysis, lossLabels=modelConstants.timeWindows, plotTitle="trainingLosses/Autoencoder Convergence Loss")
                    self.generalViz.plotTrainingLosses(model.autoencoderModel.trainingLosses_timeMeanAnalysis, model.autoencoderModel.testingLosses_timeMeanAnalysis, lossLabels=modelConstants.timeWindows, plotTitle="trainingLosses/Autoencoder Mean Loss")
                    self.generalViz.plotTrainingLosses(model.autoencoderModel.trainingLosses_timeMinMaxAnalysis, model.autoencoderModel.testingLosses_timeMinMaxAnalysis, lossLabels=modelConstants.timeWindows, plotTitle="trainingLosses/Autoencoder MinMax Loss")

            # Wait before continuing.
            self.accelerator.wait_for_everyone()

        # ------------------------------------------------------------------ #

    def plotTrainingEvent(self, model, currentEpoch, allSignalData, allMetadata, allTrainingMasks, allTestingMasks, reconstructionIndex, submodel, trainingDate, timeWindow, datasetName):
        # Prepare the model/data for evaluation.
        self.setSavingFolder(f"trainingFigures/{submodel}/{trainingDate}/{datasetName}/{timeWindow}/")  # Label the correct folder to save this analysis.
        model.eval()

        # General plotting parameters.
        _, numSignals, _ = allSignalData.shape

        # Augment the time axis.
        segmentedSignalData = dataAugmentation.getRecentSignalPoints(allSignalData, timeWindow)

        # ---------------------- Get the Data to Plot ---------------------- # 

        # Specify the amount of training/testing information.
        numTrainingInstances = 4
        numTestingInstances = 4

        # Get the reconstruction data mask
        reconstructionDataTrainingMask = self.dataInterface.getEmotionColumn(allTrainingMasks, reconstructionIndex)  # Dim: numExperiments
        reconstructionDataTestingMask = self.dataInterface.getEmotionColumn(allTestingMasks, reconstructionIndex)  # Dim: numExperiments
        # Only take a small set of training examples.
        reconstructionDataTrainingMask = reconstructionDataTrainingMask
        reconstructionDataTestingMask = reconstructionDataTestingMask

        # Compile the training data to plot
        trainingSignalData = segmentedSignalData[reconstructionDataTrainingMask][0: numTrainingInstances]
        trainingmetadata = allMetadata[reconstructionDataTrainingMask][0: numTrainingInstances]
        # Compile the training data to plot
        testingSignalData = segmentedSignalData[reconstructionDataTestingMask][0: numTestingInstances]
        testingmetadata = allMetadata[reconstructionDataTestingMask][0: numTestingInstances]

        # Stop gradient tracking
        with torch.no_grad():
            # Pass all the data through the model and store the emotions, activity, and intermediate variables.
            signalEncodingTrainingOutputs, autoencodingTrainingOutputs, emotionModelTrainingOutputs = model(trainingSignalData, trainingmetadata, trainingSignalData, reconstructSignals=True, compileVariables=False, submodel=submodel, trainingFlag=False)

            # Unpack all the data.
            trainingCompressedData, trainingReconstructedEncodedData, trainingDenoisedDoubleReconstructedData, trainingAutoencoderLayerLoss = autoencodingTrainingOutputs
            trainingEncodedData, trainingReconstructedData, trainingPredictedIndexProbabilities, trainingDecodedPredictedIndexProbabilities, trainingSignalEncodingLayerLoss = signalEncodingTrainingOutputs
            trainingMappedSignalData, trainingReconstructedCompressedData, trainingFeatureData, trainingActivityDistributions, trainingBasicEmotionDistributions, trainingFinalEmotionDistributions = emotionModelTrainingOutputs

            # Pass all the data through the model and store the emotions, activity, and intermediate variables.
            signalEncodingTestingOutputs, autoencodingTestingOutputs, emotionModelTestingOutputs = model(testingSignalData, testingmetadata, testingSignalData, reconstructSignals=True, compileVariables=False, submodel=submodel, trainingFlag=False)

            # Unpack all the data.
            testingCompressedData, testingReconstructedEncodedData, testingDenoisedDoubleReconstructedData, testingAutoencoderLayerLoss = autoencodingTestingOutputs
            testingEncodedData, testingReconstructedData, testingPredictedIndexProbabilities, testingDecodedPredictedIndexProbabilities, testingSignalEncodingLayerLoss = signalEncodingTestingOutputs
            testingMappedSignalData, testingReconstructedCompressedData, testingFeatureData, testingActivityDistributions, testingBasicEmotionDistributions, testingFinalEmotionDistributions = emotionModelTestingOutputs

        # ------------------- Plot the Data on One Device ------------------ # 

        # Plot the loss on the primary GPU.
        if self.accelerator.is_local_main_process:
            self.accelerator.print("Plotting")

            # ------------------- Signal Encoding Plots -------------------- # 

            if submodel == modelConstants.signalEncoderModel:
                with torch.no_grad():
                    # Calculate the positional encoding.
                    positionEncodedTrainingData = model.specificSignalEncoderModel.encodeSignals.positionalEncodingInterface.addPositionalEncoding(trainingSignalData.to(self.accelerator.device))
                    positionEncodedTTestingData = model.specificSignalEncoderModel.encodeSignals.positionalEncodingInterface.addPositionalEncoding(testingSignalData.to(self.accelerator.device))

                # Plot the encoding dimension.
                self.signalEncoderViz.plotOneSignalEncoding(testingEncodedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalEncoding/Test Signal Encoding", numSignalPlots=2, plotIndOffset=1)
                self.signalEncoderViz.plotOneSignalEncoding(trainingEncodedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalEncoding/Training Signal Encoding", numSignalPlots=2, plotIndOffset=1)

                # Plot the encoding example.
                self.signalEncoderViz.plotSignalEncodingMap(model, testingEncodedData.detach().cpu(), testingSignalData.detach().cpu(), epoch=currentEpoch, plotTitle="signalEncoding/Test Signal Map", numBatchPlots=1)
                self.signalEncoderViz.plotSignalEncodingMap(model, trainingEncodedData.detach().cpu(), trainingSignalData.detach().cpu(), epoch=currentEpoch, plotTitle="signalEncoding/Training Signal Encoding Map", numBatchPlots=1)

                # Plot the positional encoding example.
                self.signalEncoderViz.plotOneSignalEncoding(testingPredictedIndexProbabilities.detach().softmax(dim=-1).cpu(), testingDecodedPredictedIndexProbabilities.detach().softmax(dim=-1).cpu(), epoch=currentEpoch, plotTitle="signalEncoding/Test Positional Encoding Distribution", numSignalPlots=1, plotIndOffset=0)
                self.signalEncoderViz.plotOneSignalEncoding(trainingPredictedIndexProbabilities.detach().softmax(dim=-1).cpu(), trainingDecodedPredictedIndexProbabilities.detach().softmax(dim=-1).cpu(), epoch=currentEpoch, plotTitle="signalEncoding/Training Positional Encoding Distribution", numSignalPlots=1, plotIndOffset=0)

                # Plot the positional encoding example.
                self.signalEncoderViz.plotOneSignalEncoding(positionEncodedTrainingData.detach().cpu(), trainingSignalData.detach().cpu(), epoch=currentEpoch, plotTitle="signalEncoding/Test Positional Encoding Reconstruction", numSignalPlots=1, plotIndOffset=1)
                self.signalEncoderViz.plotOneSignalEncoding(positionEncodedTTestingData.detach().cpu(), testingSignalData.detach().cpu(), epoch=currentEpoch, plotTitle="signalEncoding/Training Positional Encoding Reconstruction", numSignalPlots=1, plotIndOffset=1)

            # Plot the autoencoder results.
            self.autoencoderViz.plotEncoder(testingSignalData.detach().cpu(), testingReconstructedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalReconstruction/Signal Encoding Test Reconstruction", numSignalPlots=1)
            self.autoencoderViz.plotEncoder(trainingSignalData.detach().cpu(), trainingReconstructedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalReconstruction/Signal Encoding Training Reconstruction", numSignalPlots=1)

            # Dont keep plotting untrained models.
            if submodel == modelConstants.signalEncoderModel: return None

            # --------------------- Autoencoder Plots ---------------------- # 

            if submodel == modelConstants.autoencoderModel:
                with torch.no_grad():
                    # Reconstruct the initial data.
                    numSignalForwardPath = model.specificSignalEncoderModel.encodeSignals.simulateSignalPath(numSignals, targetNumSignals=model.numEncodedSignals)[0]
                    _, _, propagatedReconstructedTestingData, _ = model.specificSignalEncoderModel.reconstructEncodedData(testingReconstructedEncodedData.to(self.accelerator.device), numSignalForwardPath, signalEncodingLayerLoss=None, calculateLoss=False)
                    _, _, propagatedReconstructedTrainingData, _ = model.specificSignalEncoderModel.reconstructEncodedData(trainingReconstructedEncodedData.to(self.accelerator.device), numSignalForwardPath, signalEncodingLayerLoss=None, calculateLoss=False)

                # Check the autoencoder propagated error.
                self.autoencoderViz.plotEncoder(testingSignalData.detach().cpu(), propagatedReconstructedTestingData.detach().cpu(), epoch=currentEpoch,
                                                plotTitle="signalReconstruction/Reconstructed Signal Encoded Data from Autoencoder", numSignalPlots=2)
                self.autoencoderViz.plotEncoder(trainingSignalData.detach().cpu(), propagatedReconstructedTrainingData.detach().cpu(), epoch=currentEpoch,
                                                plotTitle="signalReconstruction/Reconstructed Signal Encoded Data from Autoencoder", numSignalPlots=2)

                batchInd = 0
                numDistortedPlots = 2
                numSignalDistortions = 10
                # For each signal we are distorting.
                for distortedSignalInd in range(min(numDistortedPlots, segmentedSignalData.size(1))):
                    # Compile information about the distorted plots.
                    trueSignal = testingEncodedData[batchInd:batchInd + 1, distortedSignalInd:distortedSignalInd + 1, :]
                    distortedSignals = trueSignal.clone() + torch.randn((1, numSignalDistortions, testingEncodedData.size(2)), device=trainingSignalData.mainDevice) * 0.05
                    # If we don't have any signals to analyze, skip this step. This could happen with distributed training.
                    if distortedSignals.size(1) == 0: self.accelerator.print("No training data found on device"); continue

                    # Put the distorted signals through the autoencoder. 
                    with torch.no_grad():
                        compressedDistortedSignals, reconstructedDistortedEncodedSignals, _ = model.autoencoderModel(distortedSignals.to(self.accelerator.device), reconstructSignals=True, calculateLoss=False, trainingFlag=False)

                    # Plot the distorted signals
                    self.autoencoderViz.plotAllSignalComparisons(distortedSignals[0].detach().cpu(), reconstructedDistortedEncodedSignals[0].detach().cpu(), trueSignal[0][0].detach().cpu(), epoch=currentEpoch, signalInd=distortedSignalInd, plotTitle="signalReconstruction/Distorted Autoencoding Training Data")

            # Plot the autoencoder results.
            self.autoencoderViz.plotEncoder(testingEncodedData.detach().cpu(), testingReconstructedEncodedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalReconstruction/Autoencoder Test Reconstruction", numSignalPlots=1)
            self.autoencoderViz.plotEncoder(trainingEncodedData.detach().cpu(), trainingReconstructedEncodedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalReconstruction/Autoencoder Training Reconstruction", numSignalPlots=1)

            # Visualize autoencoding signal reconstruction.
            self.autoencoderViz.plotSignalComparison(testingEncodedData.detach().cpu(), testingCompressedData.detach().cpu(), epoch=currentEpoch, plotTitle="autoencoder/Autoencoding Test Compression", numSignalPlots=1)
            self.autoencoderViz.plotSignalComparison(trainingEncodedData.detach().cpu(), trainingCompressedData.detach().cpu(), epoch=currentEpoch, plotTitle="autoencoder/Autoencoding Training Compression", numSignalPlots=1)

            # Dont keep plotting untrained models.
            if submodel == modelConstants.autoencoderModel: return None

            # ------------------ Manifold Projection Plots ----------------- # 

            if submodel == modelConstants.emotionModel:
                # Plot latent encoding loss.
                self.generalViz.plotTrainingLosses([model.mapSignals.trainingLosses_timeReconstructionAnalysis, model.mapSignals.trainingLosses_mappedMean,
                                                    model.mapSignals.trainingLosses_mappedMinMax],
                                                   [model.mapSignals.testingLosses_timeReconstructionAnalysis, model.mapSignals.testingLosses_mappedMean,
                                                    model.mapSignals.testingLosses_mappedMinMax],
                                                   lossLabels=["Manifold Projection Reconstruction Loss", "Manifold Projection Mean Loss", "Manifold Projection Standard Deviation Loss"],
                                                   plotTitle="trainingLosses/Manifold Projection Loss")

                # Plot the common latent space.
                # self.latentEncoderViz.plotLatentSpace(allLatentData, allLatentData, epoch=currentEpoch, plotTitle = "Latent Space PCA", numSignalPlots = 8)

            # Plot the autoencoder results.
            self.autoencoderViz.plotEncoder(testingEncodedData.detach().cpu(), testingReconstructedEncodedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalReconstruction/Manifold Test Reconstruction", numSignalPlots=1)
            self.autoencoderViz.plotEncoder(trainingEncodedData.detach().cpu(), trainingReconstructedEncodedData.detach().cpu(), epoch=currentEpoch, plotTitle="signalReconstruction/Manifold Training Reconstruction", numSignalPlots=1)

            # Dont keep plotting untrained models.
            if submodel == modelConstants.emotionModel: return None

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
