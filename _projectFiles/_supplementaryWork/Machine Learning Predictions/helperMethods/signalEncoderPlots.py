# General
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os

# Import files for machine learning
from .trainingPlots import trainingPlots
from scipy.ndimage import uniform_filter1d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class signalEncoderPlots(trainingPlots):
    def __init__(self, modelName, datasetNames, sharedModelWeights, savingBaseFolder, accelerator=None):
        super(signalEncoderPlots, self).__init__(modelName, datasetNames, sharedModelWeights, savingBaseFolder, accelerator)
        # General parameters
        self.savingFolder = savingBaseFolder + "signalEncoderPlots/"    # The folder to save the figures

        self.lossColors = [
            '#3498db',  # Blue shades
            '#fc827f',  # Red shades
            '#9ED98F',  # Green shades
            '#918ae1',  # Purple shades
            '#eca163',  # Orange shades
            '#f0d0ff',  # Pink shades (ADDED TO HAVE ENOUGH COLORS, CHANGE HEX)
        ]

        cmap = plt.get_cmap('tab10')
        self.timeWindowColors = [cmap(i) for i in range(8)]

    # ---------------------------------------------------------------- #

    @staticmethod
    def getSmoothedLosses(current_losses, window_length=5):

        # Check if the current time window has enough epochs to apply the filter.
        if len(current_losses) >= window_length:
            # Apply a moving average filter to the losses.
            smoothed_losses = uniform_filter1d(current_losses, size=window_length)
        elif len(current_losses) == 0:
            # If no epochs, use -1 as the smoothed loss.
            smoothed_losses = 1
        else:
            # If not enough epochs, use the original losses as smoothed losses.
            smoothed_losses = current_losses

        return smoothed_losses

    def getSmoothedFinalLosses(self, losses, window_length=5):
        # Convert the input list to a numpy array if not already.
        losses = np.asarray(losses)  # Expected dimension: (numTimeWindows, numEpochs)

        # Initialize an array to store the minimum loss for each time window.
        min_losses = np.zeros(losses.shape[0])

        # Iterate over each time window.
        for i in range(losses.shape[0]):
            smoothedLosses = self.getSmoothedLosses(losses[i, :], window_length=window_length)

            # Find the minimum loss in the smoothed losses for the current time window.
            min_losses[i] = np.min(smoothedLosses)

        return min_losses

    def signalCompressionAnalysis(self, finalTrainingDataString, plotTitle="Signal Compression Analysis"):
        print(f"\nPlotting the {plotTitle} Information")

        # Unpack the model information.
        allDummyModelPipelines = self.modelCompiler.onlyPreloadModelAttributes(self.modelName,
                                                                               self.datasetNames,
                                                                               loadSubmodel="signalEncoder",
                                                                               loadSubmodelDate=finalTrainingDataString,
                                                                               loadSubmodelEpochs=-1,
                                                                               allDummyModelPipelines=[])
        models = self.modelMigration.loadModels(allDummyModelPipelines, loadSubmodel="signalEncoder",loadSubmodelDate=finalTrainingDataString, loadSubmodelEpochs=-1,
                                       metaTraining=True, loadModelAttributes=True, loadModelWeights=True)



        # Initialize saving folder
        saveAutoencoderLossPlots = self.savingFolder + "/SignalCompression Plots/"
        os.makedirs(saveAutoencoderLossPlots, exist_ok=True)

    def signalEncoderLossComparison(self, numExpandedSignalBounds=(2, 10), numEncodingLayerBounds=(0, 12), numLiftedChannels=[], finalTrainingDataString="2024-04-08 Final signalEncoder on HPC-GPU at numExpandedSignals XX at numEncodingLayers YY", plotTitle="Signal Encoder Loss Plots"):
        saveEncoderLossPlots = self.savingFolder + "signalEncoderLossPlots/"
        os.makedirs(saveEncoderLossPlots, exist_ok=True)
        print(f"\nPlotting the {saveEncoderLossPlots}")
        allDummyModelPipelines = []

        # Define the bounds for the number of expanded signals and encoding layers.
        numExpandedSignalsTested = numExpandedSignalBounds[1] - numExpandedSignalBounds[0] + 1  # Boundary inclusive
        numEncodingLayersTested = numEncodingLayerBounds[1] - numEncodingLayerBounds[0] + 1  # Boundary inclusive
        numLiftedChannelsTestbed = numLiftedChannels

        trainingLossHolders = np.zeros(
            (len(self.timeWindows), len(self.datasetNames), len(numLiftedChannelsTestbed), numExpandedSignalsTested, numEncodingLayersTested))
        testingLossHolders = np.zeros(
            (len(self.timeWindows), len(self.datasetNames), len(numLiftedChannelsTestbed), numExpandedSignalsTested, numEncodingLayersTested))
        optimalLossHolders = np.zeros(
            (len(self.timeWindows), len(self.datasetNames), len(numLiftedChannelsTestbed), numExpandedSignalsTested, numEncodingLayersTested))

        # Dimension: (numTimeWindows, numDatasets, numExpandedSignalsTested, numEncodingLayersTested)

        # For each parameter value.
        for numLiftedChannelInd in range(len(numLiftedChannelsTestbed)):
            numLiftedChannels = numLiftedChannelsTestbed[numLiftedChannelInd]
            for numExpandedSignalInd in range(numExpandedSignalsTested):
                numExpandedSignals = numExpandedSignalBounds[0] + numExpandedSignalInd

                # For each parameter value.
                for numEncodingLayerInd in range(numEncodingLayersTested):
                    numEncodingLayers = numEncodingLayerBounds[0] + numEncodingLayerInd

                    # Load in the previous model attributes.
                    loadSubmodelDate = finalTrainingDataString.replace("XX", str(numExpandedSignals)).replace("YY",
                                                                                                              str(numEncodingLayers)).replace(
                        "ZZ", str(numLiftedChannels))
                    allDummyModelPipelines = self.modelCompiler.onlyPreloadModelAttributes(self.modelName,
                                                                                           self.datasetNames,
                                                                                           loadSubmodel="signalEncoder",
                                                                                           loadSubmodelDate=loadSubmodelDate,
                                                                                           loadSubmodelEpochs=-1,
                                                                                           allDummyModelPipelines=allDummyModelPipelines)
                    print(numLiftedChannels, numExpandedSignals, numEncodingLayers)

                    # For each model, get the losses.
                    lossCurves = np.empty((len(self.timeWindows), len(self.datasetNames)), dtype=object)
                    smoothedLossCurves = np.empty((len(self.timeWindows), len(self.datasetNames)), dtype=object)
                    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
                    for modelInd in range(len(allDummyModelPipelines)):
                        currentModel = self.getSubmodel(allDummyModelPipelines[modelInd], submodel="signalEncoder")

                        for timeWindowInd in range(len(self.timeWindows)):
                            trainLoss = currentModel.trainingLosses_timeReconstructionAnalysis[timeWindowInd]
                            smoothedTrainLoss = self.getSmoothedLosses(currentModel.trainingLosses_timeReconstructionAnalysis[timeWindowInd])

                            testLoss = currentModel.testingLosses_timeReconstructionAnalysis[timeWindowInd]
                            smoothedTestLoss = self.getSmoothedLosses(currentModel.testingLosses_timeReconstructionAnalysis[timeWindowInd])

                            lossCurves[timeWindowInd, modelInd] = [trainLoss, testLoss]
                            smoothedLossCurves[timeWindowInd, modelInd] = [smoothedTrainLoss, smoothedTestLoss]

                        # Get the losses.
                        optimalLossHolders[:, modelInd, numLiftedChannelInd, numExpandedSignalInd,
                        numEncodingLayerInd] = self.getSmoothedFinalLosses(
                            currentModel.trainingLosses_timeReconstructionSVDAnalysis)
                        trainingLossHolders[:, modelInd, numLiftedChannelInd, numExpandedSignalInd,
                        numEncodingLayerInd] = self.getSmoothedFinalLosses(
                            currentModel.trainingLosses_timeReconstructionAnalysis)
                        testingLossHolders[:, modelInd, numLiftedChannelInd, numExpandedSignalInd,
                        numEncodingLayerInd] = self.getSmoothedFinalLosses(
                            currentModel.testingLosses_timeReconstructionAnalysis)

                        # Plot the loss curves for each model
                        print(lossCurves[:, modelInd][0].shape)
                        axs[0].plot(lossCurves[:, modelInd][0], color=self.lossColors[modelInd], alpha=0.1)
                        axs[0].plot(smoothedLossCurves[:, modelInd][0], label=f'{self.datasetNames[modelInd]} Smoothed Train Loss', color=self.lossColors[modelInd])
                        axs[1].plot(lossCurves[:, modelInd][1], color=self.lossColors[modelInd], alpha=0.1)
                        axs[1].plot(smoothedLossCurves[:, modelInd][1], label=f'{self.datasetNames[modelInd]} Smoothed Test Loss', color=self.lossColors[modelInd])

                    axs[0].set_title("Training Loss")
                    axs[1].set_title("Testing Loss")
                    fig.suptitle(f"Loss Curves for numLiftedChannels {numLiftedChannels} numExpandedSignals {numExpandedSignals} numEncodingLayers {numEncodingLayers}")
                    fig.legend()

                    if self.savingFolder:
                        # Save with a high DPI for better resolution
                        print('saving to', saveEncoderLossPlots)
                        plt.savefig(f"{saveEncoderLossPlots}Loss Curve Comparison numLiftedChannels {numLiftedChannels} numExpandedSignals {numExpandedSignals} numEncodingLayers {numEncodingLayers}.pdf", format="pdf")
                    # plt.show()

        # plot losses by dataset using a bar chart with mean and error bars
        optimalLoss_plotting = []
        testingLoss_plotting = []
        trainingLoss_plotting = []
        for modelInd in range(len(allDummyModelPipelines)):
            losses = optimalLossHolders[:, modelInd, :, :, :]
            optimalLoss_plotting.append([np.mean(losses, axis=(1, 2, 3)), np.std(losses, axis=(1, 2, 3))])
            losses = trainingLossHolders[:, modelInd, :, :, :]
            trainingLoss_plotting.append([np.mean(losses, axis=(1, 2, 3)), np.std(losses, axis=(1, 2, 3))])
            losses = testingLossHolders[:, modelInd, :, :, :]
            testingLoss_plotting.append([np.mean(losses, axis=(1, 2, 3)), np.std(losses, axis=(1, 2, 3))])

        fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
        fig.suptitle(
            f"Losses for each dataset")
        axs[0].set_ylabel(f"Train Loss")
        axs[1].set_ylabel(f"Test Loss")

        print(np.array(trainingLoss_plotting).shape, np.array(testingLoss_plotting).shape)

        axs[0].bar(self.datasetNames, np.mean(np.array(trainingLoss_plotting)[:, 0], axis=1),
                   yerr=np.mean(np.array(trainingLoss_plotting)[:, 1], axis=1))
        axs[1].bar(self.datasetNames, np.mean(np.array(testingLoss_plotting)[:, 0], axis=1),
                   yerr=np.mean(np.array(testingLoss_plotting)[:, 1], axis=1))

        if self.savingFolder:
            # Save with a high DPI for better resolution
            print("saving to folder")
            plt.savefig(f"{saveEncoderLossPlots}Best Reconstruction Loss Comparison.pdf", format="pdf")
        plt.show()

        # split by time window
        fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
        fig.suptitle(
            f"Losses for each dataset")

        x = np.arange(len(self.datasetNames))
        width = 0.5

        plt.figure(figsize=(18, 5))

        plt.subplot(121)
        plt.title('Train Loss', fontsize=18)
        for timeWindowInd in range(len(self.timeWindows)):
            plt.bar(x + width / len(self.datasetNames) * timeWindowInd,
                    np.array(trainingLoss_plotting)[:, 0, timeWindowInd], width / len(self.datasetNames),
                    yerr=np.array(trainingLoss_plotting)[:, 1, timeWindowInd],
                    label=f'{self.timeWindows[timeWindowInd]}s Time Window',
                    color=self.timeWindowColors[timeWindowInd])
        plt.xticks([i + width / 2 for i in range(len(self.datasetNames))], self.datasetNames, fontsize=15)

        plt.subplot(122)
        plt.title('Test Loss', fontsize=18)
        for timeWindowInd in range(len(self.timeWindows)):
            plt.bar(x + width / len(self.datasetNames) * timeWindowInd,
                    np.array(trainingLoss_plotting)[:, 0, timeWindowInd], width / len(self.datasetNames),
                    yerr=np.array(trainingLoss_plotting)[:, 1, timeWindowInd],
                    color=self.timeWindowColors[timeWindowInd])
        plt.xticks([i + width / 2 for i in range(len(self.datasetNames))], self.datasetNames, fontsize=15)

        plt.figlegend(loc='upper right', ncol=1, labelspacing=0.5, fontsize=14)
        plt.tight_layout(w_pad=6)
        plt.show()

        if self.savingFolder:
            # Save with a high DPI for better resolution
            print('saving to', saveEncoderLossPlots)
            plt.savefig(f"{saveEncoderLossPlots}Best Reconstruction Loss Comparison.pdf", format="pdf")
        plt.show()



    def signalEncoderParamHeatmap(self, numExpandedSignalBounds=(2, 10), numEncodingLayerBounds=(0, 12), numLiftedChannels=[], finalTrainingDataString="2024-04-21 final signalEncoder on HPC-GPU at numLiftedChannels ZZ at numExpandedSignals XX at numEncodingLayers YY", plotTitle="Signal Encoder Heatmap"):
        saveEncoderLossPlots = self.savingFolder + "heatmapParamsPlots/"
        os.makedirs(saveEncoderLossPlots, exist_ok=True)
        print(f"\nPlotting the {saveEncoderLossPlots}")
        allDummyModelPipelines = []
        
        # Define the bounds for the number of expanded signals and encoding layers.
        numExpandedSignalsTested = numExpandedSignalBounds[1] - numExpandedSignalBounds[0] + 1  # Boundary inclusive
        numEncodingLayersTested = numEncodingLayerBounds[1] - numEncodingLayerBounds[0] + 1     # Boundary inclusive
        numLiftedChannelsTestbed = numLiftedChannels

        # Initialize the holders.
        trainingLossHolders = np.zeros((len(self.timeWindows), len(self.datasetNames), len(numLiftedChannelsTestbed), numExpandedSignalsTested, numEncodingLayersTested))
        testingLossHolders = np.zeros((len(self.timeWindows), len(self.datasetNames), len(numLiftedChannelsTestbed), numExpandedSignalsTested, numEncodingLayersTested))
        optimalLossHolders = np.zeros((len(self.timeWindows), len(self.datasetNames), len(numLiftedChannelsTestbed), numExpandedSignalsTested, numEncodingLayersTested))
        # Dimension: (numTimeWindows, numDatasets, numExpandedSignalsTested, numEncodingLayersTested)

        # For each parameter value.
        for numLiftedChannelInd in range(len(numLiftedChannelsTestbed)):
            numLiftedChannels = numLiftedChannelsTestbed[numLiftedChannelInd]
            for numExpandedSignalInd in range(numExpandedSignalsTested):
                numExpandedSignals = numExpandedSignalBounds[0] + numExpandedSignalInd

                # For each parameter value.
                for numEncodingLayerInd in range(numEncodingLayersTested):
                    numEncodingLayers = numEncodingLayerBounds[0] + numEncodingLayerInd

                    # Load in the previous model attributes.
                    loadSubmodelDate = finalTrainingDataString.replace("XX", str(numExpandedSignals)).replace("YY", str(numEncodingLayers)).replace("ZZ", str(numLiftedChannels))
                    allDummyModelPipelines = self.modelCompiler.onlyPreloadModelAttributes(self.modelName, self.datasetNames, loadSubmodel="signalEncoder", loadSubmodelDate=loadSubmodelDate, loadSubmodelEpochs=-1, allDummyModelPipelines=allDummyModelPipelines)
                    print(numLiftedChannels, numExpandedSignals, numEncodingLayers)

                    # For each model, get the losses.
                    for modelInd in range(len(allDummyModelPipelines)):
                        currentModel = self.getSubmodel(allDummyModelPipelines[modelInd], submodel="signalEncoder")

                        # Get the losses.
                        optimalLossHolders[:, modelInd, numLiftedChannelInd, numExpandedSignalInd, numEncodingLayerInd] = self.getSmoothedFinalLosses(currentModel.trainingLosses_timeReconstructionSVDAnalysis)
                        trainingLossHolders[:, modelInd, numLiftedChannelInd, numExpandedSignalInd, numEncodingLayerInd] = self.getSmoothedFinalLosses(currentModel.trainingLosses_timeReconstructionAnalysis)
                        testingLossHolders[:, modelInd, numLiftedChannelInd, numExpandedSignalInd, numEncodingLayerInd] = self.getSmoothedFinalLosses(currentModel.testingLosses_timeReconstructionAnalysis)

        # Parameters for the plots
        x_labels = range(numExpandedSignalBounds[0], numExpandedSignalBounds[1] + 1)
        y_labels = range(numEncodingLayerBounds[0], numEncodingLayerBounds[1] + 1)

        # Loop through each time window
        for time_index, time_window in enumerate(self.timeWindows):
            # Create subplots with shared x and y axes and a more appropriate size
            fig = plt.figure(figsize=(15, len(self.datasetNames) * 4))
            fig.suptitle(f"Loss Curves for Time Window: {time_window}")
            cmap = plt.get_cmap('viridis')
            # Assuming all data holders share the same min and max, for a shared colorbar
            vmax = max(np.amax(trainingLossHolders), np.amax(testingLossHolders))

            # Plot 3D heatmaps
            for datasetInd, datasetName in enumerate(self.datasetNames):
                print(datasetName)
                for i, data_holder in enumerate([trainingLossHolders, testingLossHolders]):
                    ax = fig.add_subplot(len(self.datasetNames), 2, datasetInd * 2 + i + 1, projection='3d')
                    # ax = axs[datasetInd, i]
                    curr_data = data_holder[time_index, datasetInd, :, :, :]
                    print(curr_data)
                    X, Y, Z = np.meshgrid(x_labels, y_labels, numLiftedChannelsTestbed)

                    # Flatten the arrays for scatter plotting
                    sc = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=curr_data.flatten(), cmap=cmap,
                                    norm=LogNorm(vmin=1E-3, vmax=vmax))
                    ax.set_title(f'{datasetName} {["Training Loss", "Testing Loss"][i]}')
                    ax.set_xlabel("No. of Expanded Signals")
                    ax.set_ylabel("No. of Encoding Layers")
                    ax.set_zlabel("No. of Lifted Channels")

            # Add a shared colorbar
            cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # x, y, width, height
            mappable = ScalarMappable(norm=LogNorm(vmin=1E-3, vmax=vmax), cmap=cmap)
            fig.colorbar(mappable, cax=cbar_ax, orientation='vertical')

            plt.subplots_adjust(hspace=0.5, wspace=0.5)

            if self.savingFolder:
                print('saving to', saveEncoderLossPlots)
                # Save with a high DPI for better resolution
                plt.savefig(f"{saveEncoderLossPlots}TimeWindow_{time_window}.pdf", format="pdf")
            plt.show()
