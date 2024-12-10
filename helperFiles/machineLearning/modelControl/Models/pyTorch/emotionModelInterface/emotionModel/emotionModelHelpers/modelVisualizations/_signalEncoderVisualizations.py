# General
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
from shap.plots.colors._colors import lch2rgb

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class signalEncoderVisualizations(globalPlottingProtocols):

    def __init__(self, baseSavingFolder, stringID, datasetName):
        super(signalEncoderVisualizations, self).__init__()
        self.setSavingFolder(baseSavingFolder, stringID, datasetName)

    # --------------------- Visualize Model Parameters --------------------- #

    def plotProfilePath(self, physiologicalTimes, physiologicalProfile, profileStatePath, epoch, saveFigureLocation="signalEncoding/", plotTitle="Physiological Profile State Path"):
        # Extract the signal dimensions.
        numProfileSteps, batchInd = len(profileStatePath), 0
        noTimes = physiologicalTimes is None

        if noTimes: physiologicalTimes = np.arange(start=0, stop=len(physiologicalProfile[batchInd]), step=1)
        plt.plot(physiologicalTimes, physiologicalProfile[batchInd], 'o--' if noTimes else '-', c=self.blackColor, label=f"Physiological profile", linewidth=1 if noTimes else 2, markersize=5, alpha=0.5 if noTimes else 0.8)
        for profileStep in range(numProfileSteps): plt.plot(physiologicalTimes, profileStatePath[profileStep, batchInd], 'o' if noTimes else '-', c=self.lightColors[0], linewidth=1, markersize=3, alpha=1 - 0.8*(numProfileSteps - profileStep)/numProfileSteps)
        plt.hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], colors='k', linestyles='dashed', linewidth=1)

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle} epoch{epoch}")
        plt.ylabel("Signal (AU)")

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None)

    def plotPhysiologicalError(self, physiologicalTimes, physiologicalProfile, reconstructedPhysiologicalProfile, epoch=0, saveFigureLocation="", plotTitle="Signal Encoding"):
        # Extract the signal dimensions.
        physiologicalError = (physiologicalProfile[:, None, :] - reconstructedPhysiologicalProfile)
        batchSize, numSignals, sequenceLength = reconstructedPhysiologicalProfile.shape
        batchInd = 0

        plt.plot(physiologicalTimes, physiologicalError[batchInd].mean(axis=0), c=self.blackColor, label=f"Physiological profile error", linewidth=2, alpha=0.8)
        for signalInd in range(numSignals): plt.plot(physiologicalTimes, physiologicalError[batchInd, signalInd], c=self.lightColors[0], linewidth=1, alpha=0.1)

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle} epoch{epoch}")
        plt.ylabel("Signal error (AU)")

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None)

    def plotPhysiologicalReconstruction(self, physiologicalTimes, physiologicalProfile, reconstructedPhysiologicalProfile, epoch=0, saveFigureLocation="", plotTitle="Signal Encoding"):
        # Extract the signal dimensions.
        batchSize, numSignals, sequenceLength = reconstructedPhysiologicalProfile.shape
        batchInd = 0

        # Plot the signal reconstruction.
        plt.plot(physiologicalTimes, physiologicalProfile[batchInd], c=self.blackColor, label=f"Physiological profile", linewidth=2, alpha=0.8)
        for signalInd in range(numSignals): plt.plot(physiologicalTimes, reconstructedPhysiologicalProfile[batchInd, signalInd], c=self.lightColors[1], linewidth=1, alpha=0.1)

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle} epoch{epoch}")
        plt.ylabel("Signal (AU)")

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None)

    def plotPhysiologicalOG(self, physiologicalProfileOG, epoch, saveFigureLocation, plotTitle):
        batchInd = 0

        # Plot the signal reconstruction.
        plt.plot(physiologicalProfileOG[batchInd], 'o', c=self.blackColor, label=f"Physiological profile", linewidth=1, alpha=0.8)

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle} epoch{epoch}")
        plt.ylabel("Signal (AU)")

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None)

    def plotEncoder(self, initialSignalData, reconstructedSignals, comparisonTimes, comparisonSignal, epoch, saveFigureLocation="", plotTitle="Encoder Prediction", numSignalPlots=1):
        # Assert the integrity of the incoming data
        assert initialSignalData.shape[0:2] == comparisonSignal.shape[0:2], f"{initialSignalData.shape} {comparisonSignal.shape}"
        batchSize, numSignals, numEncodedPoints = comparisonSignal.shape
        if batchSize == 0: return None

        # Get the signals to plot.
        plottingSignals = np.arange(0, numSignalPlots)
        plottingSignals = np.concatenate((plottingSignals, np.sort(numSignals - plottingSignals - 1)))
        assert plottingSignals[-1] == numSignals - 1, f"{plottingSignals} {numSignals}"

        # Unpack the data
        datapoints = emotionDataInterface.getChannelData(initialSignalData, channelName=modelConstants.signalChannel)
        timepoints = emotionDataInterface.getChannelData(initialSignalData, channelName=modelConstants.timeChannel)

        batchInd = 0
        for signalInd in plottingSignals:
            # Plot the signal reconstruction.
            plt.plot(timepoints[batchInd, signalInd, :], datapoints[batchInd, signalInd, :], 'o', color=self.blackColor, markersize=2, alpha=0.75, label="Initial Signal")
            plt.plot(comparisonTimes, comparisonSignal[batchInd, signalInd, :], self.lightColors[1], linewidth=2, alpha=0.8, label="Resampled Signal")
            plt.plot(timepoints[batchInd, signalInd, :], reconstructedSignals[batchInd, signalInd, :], 'o', color=self.lightColors[0], markersize=3, alpha=1, label="Reconstructed Signal")

            # Plotting aesthetics.
            plt.title(f"{plotTitle} epoch{epoch} signal{signalInd + 1}")
            plt.ylabel("Signal (AU)")
            plt.legend(loc="best")
            plt.xlabel("Points")

            # Save the figure.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
            else: self.clearFigure(fig=None, legend=None)

    def plotSignalEncodingStatePath(self, physiologicalTimes, compiledSignalEncoderLayerStates, epoch, saveFigureLocation, plotTitle):
        numLayers, numExperiments, numSignals, encodedDimension = compiledSignalEncoderLayerStates.shape
        batchInd, signalInd = 0, 0

        # Interpolate the states.
        compiledSignalEncoderLayerStates = compiledSignalEncoderLayerStates[:, batchInd, signalInd, :]
        interp_func = interp1d(physiologicalTimes, compiledSignalEncoderLayerStates, axis=-1)
        interp_points = np.linspace(0, physiologicalTimes.max(), num=1024)
        interpolated_states = interp_func(interp_points)

        # Define your colors (min, mean, max)
        # colors = ["#56d0f7", "#ffffff", "#f27fb2"]  # Blue -> Purple -> Red
        # custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", colors)

        blue_lch = [54., 70., 4.6588]
        red_lch = [54., 90., 0.35470565 + 2 * np.pi]
        blue_rgb = lch2rgb(blue_lch)
        red_rgb = lch2rgb(red_lch)
        white_rgb = np.array([1., 1., 1.])

        colors = []
        for alpha in np.linspace(1, 0, 100):
            c = blue_rgb * alpha + (1 - alpha) * white_rgb
            colors.append(c)
        for alpha in np.linspace(0, 1, 100):
            c = red_rgb * alpha + (1 - alpha) * white_rgb
            colors.append(c)
        custom_cmap = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

        # Create the heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(interpolated_states, cmap=custom_cmap, interpolation=None, extent=(physiologicalTimes.min(), physiologicalTimes.max(), 0, numLayers), aspect='auto', origin='lower')
        # Magma is also good

        # Add a colorbar
        cbar = plt.colorbar()
        cbar.set_label("Signal Amplitude", fontsize=12)

        # Add ticks and grid for clarity
        plt.xticks(fontsize=12)
        plt.yticks(ticks=np.arange(numLayers + 1), labels=np.arange(0, numLayers + 1), fontsize=12)
        plt.grid(False)

        # Format the plotting
        plt.title(f"{plotTitle} epoch{epoch}", fontsize=16)
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Layer Index", fontsize=14)

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None)

    # --------------------- Visualize Model Training --------------------- #

    def plotSignalComparison(self, originalSignal, comparisonSignal, epoch, saveFigureLocation, plotTitle, numSignalPlots=1):
        """ originalSignal dimension: batchSize, numSignals, numTotalPoints """
        # Assert the integrity of the incoming data
        assert originalSignal.shape[0:2] == comparisonSignal.shape[0:2], f"{originalSignal.shape} {comparisonSignal.shape}"

        # Extract the shapes of the data
        batchSize, numSignals, numTotalPoints = originalSignal.shape
        batchSize, numSignals, numEncodedPoints = comparisonSignal.shape

        if batchSize == 0: return None

        batchInd = 0
        # For each signal
        for signalInd in range(numSignals):
            # Plot both the signals alongside each other.
            plt.plot(originalSignal[batchInd, signalInd], 'k', marker='o', linewidth=2)
            plt.plot(np.linspace(0, numTotalPoints, numEncodedPoints), comparisonSignal[batchInd, signalInd], 'tab:red', marker='o', linewidth=1)

            # Format the plotting
            plt.ylabel("Arbitrary Axis (AU)")
            plt.xlabel("Points")
            plt.title(f"{plotTitle} epoch{epoch} signal{signalInd + 1}")

            # Save the plot
            if self.saveDataFolder: self.displayFigure(saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
            else: self.clearFigure(fig=None, legend=None)

            # There are too many signals to plot.
            if signalInd + 1 == numSignalPlots: break

    def plotAllSignalComparisons(self, distortedSignals, reconstructedDistortedSignals, trueSignal, epoch, signalInd, saveFigureLocation, plotTitle):
        numSignals, numTotalPoints = reconstructedDistortedSignals.shape
        alphas = np.linspace(0.1, 1, numSignals)

        # Plot all signals in 'distortedSignals'
        for i in range(numSignals):
            plt.plot(distortedSignals[i], '-', color='k', alpha=alphas[i], linewidth=1, markersize=2, zorder=0)
            plt.plot(trueSignal, 'o', color='tab:blue', linewidth=1, markersize=2, zorder=10)
            plt.plot(reconstructedDistortedSignals[i], '-', color='tab:red', linewidth=1, markersize=2, alpha=alphas[i], zorder=5)
        
        # Format the plotting
        plt.title(f"{plotTitle} epoch{epoch} signal{signalInd + 1}")
        plt.xlabel('Time (Seconds)')
        plt.ylabel("Arbitrary Axis (AU)")
        plt.legend(['Noisy Signal', 'True Signal', 'Reconstructed Signal'], loc='best')

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None)
