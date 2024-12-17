# General
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
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
        for profileStep in range(numProfileSteps): plt.plot(physiologicalTimes, profileStatePath[profileStep, batchInd], 'o' if noTimes else '-', c=self.lightColors[1], linewidth=1, markersize=3, alpha=0.4*(numProfileSteps - profileStep)/numProfileSteps)
        for profileStep in range(numProfileSteps): plt.plot(physiologicalTimes, profileStatePath[profileStep, batchInd], 'o' if noTimes else '-', c=self.lightColors[0], linewidth=1, markersize=3, alpha=0.8*(1 - (numProfileSteps - profileStep)/numProfileSteps))
        plt.hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], colors='k', linestyles='dashed', linewidth=1)

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle} epoch{epoch}")
        plt.ylabel("Signal (AU)")

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None, showPlot=True)

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
        else: self.clearFigure(fig=None, legend=None, showPlot=True)

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
        else: self.clearFigure(fig=None, legend=None, showPlot=True)

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
        else: self.clearFigure(fig=None, legend=None, showPlot=True)

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
            plt.plot(timepoints[batchInd, signalInd, :], reconstructedSignals[batchInd, signalInd, :], 'o', color=self.lightColors[0], markersize=2, alpha=1, label="Reconstructed Signal")
            plt.plot(comparisonTimes, comparisonSignal[batchInd, signalInd, :], self.lightColors[1], linewidth=2, alpha=1, label="Resampled Signal")

            # Plotting aesthetics.
            plt.title(f"{plotTitle} epoch{epoch} signal{signalInd + 1}")
            plt.ylabel("Signal (AU)")
            plt.legend(loc="best")
            plt.xlabel("Points")

            # Save the figure.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
            else: self.clearFigure(fig=None, legend=None, showPlot=True)

    def plotSignalEncodingStatePath(self, physiologicalTimes, compiledSignalEncoderLayerStates, epoch, saveFigureLocation, plotTitle):
        numLayers, numExperiments, numSignals, encodedDimension = compiledSignalEncoderLayerStates.shape
        batchInd, signalInd = 0, 0

        # Interpolate the states.
        compiledSignalEncoderLayerStates = compiledSignalEncoderLayerStates[:, batchInd, signalInd, :]
        # interp_func = interp1d(physiologicalTimes, compiledSignalEncoderLayerStates, axis=-1)
        # interp_points = np.linspace(start=0, stop=physiologicalTimes.max(), num=1024)
        # interpolated_states = interp_func(interp_points)
        interpolated_states = compiledSignalEncoderLayerStates

        # Create custom colormap (as in your original code)
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

        # These should be chosen based on your data and how you want to "zoom"
        physiologicalTimes_finalExtent = (physiologicalTimes.min(), physiologicalTimes.max(), numLayers - 1, numLayers)
        physiologicalTimes_initExtent1 = (physiologicalTimes.min(), physiologicalTimes.max(), 0, 1)
        physiologicalTimes_initExtent2 = (physiologicalTimes.min(), physiologicalTimes.max(), 1, 2)
        physiologicalTimes = (physiologicalTimes.min(), physiologicalTimes.max(), 2, numLayers)
        first_layer_vmin = interpolated_states.min()
        first_layer_vmax = interpolated_states.max()
        plt.figure(figsize=(12, 8))

        # Plot the last layer with its own normalization and colorbar
        plt.imshow(interpolated_states[2:-1], cmap=custom_cmap, interpolation='bilinear', extent=physiologicalTimes, aspect='auto', origin='lower', vmin=first_layer_vmin, vmax=first_layer_vmax)
        im0 = plt.imshow(interpolated_states[-1:], cmap=custom_cmap, interpolation=None, extent=physiologicalTimes_finalExtent, aspect='auto', origin='lower', vmin=first_layer_vmin, vmax=first_layer_vmax)
        plt.colorbar(im0, fraction=0.046, pad=0.04)

        # Plot the rest of the layers with the same normalization.
        plt.imshow(interpolated_states[0:1], cmap=custom_cmap, interpolation=None, extent=physiologicalTimes_initExtent1, aspect='auto', origin='lower', vmin=first_layer_vmin, vmax=first_layer_vmax)
        plt.imshow(interpolated_states[1:2], cmap=custom_cmap, interpolation=None, extent=physiologicalTimes_initExtent2, aspect='auto', origin='lower', vmin=first_layer_vmin, vmax=first_layer_vmax)

        # Add horizontal lines to mark layer boundaries
        plt.hlines(y=numLayers - 1, xmin=plt.xlim()[0], xmax=plt.xlim()[1], colors=self.blackColor, linestyles='dashed', linewidth=2)
        plt.hlines(y=2, xmin=plt.xlim()[0], xmax=plt.xlim()[1], colors=self.blackColor, linestyles='dashed', linewidth=2)
        plt.hlines(y=1, xmin=plt.xlim()[0], xmax=plt.xlim()[1], colors=self.blackColor, linestyles='-', linewidth=2)

        # Ticks, labels, and formatting
        yticks = np.array([0, 1, 1] + list(range(2, numLayers - 2)) + [1])
        plt.yticks(ticks=np.arange(start=0.5, stop=numLayers, step=1), labels=yticks, fontsize=12)
        plt.title(label=f"{plotTitle} epoch{epoch}", fontsize=16)
        plt.ylabel(ylabel="Layer Index", fontsize=14)
        plt.xlabel(xlabel="Time", fontsize=14)
        plt.xticks(fontsize=12)
        plt.grid(False)

        # Save or clear figure
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", showPlot=True)
        else: self.clearFigure(fig=None, legend=None, showPlot=False)

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
            else: self.clearFigure(fig=None, legend=None, showPlot=True)
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
        else: self.clearFigure(fig=None, legend=None, showPlot=True)
