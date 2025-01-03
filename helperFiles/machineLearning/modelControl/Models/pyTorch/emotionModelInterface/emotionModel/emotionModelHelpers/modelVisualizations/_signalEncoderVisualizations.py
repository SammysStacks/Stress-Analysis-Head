# General
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from shap.plots.colors._colors import lch2rgb

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants


class signalEncoderVisualizations(globalPlottingProtocols):

    def __init__(self, baseSavingFolder, stringID, datasetName):
        super(signalEncoderVisualizations, self).__init__()
        self.setSavingFolder(baseSavingFolder, stringID, datasetName)

        # Create custom colormap (as in your original code)
        blue_lch = [54., 70., 4.6588]
        red_lch = [54., 90., 0.35470565 + 2 * np.pi]
        blue_rgb = lch2rgb(blue_lch)
        red_rgb = lch2rgb(red_lch)
        white_rgb = np.asarray([1., 1., 1.])

        colors = []
        for alpha in np.linspace(1, 0, 100):
            c = blue_rgb * alpha + (1 - alpha) * white_rgb
            colors.append(c)
        for alpha in np.linspace(0, 1, 100):
            c = red_rgb * alpha + (1 - alpha) * white_rgb
            colors.append(c)
        self.custom_cmap = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

    # --------------------- Visualize Model Parameters --------------------- #

    def plotProfilePath(self, relativeTimes, healthProfile, retrainingProfilePath, epoch, saveFigureLocation="signalEncoding/", plotTitle="Health Profile State Path"):
        # Extract the signal dimensions.
        numProfileSteps, batchInd = len(retrainingProfilePath), 0
        noTimes = relativeTimes is None

        if noTimes: relativeTimes = np.arange(start=0, stop=len(healthProfile[batchInd]), step=1)
        for profileStep in range(numProfileSteps): plt.plot(relativeTimes, retrainingProfilePath[profileStep, batchInd], 'o--' if noTimes else '-', c=self.lightColors[1], linewidth=0.25 if noTimes else 1, markersize=4, alpha=0.3*(numProfileSteps - profileStep)/numProfileSteps)
        for profileStep in range(numProfileSteps): plt.plot(relativeTimes, retrainingProfilePath[profileStep, batchInd], 'o--' if noTimes else '-', c=self.lightColors[0], linewidth=0.25 if noTimes else 1, markersize=4, alpha=0.6*(1 - (numProfileSteps - profileStep)/numProfileSteps))
        plt.plot(relativeTimes, healthProfile[batchInd], 'o-' if noTimes else '-', c=self.blackColor, label=f"Health profile", linewidth=1 if noTimes else 2, markersize=7, alpha=0.6 if noTimes else 0.25)
        plt.hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], colors='k', linestyles='dashed', linewidth=1)

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle} epoch{epoch}")
        plt.ylabel("Signal (AU)")
        if not noTimes: plt.ylim((-1.5, 1.5))
        else: plt.ylim((-1.5, 1.5))

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None, showPlot=True)

    def plotProfileReconstructionError(self, relativeTimes, healthProfile, reconstructedHealthProfile, epoch=0, saveFigureLocation="", plotTitle="Signal Encoding"):
        # Extract the signal dimensions.
        healthError = (healthProfile[:, None, :] - reconstructedHealthProfile)
        batchSize, numSignals, sequenceLength = reconstructedHealthProfile.shape
        batchInd = 0

        plt.plot(relativeTimes, healthError[batchInd].mean(axis=0), c=self.blackColor, label=f"Health profile error", linewidth=2, alpha=0.8)
        for signalInd in range(numSignals): plt.plot(relativeTimes, healthError[batchInd, signalInd], c=self.lightColors[0], linewidth=1, alpha=0.1)

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle} epoch{epoch}")
        plt.ylabel("Signal error (AU)")

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None, showPlot=True)

    def plotProfileReconstruction(self, relativeTimes, healthProfile, reconstructedHealthProfile, epoch=0, saveFigureLocation="", plotTitle="Signal Encoding"):
        batchInd = 0

        # Extract the signal dimensions.
        reconstructionError = np.square(healthProfile[:, None, :] - reconstructedHealthProfile)[batchInd]
        batchSize, numSignals, sequenceLength = reconstructedHealthProfile.shape

        # Plot the signal reconstruction.
        plt.plot(relativeTimes, healthProfile[batchInd], c=self.blackColor, label=f"Health profile", linewidth=2, alpha=0.8)
        plt.errorbar(x=np.arange(0, len(reconstructionError)), y=reconstructionError.mean(axis=-1), yerr=reconstructionError.std(axis=-1), color=self.darkColors[1], capsize=3, linewidth=2)

        # Plot the signal reconstruction.
        for signalInd in range(numSignals): plt.plot(relativeTimes, reconstructedHealthProfile[batchInd, signalInd], c=self.lightColors[1], linewidth=1, alpha=0.1)

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle} epoch{epoch}")
        plt.ylabel("Signal (AU)")
        plt.ylim((-1.5, 1.5))

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
            plt.ylim((-1.5, 1.5))

            # Save the figure.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
            else: self.clearFigure(fig=None, legend=None, showPlot=True)

    def plotSignalEncodingStatePath(self, relativeTimes, compiledSignalEncoderLayerStates, epoch, hiddenLayers, saveFigureLocation, plotTitle):
        numLayers, numExperiments, numSignals, encodedDimension = compiledSignalEncoderLayerStates.shape
        if relativeTimes is None: relativeTimes = np.arange(start=1, stop=1 + encodedDimension, step=1)
        batchInd, signalInd = 0, 0

        # Interpolate the states.
        compiledSignalEncoderLayerStates = compiledSignalEncoderLayerStates[:, batchInd, signalInd, :]
        numSpecificEncoderLayers = modelConstants.userInputParams['numSpecificEncoderLayers']
        interpolated_states = compiledSignalEncoderLayerStates.real

        # These should be chosen based on your data and how you want to "zoom"
        relativeTimesExtent = (relativeTimes.min(), relativeTimes.max(), 0, numLayers)
        plt.figure(figsize=(12, 8))

        # Plot the rest of the layers with the same normalization.
        im0 = plt.imshow(interpolated_states, cmap=self.custom_cmap, interpolation=None, extent=relativeTimesExtent, aspect='auto', origin='lower', vmin=-1.25, vmax=1.25)
        plt.colorbar(im0, fraction=0.046, pad=0.04)

        # Add horizontal lines to mark layer boundaries
        plt.hlines(y=hiddenLayers + numSpecificEncoderLayers, xmin=plt.xlim()[0], xmax=plt.xlim()[1], colors=self.blackColor, linestyles='dashed', linewidth=2)
        plt.hlines(y=hiddenLayers, xmin=plt.xlim()[0], xmax=plt.xlim()[1], colors=self.blackColor, linestyles='-', linewidth=2)

        # Ticks, labels, and formatting
        yticks = np.asarray([0, 1, 1] + list(range(2, numLayers - 1)))
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
            plt.ylim((-1.5, 1.5))

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
            plt.plot(trueSignal, 'o', color=self.darkColors[1], linewidth=1, markersize=2, zorder=10)
            plt.plot(reconstructedDistortedSignals[i], '-', color='tab:red', linewidth=1, markersize=2, alpha=alphas[i], zorder=5)
        
        # Format the plotting
        plt.title(f"{plotTitle} epoch{epoch} signal{signalInd + 1}")
        plt.xlabel('Time (Seconds)')
        plt.ylabel("Arbitrary Axis (AU)")
        plt.legend(['Noisy Signal', 'True Signal', 'Reconstructed Signal'], loc='best')

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None, showPlot=True)

    def plotEigenValueLocations(self, trainingEigenValues, testingEigenValues, epoch, signalInd, saveFigureLocation, plotTitle):
        numLayers, nCols = trainingEigenValues.shape[0], min(4, trainingEigenValues.shape[0])
        nRows, layerInd = math.ceil(numLayers / nCols), 0

        # Create the figure and axes
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(5 * nCols, 5 * nRows), squeeze=False)
        axes = axes.flatten()

        for layerInd, ax in enumerate(axes[:numLayers]):
            # Scatter training eigenvalues
            ev_train = trainingEigenValues[layerInd, signalInd, :]
            ax.scatter(ev_train.real, ev_train.imag, color=self.lightColors[1], label="Training", s=10, linewidth=0.2, alpha=0.5)

            # Connect points to the origin
            for xi, yi in zip(ev_train.real.flatten(), ev_train.imag.flatten()):
                ax.plot([0, xi], [0, yi], color=self.lightColors[1], linestyle='-', linewidth=0.2)

            # Highlight the origin
            ax.scatter(0, 0, color=self.blackColor, label='Origin', linewidth=1)

            # Scatter testing eigenvalues if provided
            if testingEigenValues is not None and testingEigenValues.shape[1] > 0:
                ev_test = testingEigenValues[layerInd, signalInd, :]
                ax.scatter(ev_test.real, ev_test.imag, color=self.lightColors[0], label="Testing", s=10, linewidth=0.2, alpha=0.5)

                for xi, yi in zip(ev_test.real.flatten(), ev_test.imag.flatten()):
                    ax.plot([0, xi], [0, yi], color=self.lightColors[0], linestyle='-', linewidth=0.1)

            # Draw coordinate lines
            ax.axhline(0, color=self.blackColor, linewidth=0.5, alpha=0.25)
            ax.axvline(0, color=self.blackColor, linewidth=0.5, alpha=0.25)

            # Draw unit circle for reference
            circle = Circle((0, 0), 1.0, color=self.blackColor, fill=False, linestyle='-')
            ax.add_patch(circle)

            # Customize appearance
            ax.set_title(f"{plotTitle}\nEpoch {epoch}, Signal {signalInd + 1}, Layer {layerInd + 1}")
            ax.set_xlabel("Real part")
            ax.set_ylabel("Imag part")
            ax.axis('equal')

        # Remove unused subplots
        for idx in range(numLayers, nRows * nCols):
            fig.delaxes(axes[idx])

        # Adjust layout with padding
        plt.tight_layout(pad=2.0)

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} layerInd{layerInd} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None, showPlot=True)

    def plotEigenvalueAngles(self, trainingEigenValues, testingEigenValues, epoch, degreesFlag, signalInd, saveFigureLocation, plotTitle):
        numLayers, nCols = trainingEigenValues.shape[0], min(4, trainingEigenValues.shape[0])
        nRows = math.ceil(numLayers / nCols)

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(8 * nCols, 8 * nRows), squeeze=False)  # squeeze=False ensures axes is 2D

        # Flatten axes for easy indexing if you prefer
        axes = axes.flatten()

        for layerInd in range(numLayers):
            ax = axes[layerInd]  # which subplot to use
            # Plot training eigenvalue angles
            angles_training = np.angle(trainingEigenValues[layerInd, signalInd, :], deg=degreesFlag)
            ax.hist(angles_training, bins=64, alpha=0.75, density=True, color=self.lightColors[1], label="Training")

            # Plot testing angles if provided
            if testingEigenValues is not None and testingEigenValues.shape[1] > 0:
                angles_testing = np.angle(testingEigenValues[layerInd, signalInd, :], deg=degreesFlag)
                ax.hist(angles_testing, bins=64, alpha=0.5, density=True, color=self.lightColors[0], label="Testing")

            units = "degrees" if degreesFlag else "radians"
            degrees = 200 if degreesFlag else 3.25
            # Customize subplot title and axes
            ax.set_title(f"Layer {layerInd + 1}")
            ax.set_xlabel(f"Angle ({units})")
            ax.set_xlim((-degrees, degrees))
            ax.set_ylabel("Density")

        # Hide any extra subplots if numLayers < nRows * nCols
        for idx in range(numLayers, nRows * nCols):
            fig.delaxes(axes[idx])  # remove unused axes

        # Adjust layout to prevent overlapping titles/labels
        plt.suptitle(f"{plotTitle}\nEpoch {epoch}, Signal {signalInd+1}", fontsize=16)
        plt.tight_layout()

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None, showPlot=True)

    def modelPropagation3D(self, neuralEigenvalues, epoch, degreesFlag, batchInd, signalInd, saveFigureLocation, plotTitle):
        neuralEigenvalues = np.asarray(neuralEigenvalues)
        neuralEigenvalues = neuralEigenvalues[:, signalInd]
        numModelLayers, encodedDimension = neuralEigenvalues.shape

        # Create a meshgrid for encodedDimension and numModelLayers
        x_data, y_data = np.meshgrid(np.arange(encodedDimension), np.arange(1, 1 + numModelLayers))

        # Create a figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create the scatter plot
        maxHalfAngle = 180 if degreesFlag else np.pi
        if "3D Data Flow" in plotTitle: maxHalfAngle = 2*modelConstants.minMaxScale
        surf = ax.scatter(x_data.flatten(), y_data.flatten(), np.imag(neuralEigenvalues.flatten()),  # Use z-values for coloring
                          c=np.angle(neuralEigenvalues.flatten(), deg=degreesFlag), cmap='viridis', alpha=1, s=7, vmin=-maxHalfAngle, vmax=maxHalfAngle)

        # Customize the view angle
        ax.view_init(elev=30, azim=135)

        # Add labels and title
        ax.set_title(plotTitle, fontsize=16, weight='bold', pad=20)
        ax.set_xlabel("Eigenvalue Index", fontsize=12, labelpad=10)
        ax.set_ylabel("Model Layer", fontsize=12, labelpad=10)
        ax.set_zlabel("Complex Domain", fontsize=12, labelpad=10)

        # Add a color bar for the last surface
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label("Spatial Domain", fontsize=12)

        # Adjust layout and aspect ratio
        ax.set_box_aspect([2, 1, 1])
        plt.tight_layout()

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} batchInd{batchInd} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None, showPlot=True)

    def modelFlow(self, dataTimes, dataStates, epoch, batchInd, signalInd, saveFigureLocation, plotTitle):
        dataStates = np.asarray(dataStates)
        dataStates = dataStates[:, signalInd]
        numModelLayers, encodedDimension = dataStates.shape
        # dataStates: numModelLayers, encodedDimension

        # Create a meshgrid for encodedDimension and numModelLayers
        x_data, y_data = np.meshgrid(dataTimes, np.arange(1, 1 + numModelLayers))
        x, y, z = x_data.flatten(), y_data.flatten(), dataStates.flatten()

        # Create a figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface.
        surf = ax.plot_surface(x_data, y_data, dataStates, cmap=self.custom_cmap, alpha=0.5, linewidth=0, antialiased=True, vmin=-1.5*modelConstants.minMaxScale, vmax=1.5*modelConstants.minMaxScale)
        ax.scatter(x, y, z, c=dataStates.flatten(), linewidths=2, cmap='viridis', alpha=1, s=15)

        # Customize the view angle
        ax.view_init(elev=30, azim=135)

        # Axis labels and title
        ax.set_title(plotTitle, fontsize=16, weight='bold', pad=20)
        ax.set_xlabel("Time (Sec)", fontsize=12, labelpad=10)
        ax.set_ylabel("Model Layer", fontsize=12, labelpad=10)
        ax.set_zlabel("Complex Domain", fontsize=12, labelpad=10)

        # Add a color bar for the surface
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label("Spatial Domain", fontsize=12)

        # Make the aspect ratio look nicer in 3D
        ax.set_box_aspect([2, 1, 1])
        plt.tight_layout()

        # Optionally, save or show the figure
        plt.savefig(saveFigureLocation, dpi=300)
        plt.show()

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} batchInd{batchInd} signalInd{signalInd}.pdf", baseSaveFigureName=f"{plotTitle}.pdf")
        else: self.clearFigure(fig=None, legend=None, showPlot=True)
