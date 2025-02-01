# General
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Wedge
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

        # Create the colormap
        colors = []; num_steps = 200
        for alpha in np.linspace(start=1, stop=0, num=num_steps):
            c = blue_rgb * alpha + (1 - alpha) * white_rgb
            colors.append(c)
        for alpha in np.linspace(start=0, stop=1, num=num_steps):
            c = red_rgb * alpha + (1 - alpha) * white_rgb
            colors.append(c)

        # Create the colormap
        self.custom_cmap = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

    # --------------------- Visualize Model Parameters --------------------- #

    def plotProfilePath(self, relativeTimes, healthProfile, retrainingProfilePath, epoch, saveFigureLocation="signalEncoding/", plotTitle="Health Profile State Path"):
        # retrainingProfilePath: (numProfileShots or numProcessingLayers, numExperiments, encodedDimension)
        # Extract the signal dimensions.
        numProfileSteps, batchInd = len(retrainingProfilePath), 0

        for profileStep in range(numProfileSteps):
            plt.plot(relativeTimes, retrainingProfilePath[profileStep, batchInd], '-', c=self.lightColors[1], linewidth=1, markersize=4, alpha=0.3*(numProfileSteps - profileStep)/numProfileSteps)
            plt.plot(relativeTimes, retrainingProfilePath[profileStep, batchInd], '-', c=self.lightColors[0], linewidth=1, markersize=4, alpha=0.6*(1 - (numProfileSteps - profileStep)/numProfileSteps))
        plt.plot(relativeTimes, healthProfile[batchInd], '-', c=self.blackColor, label=f"Health profile", linewidth=1, markersize=6, alpha=0.3)
        plt.hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], colors=self.blackColor, linestyles='-', linewidth=1)

        # Plotting aesthetics.
        plt.xlabel("Time (sec)")
        plt.title(f"{plotTitle} epoch{epoch}")
        plt.ylabel("Signal (AU)")
        if "Health Profile Generation" in plotTitle: plt.ylim((-1, 1))
        else: plt.ylim((-1.5, 1.5))

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", showPlot=not self.hpcFlag)
        else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)

    def plotProfileReconstructionError(self, relativeTimes, healthProfile, reconstructedHealthProfile, epoch=0, batchInd=0, saveFigureLocation="", plotTitle="Signal Encoding"):
        # Extract the signal dimensions.
        healthError = (healthProfile[:, None, :] - reconstructedHealthProfile)[batchInd]

        # Plot the signal reconstruction error.
        plt.plot(relativeTimes, healthError.mean(axis=0), c=self.blackColor, label=f"Health profile error", linewidth=2, alpha=0.8)
        plt.plot(relativeTimes, healthError.T, c=self.lightColors[0], linewidth=1, alpha=0.1)

        # Plotting aesthetics.
        plt.xlabel("Time (Seconds)")
        plt.title(f"{plotTitle} epoch{epoch}")
        plt.ylabel("Signal error (AU)")

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", showPlot=not self.hpcFlag)
        else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)

    def plotProfileReconstruction(self, relativeTimes, healthProfile, reconstructedHealthProfile, epoch=0, batchInd=0, saveFigureLocation="", plotTitle="Signal Encoding"):
        # Extract the signal dimensions.
        reconstructionError = np.square(healthProfile[:, None, :] - reconstructedHealthProfile)[batchInd]

        # Plot the signal reconstruction.
        plt.plot(relativeTimes, healthProfile[batchInd], c=self.blackColor, label=f"Health profile", linewidth=2, alpha=0.8)
        plt.errorbar(x=np.arange(0, len(reconstructionError)), y=reconstructionError.mean(axis=-1), yerr=reconstructionError.std(axis=-1), color=self.darkColors[1], capsize=3, linewidth=2)

        # Plot the signal reconstruction.
        plt.plot(relativeTimes, reconstructedHealthProfile[batchInd].T, c=self.lightColors[0], linewidth=1, alpha=0.1)
        plt.axhline(y=0, color=self.blackColor, linewidth=0.5, alpha=0.25)

        # Plotting aesthetics.
        plt.xlabel("Time (sec)")
        plt.title(f"{plotTitle} epoch{epoch}")
        plt.ylabel("Signal (AU)")
        plt.ylim((-1, 1))

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", showPlot=not self.hpcFlag)
        else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)

    def plotEncoder(self, initialSignalData, reconstructedSignals, comparisonTimes, comparisonSignal, signalNames, epoch, batchInd, saveFigureLocation="", plotTitle="Encoder Prediction"):
        # Assert the integrity of the incoming data
        assert initialSignalData.shape[0:2] == comparisonSignal.shape[0:2], f"{initialSignalData.shape} {comparisonSignal.shape}"
        batchSize, numSignals, numEncodedPoints = comparisonSignal.shape
        if batchSize == 0: return None

        # Unpack the data
        datapoints = emotionDataInterface.getChannelData(initialSignalData, channelName=modelConstants.signalChannel)
        timepoints = emotionDataInterface.getChannelData(initialSignalData, channelName=modelConstants.timeChannel)

        for signalInd in range(min(2, numSignals)):
            times, data = timepoints[batchInd, signalInd, :], datapoints[batchInd, signalInd, :]
            reconstructedData = reconstructedSignals[batchInd, signalInd, :]

            # Plot the signal reconstruction.
            plt.plot(times, data, 'o', color=self.blackColor, markersize=2, alpha=0.75, label="Initial Signal")
            plt.plot(times, reconstructedData, 'o', color=self.lightColors[0], markersize=2, alpha=1, label="Reconstructed Signal")
            plt.plot(comparisonTimes, comparisonSignal[batchInd, signalInd, :], self.lightColors[1], linewidth=2, alpha=1, label="Resampled Signal")
            plt.axhline(y=0, color=self.blackColor, linewidth=0.5, alpha=0.25)

            # Plotting aesthetics.
            plt.title(f"{plotTitle} {signalNames[signalInd]} epoch{epoch}")
            plt.ylabel("Signal (AU)")
            plt.legend(loc="best")
            plt.xlabel("Time (sec)")
            plt.ylim((-1.5, 1.5))

            # Save the figure.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} {signalNames[signalInd]} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle} {signalNames[signalInd]}.pdf", showPlot=not self.hpcFlag)
            else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)

            # Plot the signal reconstruction.
            plt.plot(times, reconstructedData - data, 'o', color=self.darkColors[0], markersize=2, alpha=0.9, label="Signal Reconstruction Error")
            plt.axhline(y=0, color=self.blackColor, linewidth=0.5, alpha=0.25)

            # Plotting aesthetics.
            plt.title(f"{plotTitle} {signalNames[signalInd]} Error epoch{epoch}")
            plt.ylabel("Signal (AU)")
            plt.legend(loc="best")
            plt.xlabel("Time (sec)")
            plt.ylim((-1.5, 1.5))

            # Save the figure.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} {signalNames[signalInd]} Error epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle} {signalNames[signalInd]} Error.pdf", showPlot=not self.hpcFlag)
            else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)
            break

    def plotSignalEncodingStatePath(self, relativeTimes, compiledSignalEncoderLayerStates, batchInd, signalInd, vMin, signalNames, epoch, hiddenLayers, saveFigureLocation, plotTitle):
        numLayers, numExperiments, numSignals, encodedDimension = compiledSignalEncoderLayerStates.shape

        # Interpolate the states.
        compiledSignalEncoderLayerStates = compiledSignalEncoderLayerStates[:, batchInd, signalInd, :]
        numSpecificEncoderLayers = modelConstants.userInputParams['numSpecificEncoderLayers']
        numSharedEncoderLayers = modelConstants.userInputParams['numSharedEncoderLayers']
        interpolated_states = compiledSignalEncoderLayerStates

        # These should be chosen based on your data and how you want to "zoom"
        relativeTimesExtent = (relativeTimes.min(), relativeTimes.max(), 0, numLayers)
        plt.figure(figsize=(12, 8))

        # Plot the rest of the layers with the same normalization.
        im0 = plt.imshow(interpolated_states, cmap='viridis', interpolation=None, extent=relativeTimesExtent, aspect='auto', origin='lower', vmin=-vMin, vmax=vMin)
        plt.colorbar(im0, fraction=0.046, pad=0.04)

        # Add horizontal lines to mark layer boundaries
        plt.hlines(y=hiddenLayers + numSpecificEncoderLayers, xmin=plt.xlim()[0], xmax=plt.xlim()[1], colors=self.blackColor, linestyles='dashed', linewidth=2)
        plt.hlines(y=hiddenLayers, xmin=plt.xlim()[0], xmax=plt.xlim()[1], colors=self.blackColor, linestyles='-', linewidth=2)

        # Ticks, labels, and formatting
        yTicks = np.asarray([0] + list(range(1, 1 + numSpecificEncoderLayers)) + list(range(1, 1 + numSharedEncoderLayers)))
        plt.yticks(ticks=np.arange(start=0.5, stop=1 + numSpecificEncoderLayers + numSharedEncoderLayers, step=1), labels=yTicks, fontsize=12)
        plt.title(label=f"{plotTitle} epoch{epoch}", fontsize=16)
        plt.ylabel(ylabel="Module layer", fontsize=14)
        plt.xlabel(xlabel="Time (sec)", fontsize=14)
        plt.xticks(fontsize=12)
        plt.grid(False)

        # Save or clear figure
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} {signalNames[signalInd]}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", showPlot=not self.hpcFlag)
        else: self.clearFigure(fig=None, legend=None, showPlot=False)

    # --------------------- Visualize Model Training --------------------- #

    def plotAngleLocations(self, givensAnglesPath, reversibleModuleNames, signalNames, epoch, signalInd, saveFigureLocation, plotTitle):
        # givensAnglesPath: numModuleLayers, numSignals, numParams
        nRows, nCols = self.getRowsCols(numModuleLayers=len(givensAnglesPath))
        initialAngle = np.pi / 4; initX, initY = np.cos(initialAngle), np.sin(initialAngle)

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(4 * nCols, 4 * nRows), squeeze=False, sharex=True, sharey=False)  # squeeze=False ensures axes is 2D
        numProcessing, numLow, numHigh, highFreqCol = -1, -1, -1, -1
        plt.ylim((-0.05, 1.05))
        plt.xlim((-0.05, 1.05))

        # Get the angular thresholds.
        angularThresholdMin = modelConstants.userInputParams['angularThresholdMin']
        angularThresholdMax = modelConstants.userInputParams['angularThresholdMax']
        center = (0, 0)

        for layerInd in range(len(givensAnglesPath)):
            moduleName = reversibleModuleNames[layerInd].lower()

            if "spatial" in moduleName: numProcessing += 1; rowInd, colInd = numProcessing, 0
            elif "low" in moduleName: numLow += 1; rowInd, colInd = numLow, nCols - 1
            elif "high" in moduleName: highFreqCol += 1; rowInd = highFreqCol // (nCols - 2); colInd = 1 + highFreqCol % (nCols - 2)
            else: raise ValueError("Activation module name must contain 'specific' or 'shared'.")
            ax = axes[rowInd, colInd]

            if "specific" in moduleName: lineColor = self.lightColors[0]; alpha = 0.75; centerColor = self.darkColors[0]
            elif "shared" in moduleName: lineColor = self.lightColors[1]; alpha = 0.33; centerColor = self.darkColors[1]
            else: raise ValueError("Activation module name must contain 'specific' or 'shared'.")
            if rowInd == 0: ax.set_title(" ".join(moduleName.split(" ")[1:]).capitalize(), fontsize=16)
            if colInd == 0: ax.set_ylabel(f"Layer {rowInd + 1}", fontsize=16)

            # Plot the potential output vectors.
            angles = initialAngle + givensAnglesPath[layerInd][signalInd]; centerPoint = np.zeros_like(angles)
            ax.quiver(centerPoint, centerPoint, np.cos(angles), np.sin(angles), scale=1, angles='xy', scale_units='xy', color=lineColor, width=0.0025, zorder=8, alpha=alpha)

            # Draw unit circle for reference
            arc = Arc(xy=(0, 0), width=2, height=2, theta1=0, theta2=90, edgecolor=self.darkColors[-1], facecolor='none', linewidth=0.5, alpha=0.25, zorder=1)
            ax.scatter(0, 0, color=centerColor, linewidth=0.5, s=10)  # Highlight the origin
            ax.add_patch(arc)

            # Draw arrow from (0,0) to (arrow_x, arrow_y)
            ax.quiver(0, 0, initX, initY, scale=1, angles='xy', scale_units='xy', color=self.blackColor, width=0.0075, headwidth=4.5, headlength=6, zorder=10)

            # Axis arrows
            ax.quiver(0, 0, 0, 1, scale=1, angles='xy', scale_units='xy', color=self.darkColors[-1], width=0.01, headwidth=6, headlength=8, zorder=9)  # +Y direction
            ax.quiver(0, 0, 1, 0, scale=1, angles='xy', scale_units='xy', color=self.darkColors[-1], width=0.01, headwidth=6, headlength=8, zorder=9)  # +X direction

            if 'shared' in moduleName or epoch == 0: continue
            # Define the shaded region in the bounded range [-minAngle, minAngle]
            bounded_wedge = Wedge(center=center, r=1, theta1=initialAngle * 180 / np.pi - angularThresholdMin, theta2=initialAngle * 180 / np.pi + angularThresholdMin, color=self.blackColor, alpha=0.1, zorder=0)
            lower_wedge = Wedge(center=center, r=1, theta1=0, theta2=max(0, initialAngle * 180 / np.pi - angularThresholdMax), color=self.blackColor, alpha=1, zorder=0)
            upper_wedge = Wedge(center=center, r=1, theta1=min(90, initialAngle * 180 / np.pi + angularThresholdMax), theta2=90, color=self.blackColor, alpha=1, zorder=0)
            ax.add_patch(upper_wedge); ax.add_patch(bounded_wedge); ax.add_patch(lower_wedge)
        plt.suptitle(t=f"{plotTitle}; Epoch {epoch}\n", fontsize=24)
        fig.supylabel(r"Signal index: $\mathbb{\mathit{i}}$", fontsize=20)
        fig.supxlabel(r"Signal index: $\mathbb{\mathit{j}}$", fontsize=20)
        plt.ylim((-0.05, 1.05))
        plt.xlim((-0.05, 1.05))

        # Save the plot
        plt.tight_layout()
        fig.set_constrained_layout(True)
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} {signalNames[signalInd]} cutoff{angularThresholdMax} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle} {signalNames[signalInd]}.pdf", clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)

    def plotsGivensAnglesHist(self, givensAnglesPath, reversibleModuleNames, epoch, signalInd, degreesFlag, saveFigureLocation, plotTitle):
        # givensAnglesPath: numModuleLayers, numSignals, numAngles
        nRows, nCols = self.getRowsCols(numModuleLayers=len(givensAnglesPath))
        if not degreesFlag: scaleFactor = 180 / math.pi; degreesFlag = True
        else: scaleFactor = 1
        yMax = 0.4

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6 * nCols, 4 * nRows), squeeze=False, sharex=True, sharey='col')  # squeeze=False ensures axes is 2D
        numProcessing, numLow, numHigh, highFreqCol = -1, -1, -1, -1
        units = "degrees" if degreesFlag else "radians"
        degrees = (180 if degreesFlag else math.pi) / 4
        bins = np.arange(-degrees, degrees + 1, 1)

        # Get the angular thresholds.
        angularThresholdMin = modelConstants.userInputParams['angularThresholdMin']
        angularThresholdMax = modelConstants.userInputParams['angularThresholdMax']
        plt.xlim((-degrees, degrees))
        histogramPlots = []

        for layerInd in range(len(givensAnglesPath)):
            moduleName = reversibleModuleNames[layerInd].lower()

            if "spatial" in moduleName: numProcessing += 1; rowInd, colInd = numProcessing, 0
            elif "low" in moduleName: numLow += 1; rowInd, colInd = numLow, nCols - 1
            elif "high" in moduleName: highFreqCol += 1; rowInd = highFreqCol // (nCols - 2); colInd = 1 + highFreqCol % (nCols - 2)
            else: raise ValueError("Activation module name must contain 'specific' or 'shared'.")
            ax = axes[rowInd, colInd]

            # Customize subplot title and axes
            if rowInd == 0: ax.set_title(" ".join(moduleName.split(" ")[1:]).capitalize(), fontsize=16)
            if colInd == 0: ax.set_ylabel(f"Layer {rowInd + 1}", fontsize=16)

            # Plot training eigenvalue angles
            histograms = scaleFactor * givensAnglesPath[layerInd][signalInd:signalInd + len(self.darkColors) - 1].T  # histograms: numAngles, numSignals=6
            histogramsABS = np.abs(histograms); numSignals = histograms.shape[1]

            if 'shared' in moduleName or epoch == 0:
                histogramPlots.append(ax.hist(histograms, bins=bins, color=self.darkColors[0:numSignals], alpha=1, density=True, edgecolor=self.blackColor, linewidth=0.1, histtype='bar', stacked=True, align='left', cumulative=False))
            else:
                # Split the histograms into small and large angles
                smallAngles = np.where(histogramsABS < angularThresholdMin, histograms, np.nan)
                largeAngles = np.where(histogramsABS >= angularThresholdMin, histograms, np.nan)

                # Plot the histograms.
                histogramPlots.append(ax.hist(smallAngles, bins=bins, color=self.darkColors[0:numSignals], alpha=0.5, density=True, edgecolor=self.blackColor, linewidth=0.1, histtype='bar', stacked=True, align='left', cumulative=False))
                histogramPlots.append(ax.hist(largeAngles, bins=bins, color=self.darkColors[0:numSignals], alpha=1, density=True, edgecolor=self.blackColor, linewidth=0.1, histtype='bar', stacked=True, align='left', cumulative=False))

            # Customize subplot title and axes
            ax.set_title(f"{reversibleModuleNames[layerInd]}")
            ax.set_ylim((0, yMax))

            # Shade the angular thresholds
            if 'shared' in moduleName or epoch == 0: continue
            ax.fill_betweenx(y=(0, yMax), x1=-angularThresholdMin, x2=angularThresholdMin, color=self.blackColor, alpha=0.1, zorder=0)
            ax.axvspan(-degrees, -angularThresholdMax, color=self.blackColor, alpha=1, zorder=0)
            ax.axvspan(angularThresholdMax, degrees, color=self.blackColor, alpha=1, zorder=0)
        plt.xlim((-angularThresholdMax, angularThresholdMax))

        # Adjust layout to prevent overlapping titles/labels
        plt.suptitle(t=f"{plotTitle}; Epoch {epoch}\n", fontsize=16)
        fig.supxlabel(f"Angle ({units})")
        fig.supylabel("Density")
        plt.tight_layout()
        fig.set_constrained_layout(True)

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} cutoff{str(round(angularThresholdMax, 4)).replace('.', '-')} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle} cutoff{str(round(angularThresholdMax, 4)).replace('.', '-')}.pdf", clearFigure=False, showPlot=False)
        else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)

        plt.xlim((-degrees, degrees))
        # Access and modify patches correctly
        for histogramPlot in histogramPlots:  # Access histograms
            for bar_container in histogramPlot[2]:  # Access BarContainer objects
                patches = bar_container.patches if hasattr(bar_container, 'patches') else [bar_container]  # Access patches
                for patch in patches:  # Access individual bars
                    patch.set_edgecolor(None)  # Remove edge color
                    patch.set_linewidth(0)  # Remove edge line width

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)

    def plotsGivensAnglesLine(self, givensAnglesPath, reversibleModuleNames, epoch, signalInd, degreesFlag, saveFigureLocation, plotTitle):
        # givensAnglesPath: numModuleLayers, numSignals, numParams
        nRows, nCols = self.getRowsCols(numModuleLayers=len(givensAnglesPath))
        if not degreesFlag: scaleFactor = 180 / math.pi; degreesFlag = True
        else: scaleFactor = 1

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6 * nCols, 4 * nRows), squeeze=False, sharex='col', sharey=True)  # squeeze=False ensures axes is 2D
        numProcessing, numLow, numHigh, highFreqCol = -1, -1, -1, -1
        units = "degrees" if degreesFlag else "radians"
        degrees = (180 if degreesFlag else math.pi) / 4
        plt.ylim((-degrees, degrees))

        # Get the angular thresholds.
        angularThresholdMin = modelConstants.userInputParams['angularThresholdMin']
        angularThresholdMax = modelConstants.userInputParams['angularThresholdMax']

        for layerInd in range(len(givensAnglesPath)):
            moduleName = reversibleModuleNames[layerInd].lower()

            if "spatial" in moduleName: numProcessing += 1; rowInd, colInd = numProcessing, 0
            elif "low" in moduleName: numLow += 1; rowInd, colInd = numLow, nCols - 1
            elif "high" in moduleName: highFreqCol += 1; rowInd = highFreqCol // (nCols - 2); colInd = 1 + highFreqCol % (nCols - 2)
            else: raise ValueError("Activation module name must contain 'specific' or 'shared'.")
            ax = axes[rowInd, colInd]

            # Customize subplot title and axes
            if rowInd == 0: ax.set_title(" ".join(moduleName.split(" ")[1:]).capitalize(), fontsize=16)
            if colInd == 0: ax.set_ylabel(f"Layer {rowInd + 1}", fontsize=16)

            # Get the angles for the current layer
            lines = scaleFactor * givensAnglesPath[layerInd][signalInd:signalInd + len(self.darkColors)]  # Dimensions: numSignals, numParams
            for lineInd in range(len(lines)): ax.plot(lines[lineInd], 'o', color=self.darkColors[lineInd], alpha=0.75, markersize=2, linewidth=1)
            # Customize subplot title and axes
            ax.set_title(f"{reversibleModuleNames[layerInd]}")

            # Shade the angular thresholds
            if 'shared' in moduleName or epoch == 0: continue
            ax.fill_between(x=(0, lines.shape[1]), y1=-angularThresholdMin, y2=angularThresholdMin, color=self.blackColor, alpha=0.1, zorder=0)
            ax.axhspan(-degrees, -angularThresholdMax, color=self.blackColor, alpha=1, zorder=0)
            ax.axhspan(angularThresholdMax, degrees, color=self.blackColor, alpha=1, zorder=0)

        # Adjust layout to prevent overlapping titles/labels
        plt.suptitle(f"{plotTitle}; Epoch {epoch}\n", fontsize=16)
        plt.tight_layout()
        fig.set_constrained_layout(True)
        fig.supylabel(f"Angle ({units})")
        fig.supxlabel("Parameter Index")

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", clearFigure=False, showPlot=False)
        else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)

        # Save the plot
        plt.ylim((-angularThresholdMax, angularThresholdMax))
        if self.saveDataFolder:  self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} cutoff{str(round(angularThresholdMax, 4)).replace('.', '-')} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle} cutoff{str(round(angularThresholdMax, 4)).replace('.', '-')}.pdf", clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)

    def plotScaleFactorLines(self, scalingFactorsPath, reversibleModuleNames, epoch, saveFigureLocation, plotTitle):
        # scalingFactorsPath: numModuleLayers, numSignals
        numModuleLayers = len(reversibleModuleNames)
        sharedValues, specificValues = [], []

        for layerInd in range(numModuleLayers):
            if "shared" in reversibleModuleNames[layerInd].lower(): sharedValues.append(scalingFactorsPath[layerInd].flatten())
            elif "specific" in reversibleModuleNames[layerInd].lower(): specificValues.append(scalingFactorsPath[layerInd].flatten())
            else: raise ValueError("Activation module name must contain 'specific' or 'shared'.")
        sharedValues = np.asarray(sharedValues); specificValues = np.asarray(specificValues)
        # sharedValues: numSharedLayers=5*y, numSignals=1; specificValues: numSpecificLayers=5*x, numSignals=numSignals
        # Every line represents one of the signals.

        # Get the angles for the current layer
        plt.plot(sharedValues, 'o', color=self.darkColors[1], alpha=0.75, linewidth=1, markersize=4, label="Shared")
        plt.plot(specificValues, 'o', color=self.darkColors[0], alpha=0.5, linewidth=1, markersize=4, label="Specific")

        # Customize plot title and axes
        plt.title(f"{plotTitle}; Epoch {epoch}", fontsize=16)
        plt.xlabel("Module component")  # X-axis: values
        plt.ylabel("Scalar values")  # Y-axis: bin counts
        plt.ylim((0.9, 1.1))

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", clearFigure=True, showPlot=not self.hpcFlag)
        else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)

    def plotScaleFactorHist(self, scalingFactorsPath, reversibleModuleNames, epoch, saveFigureLocation, plotTitle):
        # scalingFactorsPath: numModuleLayers, numSignals, numParams=1
        sharedValues, specificValues = [], []
        for layerInd in range(len(scalingFactorsPath)):
            if "shared" in reversibleModuleNames[layerInd].lower(): sharedValues.extend(scalingFactorsPath[layerInd].flatten())
            elif "specific" in reversibleModuleNames[layerInd].lower(): specificValues.extend(scalingFactorsPath[layerInd].flatten())
            else: raise ValueError("Activation module name must contain 'specific' or 'shared'.")

        plt.hist(
            sharedValues,
            bins=16,
            color=self.lightColors[1],
            alpha=0.7,
            label="Shared",
            weights=sharedValues,  # Independent normalization
        )

        plt.hist(
            specificValues,
            bins=16,
            color=self.lightColors[0],
            alpha=0.7,
            label="Specific",
            weights=specificValues,  # Independent normalization
        )

        # Customize plot title and axes
        plt.title(f"{plotTitle}; Epoch {epoch}\n", fontsize=16)
        plt.xlabel("Scale factor")  # X-axis: values
        plt.ylabel("Frequency")  # Y-axis: bin counts
        plt.xlim((0.9, 1.1))
        plt.ylim((0, None))
        plt.legend()

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", clearFigure=True, showPlot=not self.hpcFlag)
        else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)

    def modelFlow(self, dataTimes, dataStates, signalNames, epoch, batchInd, signalInd, saveFigureLocation, plotTitle):
        dataStates = np.asarray(dataStates)
        dataStates = dataStates[:, signalInd]
        numModelLayers, encodedDimension = dataStates.shape
        # dataStates: numModelLayers, encodedDimension

        # Create a meshgrid for encodedDimension and numModelLayers
        x_data, y_data = np.meshgrid(dataTimes, np.arange(1, 1 + numModelLayers))
        x, y, z = x_data.flatten(), y_data.flatten(), dataStates.flatten()

        # Figure and axis settings
        fig = plt.figure(figsize=(14, 10), facecolor="white")
        ax = fig.add_subplot(111, projection='3d', facecolor="white")

        # Improved scatter points
        ax.scatter(
            x, y, z, c=z,
            cmap='viridis', edgecolors="black", linewidth=0.5,
            alpha=0.95, s=20, vmin=np.min(dataStates), vmax=np.max(dataStates)
        )

        # View and perspective adjustments
        ax.view_init(elev=25, azim=135)
        ax.dist = 4  # Adjusts perspective depth

        # Axis labels and title
        ax.set_title(plotTitle, fontsize=16, weight='bold', pad=20)
        ax.set_xlabel("Time (Sec)", fontsize=12, labelpad=10)
        ax.set_ylabel("Model Layer", fontsize=12, labelpad=10)
        ax.set_zlabel("Signal value (AU)", fontsize=12, labelpad=10)
        ax.set_zlim(-1.5, 1.5)

        # Make the aspect ratio look nicer in 3D
        ax.set_box_aspect([2, 1, 1])
        plt.tight_layout()

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch} batchInd{batchInd} {signalNames[signalInd]}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)

    def plotActivationCurvesCompressed(self, activationCurves, moduleNames, epoch, saveFigureLocation, plotTitle):
        axNames = ["Specific spatial", "Specific neural low frequency", "Specific neural high frequency",
                   "Shared spatial", "Shared neural low frequency", "Shared neural high frequency"]
        numActivations, numPointsX, numPointsY = activationCurves.shape
        nCols, nRows = 3, 2

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6 * nCols, 4 * nRows), squeeze=False, sharex=True, sharey=True)
        axes = axes.flatten()  # Flatten to 1D array for easy indexing
        numSpecificActivations, numSharedActivations = 0, 0

        for activationInd in range(numActivations):
            x, y = activationCurves[activationInd]
            activationName = moduleNames[activationInd].lower()

            if "specific" in activationName: axInd = 0; numSpecificActivations += 1; totalActivations = numSpecificActivations
            elif "shared" in activationName: axInd = 3; numSharedActivations += 1; totalActivations = numSharedActivations
            else: raise ValueError(f"Unknown activation module: {activationName}")

            if "neural" in activationName and 'low' in activationName: axInd += 1
            elif "neural" in activationName and 'high' in activationName: axInd += 2
            elif "spatial" not in activationName: raise ValueError(f"Unknown activation module: {activationName}")

            ax = axes[axInd]
            # Plot the activation curves
            ax.plot(x, y, color=self.lightColors[1], linestyle='-', linewidth=1, label="Inverse Pass", alpha=0.5*totalActivations/numActivations + 0.5)  # Plot Inverse Pass
            ax.plot(y, x, color=self.lightColors[0], linestyle='-', linewidth=1, label="Forward Pass", alpha=0.5*totalActivations/numActivations + 0.5)  # Plot Forward Pass

            ax = axes[axInd]
            ax.plot(x, x, color=self.blackColor, linestyle='--', linewidth=0.5)  # Plot Identity Line
            ax.set_title(f"{axNames[axInd]}")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Set the main title
        fig.suptitle(f"{plotTitle} - Epoch {epoch}\nForward and Inverse from x ∈ [{-1.5}, {1.5}]", fontsize=16)
        fig.supylabel("Output (Y)")
        fig.supxlabel("Input (x)")
        plt.tight_layout()

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", clearFigure=True, showPlot=not self.hpcFlag)
        else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)

    def plotActivationCurves(self, activationCurves, moduleNames, epoch, saveFigureLocation, plotTitle):
        numActivations, numPointsX, numPointsY = activationCurves.shape
        nRows, nCols = self.getRowsCols(numActivations)

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6 * nCols, 4 * nRows), squeeze=False, sharex=True, sharey=True)
        numProcessing, numLow, numHigh, highFreqCol = -1, -1, -1, -1

        for layerInd in range(numActivations):
            moduleName = moduleNames[layerInd].lower()
            x, y = activationCurves[layerInd]

            if "spatial" in moduleName: numProcessing += 1; rowInd, colInd = numProcessing, 0
            elif "low" in moduleName: numLow += 1; rowInd, colInd = numLow, nCols - 1
            elif "high" in moduleName: highFreqCol += 1; rowInd = highFreqCol // (nCols - 2); colInd = 1 + highFreqCol % (nCols - 2)
            else: raise ValueError("Activation module name must contain 'specific' or 'shared'.")
            ax = axes[rowInd, colInd]

            # Customize subplot title and axes
            if rowInd == 0: ax.set_title(" ".join(moduleName.split(" ")[1:]).capitalize(), fontsize=16)
            if colInd == 0: ax.set_ylabel(f"Layer {rowInd + 1}", fontsize=16)

            # Plot the activation curves
            ax.plot(x, y, color=self.lightColors[1], linestyle='-', linewidth=1, label="Inverse Pass", alpha=1)  # Plot Inverse Pass
            ax.plot(y, x, color=self.lightColors[0], linestyle='-', linewidth=1, label="Forward Pass", alpha=1)  # Plot Forward Pass

            ax.plot(x, x, color=self.blackColor, linestyle='--', linewidth=0.5)  # Plot Identity Line
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Set the main title
        fig.suptitle(f"{plotTitle} - Epoch {epoch}\n", fontsize=16)
        fig.supylabel("Output (Y)")
        fig.supxlabel("Input (x)")
        plt.tight_layout()

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=None, legend=None, showPlot=not self.hpcFlag)
