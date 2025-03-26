# General
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Wedge
from selenium.webdriver.common.devtools.v85.fetch import continue_request
from shap.plots.colors._colors import lch2rgb

# Visualization protocols
from helperFiles.globalPlottingProtocols import globalPlottingProtocols
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.emotionDataInterface import emotionDataInterface
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.modelConstants import modelConstants
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleLieLayer import reversibleLieLayer
from helperFiles.machineLearning.modelControl.Models.pyTorch.emotionModelInterface.emotionModel.emotionModelHelpers.submodels.modelComponents.reversibleComponents.reversibleLieLayerInterface import reversibleLieLayerInterface


class signalEncoderVisualizations(globalPlottingProtocols):

    def __init__(self, baseSavingFolder, stringID, datasetName):
        super(signalEncoderVisualizations, self).__init__(interactivePlots=False)
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

    def plotProfilePath(self, relativeTimes, retrainingProfilePath, epoch, saveFigureLocation="signalEncoding/", plotTitle="Health Profile State Path"):
        # retrainingProfilePath: (numProfileShots or numModuleLayers, numExperiments, encodedDimension)
        # Extract the signal dimensions.
        numProfileSteps, batchInd, numSignals = len(retrainingProfilePath), 0, len(retrainingProfilePath[0][0])

        for signalInd in range(numSignals):
            fig, ax = plt.subplots(figsize=(6.4, 4.8))

            for profileStep in range(numProfileSteps):
                lineStyle = 'o-' if profileStep == numProfileSteps - 1 else '-'
                ax.plot(relativeTimes, retrainingProfilePath[profileStep, batchInd, signalInd], lineStyle, c=self.lightColors[1], linewidth=1, markersize=1, alpha=0.3*(numProfileSteps - profileStep)/numProfileSteps)
                ax.plot(relativeTimes, retrainingProfilePath[profileStep, batchInd, signalInd], lineStyle, c=self.lightColors[0], linewidth=1, markersize=1, alpha=0.6*(1 - (numProfileSteps - profileStep)/numProfileSteps))
            # if healthProfile is not None: ax.plot(relativeTimes, healthProfile[batchInd], '-', c=self.blackColor, label=f"Health profile", linewidth=1, markersize=6, alpha=0.3)
            ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors=self.blackColor, linestyles='-', linewidth=1)

            # Plotting aesthetics.
            ax.set_xlabel("Time (sec)")
            ax.set_title(f"{plotTitle} epoch{epoch}")
            ax.set_ylabel("Signal amplitude (au)")
            if "health profile" in plotTitle.lower(): ax.set_ylim((-1, 1))
            else: ax.set_ylim((-1.75, 1.75))

            # Save the figure.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", fig=fig, clearFigure=True, showPlot=False)
            else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotProfileReconstructionError(self, relativeTimes, healthProfile, reconstructedHealthProfile, batchInd, epoch, saveFigureLocation="", plotTitle="Signal Encoding"):
        # Extract the signal dimensions.
        healthError = (healthProfile[:, None, :] - reconstructedHealthProfile)[batchInd]
        fig, ax = plt.subplots(figsize=(6.4, 4.8))

        # Plot the signal reconstruction error.
        ax.plot(relativeTimes, healthError.mean(axis=0), c=self.blackColor, label=f"Health profile error", linewidth=2, alpha=0.8)
        ax.plot(relativeTimes, healthError.T, c=self.lightColors[0], linewidth=1, alpha=0.1)

        # Plotting aesthetics.
        ax.set_xlabel("Time (sec)")
        ax.set_title(f"{plotTitle} epoch{epoch}")
        ax.set_ylabel("Signal error (au)")

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", fig=fig, clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotProfileReconstruction(self, relativeTimes, healthProfile, reconstructedHealthProfile, batchInd, epoch, saveFigureLocation="", plotTitle="Signal Encoding"):
        # Extract the signal dimensions.
        fig, ax = plt.subplots(figsize=(6.4, 4.8))

        # Plot the signal reconstruction.
        ax.plot(relativeTimes, healthProfile[batchInd], c=self.blackColor, label=f"Health profile", linewidth=2, alpha=0.8)
        ax.plot(relativeTimes, reconstructedHealthProfile[batchInd].T, c=self.lightColors[0], linewidth=1, alpha=0.1)
        ax.axhline(y=0, color=self.blackColor, linewidth=0.5, alpha=0.25)

        # Plotting aesthetics.
        ax.set_xlabel("Time (sec)")
        ax.set_title(f"{plotTitle} epoch{epoch}")
        ax.set_ylabel("Signal amplitude (au)")
        ax.set_ylim((-1, 1))

        # Save the figure.
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", fig=fig, clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotEncoder(self, initialSignalData, reconstructedSignals, comparisonTimes, comparisonSignal, signalNames, batchInd, epoch, saveFigureLocation="", plotTitle="Encoder Prediction"):
        # Assert the integrity of the incoming data
        assert initialSignalData.shape[0:2] == comparisonSignal.shape[0:2], f"{initialSignalData.shape} {comparisonSignal.shape}"
        batchSize, numSignals, numEncodedPoints = comparisonSignal.shape
        if batchSize == 0: return None

        # Unpack the data
        datapoints = emotionDataInterface.getChannelData(initialSignalData, channelName=modelConstants.signalChannel)
        timepoints = emotionDataInterface.getChannelData(initialSignalData, channelName=modelConstants.timeChannel)

        for signalInd in range(numSignals):
            times, data = timepoints[batchInd, signalInd, :], datapoints[batchInd, signalInd, :]
            reconstructedData = reconstructedSignals[batchInd, signalInd, :]
            fig, ax = plt.subplots(figsize=(6.4, 4.8))

            # Plot the signal reconstruction.
            ax.plot(times, data, 'o', color=self.blackColor, markersize=2, alpha=0.75, label="Initial Signal")
            ax.plot(times, reconstructedData, 'o', color=self.lightColors[0], markersize=2, alpha=1, label="Reconstructed Signal")
            ax.plot(comparisonTimes, comparisonSignal[batchInd, signalInd, :], self.lightColors[1], linewidth=2, alpha=1, label="Resampled Signal")
            ax.axhline(y=0, color=self.blackColor, linewidth=0.5, alpha=0.25)

            # Plotting aesthetics.
            ax.set_title(f"{plotTitle} {signalNames[signalInd]} epoch{epoch}")
            ax.set_ylabel("Signal amplitude (au)")
            ax.legend(loc="best")
            ax.set_xlabel("Time (sec)")
            ax.set_ylim((-1.75, 1.75))

            # Save the figure.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} {signalNames[signalInd]} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle} {signalNames[signalInd]}.pdf", fig=fig, clearFigure=False, showPlot=False)
            else: self.clearFigure(fig=fig, legend=None, showPlot=True)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            # Plot the signal reconstruction.
            ax.plot(times, reconstructedData - data, 'o', color=self.darkColors[0], markersize=2, alpha=0.9, label="Signal Reconstruction Error")
            ax.axhline(y=0, color=self.blackColor, linewidth=0.5, alpha=0.25)

            # Plotting aesthetics.
            ax.set_title(f"{plotTitle} {signalNames[signalInd]} Error epoch{epoch}")
            ax.set_ylabel("Signal amplitude (au)")
            ax.legend(loc="best")
            ax.set_xlabel("Time (sec)")
            ax.set_ylim((-1.75, 1.75))

            # Save the figure.
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} {signalNames[signalInd]} Error epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle} {signalNames[signalInd]} Error.pdf", fig=fig, clearFigure=True, showPlot=False)
            else: self.clearFigure(fig=fig, legend=None, showPlot=True)
            plt.close(fig)
            break

    def plotSignalEncodingStatePath(self, relativeTimes, compiledSignalEncoderLayerStates, batchInd, signalNames, epoch, saveFigureLocation, plotTitle):
        numLayers, numExperiments, numSignals, encodedDimension = compiledSignalEncoderLayerStates.shape
        numSpecificEncoderLayers = modelConstants.userInputParams['numSpecificEncoderLayers']
        numSharedEncoderLayers = modelConstants.userInputParams['numSharedEncoderLayers']
        relativeTimesExtent = (relativeTimes.min(), relativeTimes.max(), 0, numLayers)
        hiddenLayers = 1

        for signalInd in range(len(signalNames)):
            # These should be chosen based on your data and how you want to "zoom"
            fig, ax = plt.subplots(figsize=(6.4, 4.8))

            # Plot the rest of the layers with the same normalization.
            im0 = ax.imshow(compiledSignalEncoderLayerStates[:, batchInd, signalInd, :], cmap='viridis', interpolation=None, extent=relativeTimesExtent, aspect='auto', origin='lower', vmin=-1.1, vmax=1.1)
            ax.set_xlim(relativeTimes.min(), relativeTimes.max())
            fig.colorbar(im0, fraction=0.046, pad=0.04)

            # Add horizontal lines to mark layer boundaries
            ax.hlines(y=hiddenLayers + numSpecificEncoderLayers, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors=self.blackColor, linestyles='dashed', linewidth=2)
            ax.hlines(y=hiddenLayers, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors=self.blackColor, linestyles='-', linewidth=2)

            # Ticks, labels, and formatting
            yTicks = np.asarray([0] + list(range(1, 1 + numSpecificEncoderLayers)) + list(range(1, 1 + numSharedEncoderLayers)))
            ax.set_yticks(np.arange(start=0.5, stop=1 + numSpecificEncoderLayers + numSharedEncoderLayers, step=1))
            ax.set_yticklabels(yTicks, fontsize=12)
            ax.tick_params(axis="x", labelsize=12)
            ax.set_title(f"{plotTitle} epoch {epoch}", fontsize=16)
            ax.set_ylabel("Module layer", fontsize=14)
            ax.set_xlabel("Time (sec)", fontsize=14)
            ax.grid(False)

            # Save or clear figure
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} {signalNames[signalInd]} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle} {signalNames[signalInd]}.pdf", fig=fig, clearFigure=True, showPlot=False)
            else: self.clearFigure(fig=None, legend=None, showPlot=False)

    # --------------------- Visualize Model Training --------------------- #

    def plotAngleLocations(self, givensAnglesPath, reversibleModuleNames, signalNames, signalInd, epoch, saveFigureLocation, plotTitle):
        # givensAnglesPath: numModuleLayers, numSignals, numParams
        nRows, nCols = self.getRowsCols(combineSharedLayers=False)
        initialAngle = np.pi / 4; initX, initY = np.cos(initialAngle), np.sin(initialAngle)

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(4 * nCols, 4 * nRows), squeeze=False, sharex=True, sharey=False)  # squeeze=False ensures axes is 2D
        numLow, numSpecificHigh, numSharedHigh = 0, 0, 0

        # Get the angular thresholds.
        minAngularThreshold = reversibleLieLayer.getMinAngularThreshold(epoch)
        maxAngularThreshold = modelConstants.userInputParams['maxAngularThreshold']
        numSpecific = modelConstants.userInputParams['numSpecificEncoderLayers']
        numShared = modelConstants.userInputParams['numSharedEncoderLayers']
        center = (0, 0)

        for layerInd in range(len(givensAnglesPath)):
            moduleName = reversibleModuleNames[layerInd].lower()

            if "low" in moduleName: rowInd, colInd = numLow, nCols - 1; numLow += 1
            elif "high" in moduleName:
                if 'shared' in moduleName: rowInd = numSpecific + (numSharedHigh % numShared); colInd = numSharedHigh // numShared; numSharedHigh += 1
                elif 'specific' in moduleName: rowInd = numSpecificHigh % numSpecific; colInd = numSpecificHigh // numSpecific; numSpecificHigh += 1
                else: raise ValueError("Module name must contain 'shared' or 'specific'")
            else: raise ValueError("Module name must contain 'low' or 'high'.")
            ax = axes[rowInd, colInd]

            if colInd == 0: ax.set_ylabel(f"Layer {rowInd + 1}", fontsize=16)
            ax.set_title(moduleName.capitalize(), fontsize=16)
            ax.set_ylim((-0.05, 1.05))
            ax.set_xlim((-0.05, 1.05))

            # Get the plot colors.
            if "specific" in moduleName: lineColor = self.lightColors[0]; alpha = 0.75; centerColor = self.darkColors[0]
            elif "shared" in moduleName: lineColor = self.lightColors[1]; alpha = 0.33; centerColor = self.darkColors[1]
            else: raise ValueError("Module name must contain 'specific' or 'shared'.")

            # Draw unit circle for reference
            arc = Arc(xy=(0, 0), width=2, height=2, theta1=0, theta2=90, edgecolor=self.darkColors[-1], facecolor='none', linewidth=0.5, alpha=0.25, zorder=1)
            ax.scatter(0, 0, color=centerColor, linewidth=0.5, s=10)  # Highlight the origin
            ax.add_patch(arc)

            # Draw arrow from (0,0) to (arrow_x, arrow_y)
            ax.quiver(0, 0, initX, initY, scale=1, angles='xy', scale_units='xy', color=self.blackColor, width=0.0075, headwidth=4.5, headlength=6, zorder=10)

            # Axis arrows
            ax.quiver(0, 0, 0, 1, scale=1, angles='xy', scale_units='xy', color=self.darkColors[-1], width=0.01, headwidth=6, headlength=8, zorder=9)  # +Y direction
            ax.quiver(0, 0, 1, 0, scale=1, angles='xy', scale_units='xy', color=self.darkColors[-1], width=0.01, headwidth=6, headlength=8, zorder=9)  # +X direction

            # Plot the potential output vectors.
            if np.all(givensAnglesPath[layerInd][signalInd] == 0): continue
            angles = givensAnglesPath[layerInd][signalInd]; angles = initialAngle + angles[angles != 0]; centerPoint = np.zeros_like(angles)
            ax.quiver(centerPoint, centerPoint, np.cos(angles), np.sin(angles), scale=1, angles='xy', scale_units='xy', color=lineColor, width=0.001, zorder=8, alpha=alpha, linewidth=0.1)

            if 'shared' in moduleName or epoch == 0: continue
            # Define the shaded region in the bounded range [-minAngle, minAngle]
            bounded_wedge = Wedge(center=center, r=1, theta1=initialAngle * 180 / np.pi - minAngularThreshold*(2 if rowInd == 0 else 1), theta2=initialAngle * 180 / np.pi + minAngularThreshold*(2 if rowInd == 0 else 1), color=self.blackColor, alpha=0.1, zorder=0)
            lower_wedge = Wedge(center=center, r=1, theta1=0, theta2=max(0, initialAngle * 180 / np.pi - maxAngularThreshold), color=self.blackColor, alpha=1, zorder=0)
            upper_wedge = Wedge(center=center, r=1, theta1=min(90, initialAngle * 180 / np.pi + maxAngularThreshold), theta2=90, color=self.blackColor, alpha=1, zorder=0)
            ax.add_patch(upper_wedge); ax.add_patch(bounded_wedge); ax.add_patch(lower_wedge)

        fig.suptitle(t=f"{plotTitle}; Epoch {epoch}", fontsize=24)
        fig.supylabel(r"Signal index: $\mathbb{\mathit{i}}$", fontsize=20)
        fig.supxlabel(r"Signal index: $\mathbb{\mathit{j}}$", fontsize=20)

        # Save the plot
        # fig.tight_layout()
        fig.set_constrained_layout(True)
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} {signalNames[signalInd]} cutoff{maxAngularThreshold} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle} {signalNames[signalInd]} cutoff{maxAngularThreshold}.pdf", fig=fig, clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotsGivensAnglesHist(self, givensAnglesPath, reversibleModuleNames, signalInd, degreesFlag, epoch, saveFigureLocation, plotTitle):
        # givensAnglesPath: numModuleLayers, numSignals, numAngles
        nRows, nCols = self.getRowsCols(combineSharedLayers=False)
        if not degreesFlag: scaleFactor = 180 / math.pi; degreesFlag = True
        else: scaleFactor = 1
        yMax = 1/4

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex=True, sharey='col')  # squeeze=False ensures axes is 2D
        numLow, numSpecificHigh, numSharedHigh = 0, 0, 0
        units = "degrees" if degreesFlag else "radians"
        degrees = (180 if degreesFlag else math.pi) / 4
        bins = np.arange(-degrees, degrees + 1, 1)

        # Get the angular thresholds.
        minAngularThreshold = reversibleLieLayer.getMinAngularThreshold(epoch)
        maxAngularThreshold = modelConstants.userInputParams['maxAngularThreshold']
        numSpecific = modelConstants.userInputParams['numSpecificEncoderLayers']
        numShared = modelConstants.userInputParams['numSharedEncoderLayers']
        histogramPlots = []

        for layerInd in range(len(givensAnglesPath)):
            moduleName = reversibleModuleNames[layerInd].lower()

            if "low" in moduleName: rowInd, colInd = numLow, nCols - 1; numLow += 1
            elif "high" in moduleName:
                if 'shared' in moduleName: rowInd = numSpecific + (numSharedHigh % numShared); colInd = numSharedHigh // numShared; numSharedHigh += 1
                elif 'specific' in moduleName: rowInd = numSpecificHigh % numSpecific; colInd = numSpecificHigh // numSpecific; numSpecificHigh += 1
                else: raise ValueError("Module name must contain 'shared' or 'specific'")
            else: raise ValueError("Module name must contain 'low' or 'high'.")
            ax = axes[rowInd, colInd]

            # Customize subplot title and axes
            if colInd == 0: ax.set_ylabel(f"Layer {rowInd + 1}", fontsize=16)
            ax.set_title(moduleName.capitalize(), fontsize=16)
            ax.set_xlim((-maxAngularThreshold, maxAngularThreshold))
            ax.set_ylim((0, yMax))

            # Plot training eigenvalue angles
            histograms = scaleFactor * givensAnglesPath[layerInd][signalInd:signalInd + len(self.darkColors) - 1].T  # histograms: numAngles, numSignals=6
            histogramsABS = np.abs(histograms); numSignals = histograms.shape[1]
            smallAngles = []

            if 'shared' in moduleName or epoch == 0:
                histogramPlots.append(ax.hist(histograms, bins=bins, color=self.darkColors[0:numSignals], alpha=1, density=True, edgecolor=self.blackColor, linewidth=0.1, histtype='bar', stacked=True, align='left', cumulative=False))
            else:
                # Split the histograms into small and large angles
                smallAngles = np.where(histogramsABS < minAngularThreshold, histograms, np.nan)
                largeAngles = np.where(histogramsABS >= minAngularThreshold, histograms, np.nan)

                # Plot the histograms.
                histogramPlots.append(ax.hist(smallAngles, bins=bins, color=self.darkColors[0:numSignals], alpha=0.5, density=True, edgecolor=self.blackColor, linewidth=0.1, histtype='bar', stacked=True, align='left', cumulative=False))
                histogramPlots.append(ax.hist(largeAngles, bins=bins, color=self.darkColors[0:numSignals], alpha=1, density=True, edgecolor=self.blackColor, linewidth=0.1, histtype='bar', stacked=True, align='left', cumulative=False))

            # Shade the angular thresholds
            if len(smallAngles) == 0: continue
            if 'shared' in moduleName or epoch == 0: continue
            ax.fill_betweenx(y=(0, yMax), x1=-minAngularThreshold*(2 if rowInd == 0 else 1), x2=minAngularThreshold*(2 if rowInd == 0 else 1), color=self.blackColor, alpha=0.1, zorder=0)
            ax.axvspan(-degrees, -maxAngularThreshold, color=self.blackColor, alpha=1, zorder=0)
            ax.axvspan(maxAngularThreshold, degrees, color=self.blackColor, alpha=1, zorder=0)

        # Adjust layout to prevent overlapping titles/labels
        fig.suptitle(t=f"{plotTitle}; Epoch {epoch}", fontsize=24)
        fig.supxlabel(f"Angle ({units})", fontsize=20)
        fig.supylabel("Density", fontsize=20)
        fig.set_constrained_layout(True)

        for ax in axes.flatten(): ax.set_xlim((-degrees, degrees))
        # Access and modify patches correctly
        for histogramPlot in histogramPlots:  # Access histograms
            for bar_container in histogramPlot[2]:  # Access BarContainer objects
                patches = bar_container.patches if hasattr(bar_container, 'patches') else [bar_container]  # Access patches
                for patch in patches:  # Access individual bars
                    patch.set_edgecolor(None)  # Remove edge color
                    patch.set_linewidth(0)  # Remove edge line width

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", fig=fig, clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotsGivensAnglesLine(self, givensAnglesPath, reversibleModuleNames, signalInd, degreesFlag, epoch, saveFigureLocation, plotTitle):
        # givensAnglesPath: numModuleLayers, numSignals, numParams
        nRows, nCols = self.getRowsCols(combineSharedLayers=False)
        if not degreesFlag: scaleFactor = 180 / math.pi; degreesFlag = True
        else: scaleFactor = 1

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex='col', sharey=True)  # squeeze=False ensures axes is 2D
        numLow, numSpecificHigh, numSharedHigh = 0, 0, 0
        units = "degrees" if degreesFlag else "radians"
        degrees = (180 if degreesFlag else math.pi) / 4

        # Get the angular thresholds.
        minAngularThreshold = reversibleLieLayer.getMinAngularThreshold(epoch)
        maxAngularThreshold = modelConstants.userInputParams['maxAngularThreshold']
        numSpecific = modelConstants.userInputParams['numSpecificEncoderLayers']
        numShared = modelConstants.userInputParams['numSharedEncoderLayers']

        for layerInd in range(len(givensAnglesPath)):
            moduleName = reversibleModuleNames[layerInd].lower()

            if "low" in moduleName: rowInd, colInd = numLow, nCols - 1; numLow += 1
            elif "high" in moduleName:
                if 'shared' in moduleName: rowInd = numSpecific + (numSharedHigh % numShared); colInd = numSharedHigh // numShared; numSharedHigh += 1
                elif 'specific' in moduleName: rowInd = numSpecificHigh % numSpecific; colInd = numSpecificHigh // numSpecific; numSpecificHigh += 1
                else: raise ValueError("Module name must contain 'shared' or 'specific'")
            else: raise ValueError("Module name must contain 'low' or 'high'.")
            ax = axes[rowInd, colInd]

            # Customize subplot title and axes
            if colInd == 0: ax.set_ylabel(f"Layer {rowInd + 1}", fontsize=16)
            ax.set_title(moduleName.capitalize(), fontsize=16)
            ax.set_ylim((-degrees, degrees))

            # Get the angles for the current layer
            lines = scaleFactor * givensAnglesPath[layerInd][signalInd:signalInd + len(self.darkColors) - 1]  # Dimensions: numSignals, numParams
            # for lineInd in range(len(lines)): ax.plot(sorted(lines[lineInd], reverse=True), 'o', color=self.darkColors[lineInd], alpha=0.75, markersize=2, linewidth=1)
            # for lineInd in range(len(lines)): ax.plot(lines[lineInd][lines[lineInd] != 0], 'o', color=self.darkColors[lineInd], alpha=0.75, markersize=2, linewidth=1)
            for lineInd in range(len(lines)): ax.plot(lines[lineInd], 'o', color=self.darkColors[lineInd], alpha=0.75, markersize=2, linewidth=1)
            # Customize subplot title and axes

            # Shade the angular thresholds
            if np.all(lines == 0): continue
            if 'shared' in moduleName or epoch == 0: continue
            ax.fill_between(x=(0, lines.shape[1]), y1=-minAngularThreshold, y2=minAngularThreshold, color=self.blackColor, alpha=0.1, zorder=0)
            ax.axhspan(-degrees, -maxAngularThreshold, color=self.blackColor, alpha=1, zorder=0)
            ax.axhspan(maxAngularThreshold, degrees, color=self.blackColor, alpha=1, zorder=0)

        # Adjust layout to prevent overlapping titles/labels
        fig.suptitle(f"{plotTitle}; Epoch {epoch}", fontsize=24)
        fig.supylabel(f"Angle ({units})", fontsize=20)
        # fig.supxlabel("Sorted parameter index", fontsize=20)
        fig.supxlabel("Parameter index", fontsize=20)
        fig.set_constrained_layout(True)

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", clearFigure=False, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

        # Apply the angular threshold if needed.
        if maxAngularThreshold == degrees: self.clearFigure(fig=None, legend=None, showPlot=False); return None
        for ax in axes.flatten(): ax.set_ylim((-maxAngularThreshold, maxAngularThreshold))

        # Save the plot
        if self.saveDataFolder:  self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} cutoff{str(round(maxAngularThreshold, 4)).replace('.', '-')} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle} cutoff{str(round(maxAngularThreshold, 4)).replace('.', '-')}.pdf", fig=fig, clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotsGivensAnglesHeatmap(self, givensAnglesPath, reversibleModuleNames, signalInd, degreesFlag, epoch, saveFigureLocation, plotTitle):
        # givensAnglesPath: numModuleLayers, numSignals, numAngles
        # maxFreeParamsPath: numModuleLayers
        nRows, nCols = self.getRowsCols(combineSharedLayers=False)
        if not degreesFlag: scaleFactor = 180 / math.pi; degreesFlag = True
        else: scaleFactor = 1

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(4 * nCols, 4 * nRows), squeeze=False, sharex='col', sharey='col')  # squeeze=False ensures axes is 2D
        numLow, numSpecificHigh, numSharedHigh = 0, 0, 0
        degrees = (180 if degreesFlag else math.pi) / 4
        colorbarAxes = []

        # Get the angular thresholds.
        maxAngularThreshold = modelConstants.userInputParams['maxAngularThreshold']
        numSpecific = modelConstants.userInputParams['numSpecificEncoderLayers']
        numShared = modelConstants.userInputParams['numSharedEncoderLayers']

        for layerInd in range(len(givensAnglesPath)):
            moduleName = reversibleModuleNames[layerInd].lower()

            if "low" in moduleName: rowInd, colInd = numLow, nCols - 1; numLow += 1
            elif "high" in moduleName:
                if 'shared' in moduleName: rowInd = numSpecific + (numSharedHigh % numShared); colInd = numSharedHigh // numShared; numSharedHigh += 1
                elif 'specific' in moduleName: rowInd = numSpecificHigh % numSpecific; colInd = numSpecificHigh // numSpecific; numSpecificHigh += 1
                else: raise ValueError("Module name must contain 'shared' or 'specific'")
            else: raise ValueError("Module name must contain 'low' or 'high'.")
            ax = axes[rowInd, colInd]

            # Customize subplot title and axes
            if colInd == 0: ax.set_ylabel(f"Layer {rowInd + 1}", fontsize=16)
            ax.set_title(moduleName.capitalize(), fontsize=16)

            numSignals, numAngles = givensAnglesPath[layerInd].shape
            numSignalsPlotting = min(1, numSignals, len(self.darkColors) - 1)
            sequenceLength = int((1 + (1 + 8 * numAngles) ** 0.5) // 2)
            signalWeightMatrix = np.zeros((sequenceLength, sequenceLength))

            weightMatrix = scaleFactor * givensAnglesPath[layerInd][0:numSignalsPlotting]  # histograms: numSignalsPlotting, numAngles
            rowInds, colInds = np.triu_indices(sequenceLength, k=1)

            # Create the signal weight matrix
            signalWeightMatrix[rowInds, colInds] = -weightMatrix[signalInd]
            signalWeightMatrix[colInds, rowInds] = weightMatrix[signalInd]

            # Plot the heatmap
            colorbarAxes.append(ax.imshow(signalWeightMatrix, cmap=self.custom_cmap, interpolation=None, aspect="equal", vmin=-degrees, vmax=degrees))
        # Adjust layout to prevent overlapping titles/labels
        fig.suptitle(t=f"{plotTitle}; Epoch {epoch}", fontsize=24)
        fig.supylabel(r"$S_{i}$", fontsize=20)
        fig.supxlabel(r"$S_{j}$", fontsize=20)
        fig.colorbar(colorbarAxes[-1], ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04)
        fig.set_constrained_layout(True)

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} cutoff{str(round(maxAngularThreshold, 4)).replace('.', '-')} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle} cutoff{str(round(maxAngularThreshold, 4)).replace('.', '-')}.pdf", fig=fig, clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotScaleFactorLines(self, scalingFactorsPath, reversibleModuleNames, epoch, saveFigureLocation, plotTitle):
        # scalingFactorsPath: numModuleLayers, numSignals
        numModuleLayers = len(reversibleModuleNames)
        sharedValues, specificValues = [], []
        maxCols = 3

        # Get the layer information.
        numSharedScalarSections = 1 + int(math.log2(modelConstants.userInputParams['encodedDimension'] // modelConstants.userInputParams['minWaveletDim']))
        numSpecificLayers, numSharedLayers = modelConstants.userInputParams['numSpecificEncoderLayers'], modelConstants.userInputParams['numSharedEncoderLayers']
        numSpecificScalarSections = 2

        # Determine the number of rows and columns for the plot.
        nCols = min(max(numSpecificLayers, numSharedLayers), maxCols)
        numSpecificRows = max(1, numSpecificLayers // nCols + (1 if numSpecificLayers % nCols != 0 else 0))
        nRows = numSpecificRows + max(1, numSharedLayers // nCols + (1 if numSharedLayers % nCols != 0 else 0))
        extraSpecificAxes = numSpecificRows*nCols - numSpecificLayers

        xTickLabelShared = []
        xTickLabelSpecific = ["Detailed decomposition layer 1", "Approximate decomposition layer 1"]
        for decompositionInd in range(numSharedScalarSections - 1): xTickLabelShared.append(f"Detailed decomposition layer {decompositionInd + 1}")
        xTickLabelShared.append(f"Approximate decomposition layer {numSharedScalarSections - 1}")

        for layerInd in range(numModuleLayers):
            if "shared" in reversibleModuleNames[layerInd].lower(): sharedValues.append(scalingFactorsPath[layerInd].flatten())
            elif "specific" in reversibleModuleNames[layerInd].lower(): specificValues.append(scalingFactorsPath[layerInd].flatten())
            else: raise ValueError("Module name must contain 'specific' or 'shared'.")
        sharedValues = np.asarray(sharedValues); specificValues = np.asarray(specificValues)
        # sharedValues: numSharedLayers=numSections*y, numSignals=1; specificValues: numSpecificLayers=numSections*x, numSignals=numSignals
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex=False, sharey=False)
        fig.suptitle(f"{plotTitle}; Epoch {epoch}", fontsize=24)
        axes = axes.flatten()

        for axInd, ax in enumerate(axes):
            rowInd, colInd = axInd // nCols, axInd % nCols
            specificFlag = axInd < numSpecificLayers
            if not specificFlag: axInd -= numSpecificLayers + extraSpecificAxes
            if not specificFlag and axInd < 0: ax.remove(); continue

            # Customize plot title and axes
            ax.set_ylabel(f"{"Specific" if specificFlag else "Shared"} normalization factors")  # Y-axis: bin counts
            ax.set_ylim((0.925, 1.075))

            if numSharedLayers <= axInd: ax.remove(); continue
            # Get the angles for the current layer
            ax.set_title(f"Specific layer: {axInd+1}" if specificFlag else f"Shared layer: {axInd+1}", fontsize=16)
            if specificFlag: ax.plot(specificValues[axInd:numSpecificScalarSections*(axInd+1)], 'o', color=self.darkColors[0], alpha=0.5, linewidth=1, markersize=4)
            else: ax.plot(sharedValues[axInd::numSharedLayers], 'o', color=self.darkColors[1], alpha=0.75, linewidth=1, markersize=4)

            ax.set_xticks(range(len(xTickLabelSpecific if specificFlag else xTickLabelShared)))  # Set x-ticks positions
            ax.set_xticklabels(xTickLabelSpecific if specificFlag else xTickLabelShared, rotation=45, ha='right')  # Set x-tick labels with rotation

        # Save the plot
        fig.tight_layout(pad=2.0)
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", fig=fig, clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotScaleFactorHist(self, scalingFactorsPath, reversibleModuleNames, epoch, saveFigureLocation, plotTitle):
        # scalingFactorsPath: numModuleLayers, numSignals, numParams=1
        sharedValues, specificValues = [], []
        for layerInd in range(len(scalingFactorsPath)):
            if "shared" in reversibleModuleNames[layerInd].lower(): sharedValues.extend(scalingFactorsPath[layerInd].flatten())
            elif "specific" in reversibleModuleNames[layerInd].lower(): specificValues.extend(scalingFactorsPath[layerInd].flatten())
            else: raise ValueError("Module name must contain 'specific' or 'shared'.")

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.hist(sharedValues, bins=10, color=self.lightColors[1], alpha=0.7, label="Shared", density=True, align='left')
        ax.hist(specificValues, bins=10, color=self.lightColors[0], alpha=0.7, label="Specific", density=True, align='left')

        # Customize plot title and axes
        ax.set_title(f"{plotTitle}; Epoch {epoch}", fontsize=16)
        ax.set_xlabel("Scale factor")  # X-axis: values
        ax.set_ylabel("Frequency")  # Y-axis: bin counts
        ax.set_xlim((0.925, 1.075))
        ax.set_ylim((0, None))
        ax.legend()

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", fig=fig, clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def modelFlow(self, dataTimes, dataStatesAll, signalNames, batchInd, epoch, saveFigureLocation, plotTitle):
        for signalInd in range(len(signalNames)):
            dataStates = np.asarray(dataStatesAll[:, batchInd, signalInd, :])
            numModelLayers, encodedDimension = dataStates.shape
            # dataStates: numModelLayers, encodedDimension

            # Create a meshgrid for encodedDimension and numModelLayers
            x_data, y_data = np.meshgrid(dataTimes, np.flip(np.arange(1, 1 + numModelLayers), axis=-1))
            x, y, z = x_data.flatten(), np.flip(y_data.flatten()), dataStates.flatten()

            # Figure and axis settings
            fig = plt.figure(figsize=(12, 8), facecolor="white")
            ax = fig.add_subplot(111, projection='3d', facecolor="white")

            # Improved scatter points
            ax.scatter(
                x, y, z, c=z,
                cmap='viridis', edgecolors=self.blackColor, linewidth=0.5,
                alpha=0.95, s=20, vmin=-1.5, vmax=1.5)
            for modelLayerInd in range(numModelLayers):
                ax.plot(x[modelLayerInd*encodedDimension:(modelLayerInd + 1)*encodedDimension], y[modelLayerInd*encodedDimension:(modelLayerInd + 1)*encodedDimension], z[modelLayerInd*encodedDimension:(modelLayerInd + 1)*encodedDimension],
                        color=self.blackColor, linestyle='-', linewidth=0.5, alpha=0.5)

            # View and perspective adjustments
            ax.view_init(elev=25, azim=135)
            ax.dist = 8  # Adjusts perspective depth

            # Axis labels and title
            ax.set_title(f"{plotTitle}; Epoch {epoch}", fontsize=16, weight='bold', pad=20)
            ax.set_xlabel("Time (Sec)", fontsize=12, labelpad=10)
            ax.set_ylabel("Model Layer", fontsize=12, labelpad=10)
            ax.set_zlabel("Signal value (au)", fontsize=12, labelpad=10)
            ax.set_zlim(-1.75, 1.75)
            ax.invert_yaxis()

            # Make the aspect ratio look nicer in 3D
            ax.set_box_aspect([2, 1, 1])
            fig.tight_layout()

            # Save the plot
            if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} {signalNames[signalInd]} batchInd{batchInd} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle} {signalNames[signalInd]} batchInd{batchInd}.pdf", fig=fig, clearFigure=True, showPlot=False)
            else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotActivationCurvesCompressed(self, activationCurves, moduleNames, epoch, saveFigureLocation, plotTitle):
        axNames = ["Specific neural low frequency", "Specific neural high frequency",
                   "Shared neural low frequency", "Shared neural high frequency"]
        numActivations, numPointsX, numPointsY = activationCurves.shape
        nCols, nRows = 2, 2

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex=True, sharey=True)
        numSpecificActivations, numSharedActivations = 0, 0
        axes = axes.flatten()

        for activationInd in range(numActivations):
            x, y = activationCurves[activationInd]
            activationName = moduleNames[activationInd].lower()

            if "specific" in activationName: axInd = 0; numSpecificActivations += 1; totalActivations = numSpecificActivations
            elif "shared" in activationName: axInd = 2; numSharedActivations += 1; totalActivations = numSharedActivations
            else: raise ValueError(f"Unknown activation module: {activationName}")

            if 'low' in activationName: axInd += 0
            elif 'high' in activationName: axInd += 1
            else: raise ValueError(f"Unknown activation module: {activationName}")

            ax = axes[axInd]
            # Plot the activation curves
            ax.plot(x, y, color=self.lightColors[1], linestyle='-', linewidth=1, label="Inverse Pass", alpha=0.5*totalActivations/numActivations + 0.5)  # Plot Inverse Pass
            ax.plot(y, x, color=self.lightColors[0], linestyle='-', linewidth=1, label="Forward Pass", alpha=0.5*totalActivations/numActivations + 0.5)  # Plot Forward Pass

            ax = axes[axInd]
            ax.plot(x, x, color=self.blackColor, linestyle='--', linewidth=0.5)  # Plot Identity Line
            ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.8)
            ax.set_title(f"{axNames[axInd]}")

        # Set the main title
        fig.suptitle(f"{plotTitle} - Epoch {epoch}\nForward and Inverse from x ∈ [{-1.5}, {1.5}]", fontsize=24)
        fig.supylabel("Output (Y)", fontsize=20)
        fig.supxlabel("Input (x)", fontsize=20)
        fig.set_constrained_layout(True)

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", fig=fig, clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)

    def plotActivationCurves(self, activationCurves, moduleNames, epoch, saveFigureLocation, plotTitle):
        numModuleLayers, numPointsX, numPointsY = activationCurves.shape
        nRows, nCols = self.getRowsCols(combineSharedLayers=False)

        # Create a figure and axes array
        fig, axes = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6.4 * nCols, 4.8 * nRows), squeeze=False, sharex=True, sharey=True)
        numLow, numSpecificHigh, numSharedHigh = 0, 0, 0

        # Get the parameters.
        numSpecific = modelConstants.userInputParams['numSpecificEncoderLayers']
        numShared = modelConstants.userInputParams['numSharedEncoderLayers']

        for layerInd in range(numModuleLayers):
            moduleName = moduleNames[layerInd].lower()
            x, y = activationCurves[layerInd]

            if "low" in moduleName: rowInd, colInd = numLow, nCols - 1; numLow += 1
            elif "high" in moduleName:
                if 'shared' in moduleName: rowInd = numSpecific + (numSharedHigh % numShared); colInd = numSharedHigh // numShared; numSharedHigh += 1
                elif 'specific' in moduleName: rowInd = numSpecificHigh % numSpecific; colInd = numSpecificHigh // numSpecific; numSpecificHigh += 1
                else: raise ValueError("Module name must contain 'shared' or 'specific'")
            else: raise ValueError("Module name must contain 'low' or 'high'.")
            ax = axes[rowInd, colInd]

            # Customize subplot title and axes
            if colInd == 0: ax.set_ylabel(f"Layer {rowInd + 1}", fontsize=16)
            ax.set_title(moduleName.capitalize(), fontsize=16)
            ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.8)

            # Plot the activation curves
            ax.plot(x, y, color=self.lightColors[1], linestyle='-', linewidth=1, label="Inverse Pass", alpha=1)  # Plot Inverse Pass
            ax.plot(y, x, color=self.lightColors[0], linestyle='-', linewidth=1, label="Forward Pass", alpha=1)  # Plot Forward Pass
            ax.plot(x, x, color=self.blackColor, linestyle='--', linewidth=0.5)  # Plot Identity Line

        # Set the main title
        fig.suptitle(f"{plotTitle} - Epoch {epoch}\nForward and Inverse from x ∈ [{-1.5}, {1.5}]", fontsize=24)
        fig.supylabel("Output (Y)", fontsize=20)
        fig.supxlabel("Input (x)", fontsize=20)
        fig.set_constrained_layout(True)

        # Save the plot
        if self.saveDataFolder: self.displayFigure(saveFigureLocation=saveFigureLocation, saveFigureName=f"{plotTitle} epochs{epoch}.pdf", baseSaveFigureName=f"{plotTitle}.pdf", fig=fig, clearFigure=True, showPlot=False)
        else: self.clearFigure(fig=fig, legend=None, showPlot=True)
