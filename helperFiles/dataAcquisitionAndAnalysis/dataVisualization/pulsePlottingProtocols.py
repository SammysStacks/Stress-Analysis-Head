
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import math
import numpy as np
# Matlab Plotting API
import matplotlib as mpl
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------- #
# ------------------------ Plotting Pulse Protocols ------------------------ #

class pulsePlottingProtocols:
    
    def __init__(self):
        self.sectionColors = ['red','orange', 'blue','green', 'black']
    
    def plotData(self, xData, yData, title, ax = None, axisLimits = [], topPeaks = {}, bottomPeaks = {}, peakSize = 3, lineWidth = 2, lineColor = "tab:blue", pulsePeakInds = []):
        # Create Figure
        showFig = False
        if ax == None:
            plt.figure()
            ax = plt.gca()
            showFig = True
        # Plot the Data
        ax.plot(xData, yData, linewidth = lineWidth, color = lineColor)
        if topPeaks:
            ax.plot(topPeaks[1], topPeaks[2], 'or', markersize=peakSize)
        if bottomPeaks:
            ax.plot(bottomPeaks[1], bottomPeaks[2], 'ob', markersize=peakSize)
        if len(pulsePeakInds) > 0:
            for groupInd in range(len(self.sectionColors)):
                if pulsePeakInds[groupInd] in [np.nan, None] or pulsePeakInds[groupInd+1] in [np.nan, None]: 
                    continue
                ax.fill_between(xData[pulsePeakInds[groupInd]:pulsePeakInds[groupInd+1]+1], min(yData), yData[pulsePeakInds[groupInd]:pulsePeakInds[groupInd+1]+1], color=self.sectionColors[groupInd], alpha=0.15)
        # Add Axis Labels and Figure Title
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Capacitance (pF)")
        ax.set_title(title)
        # Change Axis Limits If Given
        if axisLimits:
            ax.set_xlim(axisLimits)
        # Increase DPI (Clarity); Will Slow Down Plotting
        mpl.rcParams['figure.dpi'] = 300
        # Show the Plot
        if showFig:
            plt.show()
    

    def plotPulses(self, bloodPulse, numSubPlotsX = 3, firstPeakPlotting = 1, maxPulsesPlot = 9, figWidth = 25, figHeight = 13, finalPlot = False):
        # Create One Plot with First 9 Pulse Curves
        numSubPlots = min(maxPulsesPlot, len(bloodPulse) - firstPeakPlotting + 1)
        scaleGraph = math.ceil(numSubPlots/numSubPlotsX) / (maxPulsesPlot/numSubPlotsX)
        figHeight = int(figHeight*scaleGraph); figWidth = int(figWidth*min(numSubPlots,numSubPlotsX)/numSubPlotsX)
        
        fig, ax = plt.subplots(math.ceil(numSubPlots/numSubPlotsX), min(numSubPlotsX, numSubPlots), sharey=False, sharex = False, figsize=(figWidth,figHeight))
        fig.suptitle("Indivisual Pulse Peaks", fontsize=20, fontweight ="bold", yData=0.98)
        for figNum, pulseNum in enumerate(list(bloodPulse.keys())[firstPeakPlotting-1:]):
            if figNum == numSubPlots:
                break
            # Keep Running Order of Subplots
            if numSubPlots == 1:
                currentAxes = ax
            elif numSubPlots <= numSubPlotsX:
                currentAxes = ax[figNum]
            else:
                currentAxes = ax[figNum//numSubPlotsX][figNum%numSubPlotsX]
            # Get the Data
            time = bloodPulse[pulseNum]['time']
            filterData = bloodPulse[pulseNum]["normalizedPulse"]
            # Get the Pulse peaks
            bottomInd = []
            pulsePeakInds = bloodPulse[pulseNum]['indicesTop']
            # Plot with Pulses Sectioned Off into Regions
            if finalPlot:
                pulsePeakInds = bloodPulse[pulseNum]['pulsePeakInds']
                self.plotData(time, filterData, title = "Peak Number " + str(pulseNum), ax = currentAxes, topPeaks = {1:time[pulsePeakInds], 2:filterData[pulsePeakInds]}, bottomPeaks = {1:time[bottomInd], 2:filterData[bottomInd]}, peakSize = 5,  lineWidth = 2, lineColor = "black", pulsePeakInds = pulsePeakInds)
            # General Plot
            else:
                # Plot the Data 
                self.plotData(time, filterData, title = "Peak Number " + str(pulseNum), ax = currentAxes, topPeaks = {1:time[pulsePeakInds], 2:filterData[pulsePeakInds]}, bottomPeaks = {1:time[bottomInd], 2:filterData[bottomInd]}, peakSize = 5, lineWidth = 2, lineColor = "black")
        fig.tight_layout(pad= 2.0)
        plt.show()
    
    
    def plotPulseNum(self, bloodPulse, pulseNum, finalPlot = False):
        # Get Data
        time = bloodPulse[pulseNum]['time']
        filterData = bloodPulse[pulseNum]["normalizedPulse"]
        # Get the Pulse peaks 
        bottomInd = []
        pulsePeakInds = bloodPulse[pulseNum]['indicesTop']
        # Plot with Pulses Sectioned Off into Regions
        if finalPlot:
            pulsePeakInds = bloodPulse[pulseNum]['pulsePeakInds']
            self.plotData(time, filterData, title = "Peak Number " + str(pulseNum), topPeaks = {1:time[pulsePeakInds], 2:filterData[pulsePeakInds]}, bottomPeaks = {1:time[bottomInd], 2:filterData[bottomInd]}, peakSize = 5,  lineWidth = 2, lineColor = "black", pulsePeakInds = pulsePeakInds)
        else:
            self.plotData(time, filterData, title = "Peak Number " + str(pulseNum), topPeaks = {1:time[pulsePeakInds], 2:filterData[pulsePeakInds]}, bottomPeaks = {1:time[bottomInd], 2:filterData[bottomInd]}, peakSize = 3, lineWidth = 2, lineColor = "black")
        
    def plotPulseInfo(self, pulseTime, pulseData, pulseVelocity, pulseAcceleration, thirdDeriv, allSystolicPeaks, allTidalPeaks, allDicroticPeaks):
        from matplotlib.ticker import MaxNLocator # added 

        # use ggplot style for more sophisticated visuals
        plt.style.use('seaborn-poster')
        
        # Specify Figure aesthetics
        figWidth = 23; figHeight = 18;
        fig, axes = plt.subplots(4, 1, sharey=False, sharex = True, gridspec_kw={'hspace': 0},
                                     figsize=(figWidth, figHeight))
        
        # Plot the Data
        axes[0].plot(pulseTime, pulseData, 'k', linewidth=2)
        axes[1].plot(pulseTime, pulseVelocity, 'tab:blue', linewidth=2)
        axes[2].plot(pulseTime, pulseAcceleration, 'tab:red', linewidth=2)
        axes[3].plot(pulseTime, thirdDeriv, 'tab:green', linewidth=2)
        # Set the y-label
        axes[0].set_ylabel("Normalized Pulse")
        axes[1].set_ylabel("Normalized $1^{rst}$ Derivative")
        axes[2].set_ylabel("Normalized $2^{nd}$ Derivative")
        axes[3].set_ylabel("Normalized $3^{rd}$ Derivative")
        
        # Split up the indices
        pulseIndices = [allSystolicPeaks[3], allDicroticPeaks[0], allDicroticPeaks[2]]
        velIndices = [allSystolicPeaks[1], allTidalPeaks[0], allDicroticPeaks[1], allDicroticPeaks[3]]
        accelIndices = [allSystolicPeaks[0], allSystolicPeaks[2]]
        thirdDerivInds = [allTidalPeaks[1]]
        # Add the Points to the Pulse plot
        axes[0].plot(pulseTime[pulseIndices], pulseData[pulseIndices], 'ok', markersize=13)
        axes[0].plot(pulseTime[velIndices], pulseData[velIndices], 'ok', markersize=13)
        axes[0].plot(pulseTime[accelIndices],  pulseData[accelIndices], 'ok', markersize=13)
        axes[0].plot(pulseTime[thirdDerivInds],  pulseData[thirdDerivInds], 'ok', markersize=13)
        ymin, ymax = axes[0].get_ylim()
        axes[0].vlines(x=pulseTime[velIndices], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:blue')
        axes[0].vlines(x=pulseTime[accelIndices], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:red')
        axes[0].vlines(x=pulseTime[thirdDerivInds], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:green')
        axes[0].set_ylim((ymin, ymax))
        axes[0].set_xlim((pulseTime[0], pulseTime[-1]))
        # Add the Points to the Velocity plot
        axes[1].plot(pulseTime[velIndices], pulseVelocity[velIndices], 'ok', markersize=13)
        ymin, ymax = axes[1].get_ylim()
        axes[1].vlines(x=pulseTime[velIndices], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:blue')
        axes[1].hlines(y=0, xmin=pulseTime[0], xmax=pulseTime[allTidalPeaks[0]], color='k')
        
        axes[1].vlines(x=pulseTime[allTidalPeaks[0]], ymin=ymin, ymax=pulseVelocity[allTidalPeaks[0]], color='k')
        axes[1].set_ylim((ymin, ymax))
        axes[1].set_xlim((pulseTime[0], pulseTime[-1]))
        # Add the Points to the Acceleration plot
        axes[2].plot(pulseTime[accelIndices],  pulseAcceleration[accelIndices], 'ok', markersize=13)
        ymin, ymax = axes[2].get_ylim()
        axes[2].vlines(x=pulseTime[accelIndices], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:red')
        axes[2].set_ylim((ymin, ymax))   
        axes[2].set_xlim((pulseTime[0], pulseTime[-1]))
        # Add the thirdDeriv to the Pulse plot
        axes[3].plot(pulseTime[thirdDerivInds],  thirdDeriv[thirdDerivInds], 'ok', markersize=13)
        ymin, ymax = axes[3].get_ylim()
        axes[3].vlines(x=pulseTime[thirdDerivInds], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:green')
        # axes[0].vlines(x=pulseTime[velIndices], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:blue')
        # axes[0].vlines(x=pulseTime[accelIndices], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:red')
        axes[3].set_ylim((ymin, ymax))
        axes[3].set_xlim((pulseTime[0], pulseTime[-1]))
        
        # Create surrounding figure
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)

        # Add figure labels
        plt.suptitle('Pulse Peaks Extraction', fontsize = 22, x = 0.525, fontweight = "bold")
        plt.xlabel("Time (Seconds)", labelpad = 15)
                
        # Remove overlap in yTicks
        nbins = len(axes[0].get_yticklabels())
        for ax in axes:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='both'))  
            # Move exponent
            tickExp = ax.yaxis.get_offset_text()
            tickExp.set_y(-0.5)

        # Finalize figure spacing
        plt.tight_layout()
        plt.show()
        
    def plotPulseInfo_Amps(self, pulseTime, pulseData, pulseVelocity, pulseAcceleration, thirdDeriv, allSystolicPeaks, allTidalPeaks, allDicroticPeaks, tidalVelocity_ZeroCrossings, tidalAccel_ZeroCrossings):
        from matplotlib.ticker import MaxNLocator # added 

        # use ggplot style for more sophisticated visuals
        plt.style.use('seaborn-poster')
        
        # Specify Figure aesthetics
        figWidth = 23; figHeight = 18;
        fig, axes = plt.subplots(4, 1, sharey=False, sharex = True, gridspec_kw={'hspace': 0},
                                     figsize=(figWidth, figHeight))
        
        # Plot the Data
        axes[0].plot(pulseTime, pulseData, 'k', linewidth=2)
        axes[1].plot(pulseTime, pulseVelocity, 'tab:blue', linewidth=2)
        axes[2].plot(pulseTime, pulseAcceleration, 'tab:red', linewidth=2)
        axes[3].plot(pulseTime, thirdDeriv, 'tab:green', linewidth=2)
        # Set the y-label
        axes[0].set_ylabel("Normalized Pulse")
        axes[1].set_ylabel("Normalized $1^{rst}$ Derivative")
        axes[2].set_ylabel("Normalized $2^{nd}$ Derivative")
        axes[3].set_ylabel("Normalized $3^{rd}$ Derivative")
        
        # Split up the indices
        pulseIndices = [allSystolicPeaks[3], allTidalPeaks[0], allDicroticPeaks[0], allDicroticPeaks[2]]
        velIndices = [allTidalPeaks[0]]
        accelIndices = [allTidalPeaks[0]]
        thirdDerivInds = [allTidalPeaks[0]]
        # Add the Points to the Pulse plot
        axes[0].plot(pulseTime[pulseIndices], pulseData[pulseIndices], 'ok', markersize=13)
        ymin, ymax = axes[0].get_ylim()
        # axes[0].vlines(x=pulseTime[velIndices], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:blue')
        # axes[0].vlines(x=pulseTime[accelIndices], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:red')
        axes[0].set_ylim((ymin, ymax))
        axes[0].set_xlim((pulseTime[0], pulseTime[-1]))
        # Add the Points to the Velocity plot
        axes[1].plot(pulseTime[velIndices], pulseVelocity[velIndices], 'ok', markersize=13)
        ymin, ymax = axes[1].get_ylim()
        axes[1].vlines(x=pulseTime[velIndices], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:blue')
        if len(tidalVelocity_ZeroCrossings) != 0:
            axes[1].hlines(y=0, xmin=pulseTime[0], xmax=pulseTime[tidalVelocity_ZeroCrossings[-1]+1], color='k')
            axes[1].vlines(x=pulseTime[tidalVelocity_ZeroCrossings[-1]+1], ymin=ymin, ymax=pulseVelocity[tidalVelocity_ZeroCrossings[-1]+1], color='k')
        axes[1].set_ylim((ymin, ymax))
        axes[1].set_xlim((pulseTime[0], pulseTime[-1]))
        # Add the Points to the Acceleration plot
        axes[2].plot(pulseTime[accelIndices],  pulseAcceleration[accelIndices], 'ok', markersize=13)
        ymin, ymax = axes[2].get_ylim()
        axes[2].vlines(x=pulseTime[accelIndices], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:red')
        if len(tidalAccel_ZeroCrossings) != 0:
            axes[2].hlines(y=0, xmin=pulseTime[0], xmax=pulseTime[tidalAccel_ZeroCrossings[-1]+1], color='k')
            axes[2].vlines(x=pulseTime[tidalAccel_ZeroCrossings[-1]+1], ymin=ymin, ymax=pulseAcceleration[tidalAccel_ZeroCrossings[-1]+1], color='k')
        axes[2].set_ylim((ymin, ymax))   
        axes[2].set_xlim((pulseTime[0], pulseTime[-1]))
        # Add the thirdDeriv to the Pulse plot
        axes[3].plot(pulseTime[thirdDerivInds],  thirdDeriv[thirdDerivInds], 'ok', markersize=13)
        ymin, ymax = axes[3].get_ylim()
        axes[3].vlines(x=pulseTime[thirdDerivInds], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:green')
        # axes[0].vlines(x=pulseTime[velIndices], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:blue')
        # axes[0].vlines(x=pulseTime[accelIndices], ymin=ymin, ymax=ymax, linestyles='dashed', colors='tab:red')
        axes[3].set_ylim((ymin, ymax))
        axes[3].set_xlim((pulseTime[0], pulseTime[-1]))
            
        # Create surrounding figure
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)

        # Add figure labels
        plt.suptitle('Tidal Peak Extraction', fontsize = 22, x = 0.525, fontweight = "bold")
        plt.xlabel("Time (Seconds)", labelpad = 15)
                
        # Remove overlap in yTicks
        nbins = len(axes[0].get_yticklabels())
        for ax in axes:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='both'))  
            # Move exponent
            tickExp = ax.yaxis.get_offset_text()
            tickExp.set_y(-0.5)

        # Finalize figure spacing
        plt.tight_layout()
        plt.show()
        
# ---------------------------------------------------------------------------#
# ---------------------------------------------------------------------------#    
    
    