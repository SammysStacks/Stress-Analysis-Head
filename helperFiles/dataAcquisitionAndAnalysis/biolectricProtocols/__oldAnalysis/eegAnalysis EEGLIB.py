
# Basic Modules
import scipy
import numpy as np
# Feature Extraction Modules
import eeglib
# Matlab Plotting Modules
import matplotlib.pyplot as plt

# Import Files
import _filteringProtocols # Import files with filtering methods
import _universalProtocols # Import files with general analysis methods
        
# ---------------------------------------------------------------------------#
# ---------------------------------------------------------------------------#

class eegProtocol:
    
    def __init__(self, numPointsPerBatch = 3000, moveDataFinger = 10, numChannels = 2, plottingClass = None, readData = None):
        # Input Parameters
        self.numChannels = numChannels            # Number of Bioelectric Signals
        self.numPointsPerBatch = numPointsPerBatch        # The X-Wdith of the Plot (Number of Data-Points Shown)
        self.moveDataFinger = moveDataFinger      # The Amount of Data to Stream in Before Finding Peaks
        self.plotStreamedData = plottingClass != None  # Plot the Data
        self.plottingClass = plottingClass
        self.collectFeatures = False  # This flag will be changed by user if desired (NOT here).
        self.featureWindowSeconds = 15 # The number of seconds to analyze features for each point. 5-15
        self.featureAverageWindow = None # The number of seconds before each feature to avaerage the features together.
        self.readData = readData

        # Define general classes to process data.
        self.filteringMethods = _filteringProtocols.filteringMethods()
        self.universalMethods = _universalProtocols.universalMethods()
        # High Pass Filter Parameters
        self.dataPointBuffer = 5000        # A Prepended Buffer in the Filtered Data that Represents BAD Filtering; Units: Points
        self.cutOffFreq = [1, 50]         # Optimal LPF Cutoff in Literatrue is 6-8 or 20 Hz (Max 35 or 50); I Found 25 Hz was the Best, but can go to 15 if noisy (small amplitude cutoff)
        
        # General Parameters
        self.featureNames = None
        
        # Prepare the Program to Begin Data Analysis
        self.checkParams()              # Check to See if the User's Input Parameters Make Sense
        self.resetGlobalVariables()     # Start with Fresh Inputs (Clear All Arrays/Values)
        
        # If Plotting, Define Class for Plotting Peaks
        if self.plotStreamedData and numChannels != 0:
            self.initPlotPeaks()
        
    def resetGlobalVariables(self):
        # Data to Read in
        self.data = [ [], [[] for channel in range(self.numChannels)] ]
        # Reset Feature Extraction
        self.rawFeatures = []           # Raw features extraction at the current timepoint.
        self.featureTimes = []          # The time of each feature.
        self.compiledFeatures = []      # FINAL compiled features at the current timepoint. Could be average of last x features.
        
        # General parameters
        self.samplingFreq = None        # The Average Number of Points Steamed Into the Arduino Per Second; Depends on the User's Hardware; If NONE Given, Algorithm will Calculate Based on Initial Data
        self.featureTimeWindow = None
        self.lastAnalyzedDataInd = 0    # The index of the last point analyzed  
        
        # Close Any Opened Plots
        if self.plotStreamedData:
            plt.close()
            
    def checkParams(self):
        assert self.moveDataFinger < self.numPointsPerBatch, "You are Analyzing Too Much Data in a Batch. 'moveDataFinger' MUST be Less than 'numPointsPerBatch'"

    def setSamplingFrequency(self, startBPFindex):
        # Caluclate the Sampling Frequency
        self.samplingFreq = len(self.data[0][startBPFindex:])/(self.data[0][-1] - self.data[0][startBPFindex])
        print("\n\tSetting EEG Sampling Frequency to", self.samplingFreq)
        print("\tFor Your Reference, If Data Analysis is Longer Than", self.moveDataFinger/self.samplingFreq, ", Then You Will NOT be Analyzing in Real Time")
        
        # Set Parameters
        self.featureTimeWindow = int(self.samplingFreq*self.featureWindowSeconds) # Should be beteen 3-10 seconds
        
    def initPlotPeaks(self): 
        # Establish pointers to the figure
        self.fig = self.plottingClass.fig
        axes = self.plottingClass.axes['eeg'][0]

        # Plot the Raw Data
        yLimLow = 0; yLimHigh = 3.5; 
        self.bioelectricDataPlots = []; self.bioelectricPlotAxes = []
        for channelIndex in range(self.numChannels):
            # Create Plots
            if self.numChannels == 1:
                self.bioelectricPlotAxes.append(axes[0])
            else:
                self.bioelectricPlotAxes.append(axes[channelIndex, 0])
            
            # Generate Plot
            self.bioelectricDataPlots.append(self.bioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:green", linewidth=1, alpha = 0.65)[0])
            
            # Set Figure Limits
            self.bioelectricPlotAxes[channelIndex].set_ylim(yLimLow, yLimHigh)
            # Label Axis + Add Title
            self.bioelectricPlotAxes[channelIndex].set_ylabel("EEG Signal (Volts)", fontsize=13, labelpad = 10)
            
        # Create the Data Plots
        self.filteredBioelectricDataPlots = []
        self.filteredBioelectricPlotAxes = [] 
        for channelIndex in range(self.numChannels):
            # Create Plot Axes
            if self.numChannels == 1:
                self.filteredBioelectricPlotAxes.append(axes[1])
            else:
                self.filteredBioelectricPlotAxes.append(axes[channelIndex, 1])
            
            # Plot Flitered Peaks
            self.filteredBioelectricDataPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="tab:green", linewidth=1, alpha = 0.65)[0])

            # Set Figure Limits
            self.filteredBioelectricPlotAxes[channelIndex].set_ylim(yLimLow, yLimHigh)
            
        # Tighten figure's white space (must be at the end)
        self.plottingClass.fig.tight_layout(pad=2.0);
        
    # ----------------------------------------------------------------------- #
    # ------------------------- Data Analysis Begins ------------------------ #

    def analyzeData(self, dataFinger):
        
        # Add incoming Data to Each Respective Channel's Plot
        for channelIndex in range(self.numChannels):
            
            # ---------------------- Filter the Data ----------------------- #    
            # Band Pass Filter to Remove Noise
            startBPFindex = max(dataFinger - self.dataPointBuffer, 0)
            yDataBuffer = self.data[1][channelIndex][startBPFindex:dataFinger + self.numPointsPerBatch].copy()
            
            # Get the Sampling Frequency from the First Batch (If Not Given)
            if not self.samplingFreq:
                self.setSamplingFrequency(startBPFindex)

            # Filter the Data: Low pass Filter and Savgol Filter
            filteredData = self.filteringMethods.bandPassFilter.butterFilter(yDataBuffer, self.cutOffFreq[1], self.samplingFreq, order = 3, filterType = 'low')
            filteredData = scipy.signal.savgol_filter(filteredData, 21, 2, mode='nearest', deriv=0)[-self.numPointsPerBatch:]

            # -------------------------------------------------------------- #
            
            # ---------------------- EEG Preprocessing --------------------- #  
            if self.collectFeatures:
                newData = filteredData[max(0,self.lastAnalyzedDataInd - dataFinger):]
                if self.featureTimeWindow < len(newData):
                    helper = eeglib.helpers.Helper(np.array([newData]), sampleRate=self.samplingFreq, windowSize=self.featureTimeWindow,
                                  highpass=1, lowpass=50, normalize=True, ICA=False, selectedSignals=None, names=None)
                    # ---------------------------------------------------------- #   
    
                    # ------------------- Feature Extraction ------------------- #  
                    # Add wrapper to get features
                    wrap = eeglib.wrapper.Wrapper(helper)
                    
                    # Add features to extract
                    wrap.addFeature.DFA(i = 0, fit_degree = 1)
                    wrap.addFeature.HFD(i = 0)
                    wrap.addFeature.LZC(i = 0)
                    wrap.addFeature.PFD(i = 0)
                    # Add features to extract
                    wrap.addFeature.engagementLevel()
                    wrap.addFeature.sampEn(i = 0, m = 2, l = 1)
                    wrap.addFeature.bandPower(i = 0, normalize = True)
                    # Add Hjorth parameters
                    wrap.addFeature.hjorthActivity(i = 0)
                    wrap.addFeature.hjorthComplexity(i = 0)
                    wrap.addFeature.hjorthMobility(i = 0)
                    
                    # Extract the features
                    newFeatures = wrap.getAllFeatures()
                    if self.featureNames == None:
                        self.featureNames = [featureName.split('(')[0] + featureName.split('}')[-1] for featureName in newFeatures.columns]
                    # Calculate the feature times
                    relativeFeatureIndices = np.arange(self.featureTimeWindow/2, len(newData), self.featureTimeWindow)[0:len(newFeatures)]
                    if len(self.featureTimes) != 0:
                        relativeFeatureTimes = (relativeFeatureIndices + self.featureTimeWindow/2)/self.samplingFreq
                        relativeFeatureTimes += self.featureTimes[-1]
                    else:
                        relativeFeatureTimes = relativeFeatureIndices/self.samplingFreq
                    # Keep track of which data has been analyzed 
                    self.lastAnalyzedDataInd += int(relativeFeatureIndices[-1] + self.featureTimeWindow/2)
                        
                    # Keep track of the new features
                    self.readData.averageFeatures(relativeFeatureTimes, newFeatures.values, self.featureTimes, self.rawFeatures, self.compiledFeatures, self.featureAverageWindow)
            # -------------------------------------------------------------- #   
                    self.filteredData = wrap.helper.eeg.window.window[0]
        
            # ------------------- Plot Biolectric Signals ------------------ #
            if self.plotStreamedData:
                # Get X Data: Shared Axis for All Channels
                timePoints = np.array(self.data[0][dataFinger:dataFinger + self.numPointsPerBatch])
    
                # Get New Y Data
                newYData = self.data[1][channelIndex][dataFinger:dataFinger + self.numPointsPerBatch]
                # Plot Raw Bioelectric Data (Slide Window as Points Stream in)
                self.bioelectricDataPlots[channelIndex].set_data(timePoints, newYData)
                self.bioelectricPlotAxes[channelIndex].set_xlim(timePoints[0], timePoints[-1])
                            
                # Plot the Filtered + Digitized Data
                self.filteredBioelectricDataPlots[channelIndex].set_data(timePoints, filteredData[-len(timePoints):])
                self.filteredBioelectricPlotAxes[channelIndex].set_xlim(timePoints[0], timePoints[-1]) 
            # -------------------------------------------------------------- #   

    
    
    
    
    