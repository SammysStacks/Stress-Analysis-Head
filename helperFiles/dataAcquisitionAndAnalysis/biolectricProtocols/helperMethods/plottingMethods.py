
# --------------------------- Global Model Class --------------------------- #

class plottingMethods:
    
    def __init__(self, analysisType, numChannels, plottingClass):
        # General input parameters
        self.numChannels = numChannels
        self.analysisType = analysisType
        self.plottingClass = plottingClass
    
    def initPlotPeaks(self):
        # Establish pointers to the figure
        self.fig = self.plottingClass.fig
        axes = self.plottingClass.axes[self.analysisType][0]
        
        # Initialize the correct type of plot.
        if self.analysisType =="eog":
            self.initPlotPeaks_EOG(axes)
        elif self.analysisType =="eeg":
            self.initPlotPeaks_EEG(axes)
        elif self.analysisType =="eda":
            self.initPlotPeaks_EDA(axes)
        elif self.analysisType =="emg":
            self.initPlotPeaks_EMG(axes)
        elif self.analysisType =="temp":
            self.initPlotPeaks_Temp(axes)
        elif self.analysisType =="general_lf":
            self.initPlotPeaks_General_LF(axes)
        elif self.analysisType =="general_hf":
            self.initPlotPeaks_General_HF(axes)
        elif self.analysisType =="bvp":
            self.initPlotPeaks_General_LF(axes)
        elif self.analysisType =="acc":
            self.initPlotPeaks_General_HF(axes)
        else:
            assert False, f"No Plotting Availible for {self.analysisType}"
            
        # Tighten figure's white space (must be at the end)
        # self.plottingClass.fig.tight_layout(pad=2.0);
            
    # ---------------------------------------------------------------------- #
    # ----------------------- Specific Plotting Lines ---------------------- #
    
    def addRawPlots(self, axes, yLimits, yLabel, color):
        self.bioelectricDataPlots = []; self.bioelectricPlotAxes = []
        for channelIndex in range(self.numChannels):
            # Create Plots
            if self.numChannels == 1:
                self.bioelectricPlotAxes.append(axes[0])
            else:
                self.bioelectricPlotAxes.append(axes[channelIndex, 0])
            
            # Generate Plot
            self.bioelectricDataPlots.append(self.bioelectricPlotAxes[channelIndex].plot([], [], '-', c=color, linewidth=1, alpha = 0.65)[0])
            
            # Set Figure Limits
            self.bioelectricPlotAxes[channelIndex].set_ylim(yLimits[0], yLimits[1])
            # Label Axis + Add Title
            self.bioelectricPlotAxes[channelIndex].set_ylabel(yLabel, fontsize=13, labelpad = 10)
            
    def addFilteredPlots(self, axes, yLimits, color):
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
            self.filteredBioelectricDataPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c=color, linewidth=1, alpha = 0.65)[0])

            # Set Figure Limits
            self.filteredBioelectricPlotAxes[channelIndex].set_ylim(yLimits[0], yLimits[1])
            
    def addFeaturePlots(self, axes, yLimits, color):
        # Create the feature plots
        self.featureDataPlots = []
        self.featureDataPlotAxes = [] 
        for channelIndex in range(self.numChannels):
            # Create Plot Axes
            if self.numChannels == 1:
                self.featureDataPlotAxes.append(axes[2])
            else:
                self.featureDataPlotAxes.append(axes[channelIndex, 2])
            
            # Plot Flitered Peaks
            self.featureDataPlots.append(self.featureDataPlotAxes[channelIndex].plot([], [], '-', c=color, linewidth=1, alpha = 0.65)[0])

            # Set Figure Limits
            self.featureDataPlotAxes[channelIndex].set_ylim(yLimits[0], yLimits[1])
            self.featureDataPlotAxes[channelIndex].set_xlim(0, 900)
            
    # ---------------------------------------------------------------------- #
    # ---------------------------- Data Plotting --------------------------- #
            
    def initPlotPeaks_EOG(self, axes):         
        color = "tab:blue"
        
        # Plot the Raw Data
        yLimLow = 0; yLimHigh = 3.5; 
        self.addRawPlots(axes, (yLimLow, yLimHigh), yLabel = "EOG (Volts)", color = color)
        # Add the filtered data plots
        self.addFilteredPlots(axes, (yLimLow, yLimHigh), color = color)
        # Add the feature plots
        self.addFeaturePlots(axes, (0.14, 0.28), color = color)
            
        # Create other plots.
        self.eyeBlinkLocPlots = []
        self.eyeBlinkCulledLocPlots = []
        self.trailingAveragePlots = []
        for channelIndex in range(self.numChannels):
            # Plot filtered peaks and data
            self.trailingAveragePlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="black", linewidth=1, alpha = 0.65)[0])
            self.eyeBlinkLocPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], 'o', c="black", markersize=7, alpha = 0.65)[0])
            self.eyeBlinkCulledLocPlots.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], 'o', c="tab:red", markersize=7, alpha = 0.65)[0])
        
    def initPlotPeaks_EEG(self, axes): 
        color = "tab:green"
        
        # Plot the Raw Data
        yLimLow = 0; yLimHigh = 3.3; 
        self.addRawPlots(axes, (yLimLow, yLimHigh), yLabel = "EEG (Volts)", color = color)
        # Add the filtered data plots
        yLimLow = -1.65; yLimHigh = 1.65; 
        self.addFilteredPlots(axes, (yLimLow, yLimHigh), color = color)
        # Add the feature plots
        self.addFeaturePlots(axes, (0.001, 0.03), color = color)

    def initPlotPeaks_EDA(self, axes): 
        color = "purple"
        
        # Add the raw data plots
        yLimLow = 1E-6; yLimHigh = 1E-5; 
        self.addRawPlots(axes, (yLimLow, yLimHigh), yLabel = "EDA (Siemens)", color = color)
        # Add the filtered data plots
        self.addFilteredPlots(axes, (yLimLow, yLimHigh), color = color)
        # Add the feature plots
        self.addFeaturePlots(axes, (0, 5e-17), color = color)
        
    def initPlotPeaks_EMG(self, axes): 
        # Establish pointers to the figure
        self.fig = self.plottingClass.fig
        axes = self.plottingClass.axes[self.analysisType][0]
        
        color = "tab:orange"
        # Add the raw data plots
        yLimLow = 0; yLimHigh = 5; 
        self.addRawPlots(axes, (yLimLow, yLimHigh), yLabel = "EMG (Volts)", color = color)
        # Add the filtered data plots
        yLimLow = 0; yLimHigh = 0.3; 
        self.addFilteredPlots(axes, (yLimLow, yLimHigh), color = color)
        # Add the feature plots
        self.addFeaturePlots(axes, (0, 0.3), color = color)

        # Create other plots.
        self.timeDelayPlotsRaw = []; self.timeDelayPlotsRMS = []
        for channelIndex in range(self.numChannels):
            # Time delay plots.
            self.timeDelayPlotsRaw.append(self.bioelectricPlotAxes[channelIndex].plot([], [], '-', c="blue", linewidth=2, alpha = 0.65)[0])
            self.timeDelayPlotsRMS.append(self.filteredBioelectricPlotAxes[channelIndex].plot([], [], '-', c="blue", linewidth=2, alpha = 0.65)[0])
        self.filteredBioelectricPeakPlots = [[] for _ in range(self.numChannels)]
        
    def initPlotPeaks_Temp(self, axes): 
        # Establish pointers to the figure
        self.fig = self.plottingClass.fig
        axes = self.plottingClass.axes[self.analysisType][0]
        
        color = "tab:red"
        # Add the raw data plots
        yLimLow = 20; yLimHigh = 45; 
        self.addRawPlots(axes, (yLimLow, yLimHigh), yLabel = "Temperature (\u00B0C)", color = color)
        # Add the filtered data plots
        self.addFilteredPlots(axes, (yLimLow, yLimHigh), color = color)
        # Add the feature plots
        self.addFeaturePlots(axes, (-0.003, 0.003), color = color)
        
    def initPlotPeaks_General_LF(self, axes): 
        # Establish pointers to the figure
        self.fig = self.plottingClass.fig
        axes = self.plottingClass.axes[self.analysisType][0]
        
        color = "tab:brown"
        # Add the raw data plots
        yLimLow = 0; yLimHigh = 5; 
        self.addRawPlots(axes, (yLimLow, yLimHigh), yLabel = "Signal (AU)", color = color)
        # Add the filtered data plots
        self.addFilteredPlots(axes, (yLimLow, yLimHigh), color = color)
        # Add the feature plots
        self.addFeaturePlots(axes, (-1, 1), color = color)
        
    def initPlotPeaks_General_HF(self, axes): 
        # Establish pointers to the figure
        self.fig = self.plottingClass.fig
        axes = self.plottingClass.axes[self.analysisType][0]
        
        color = "tab:brown"
        # Add the raw data plots
        yLimLow = 0; yLimHigh = 5; 
        self.addRawPlots(axes, (yLimLow, yLimHigh), yLabel = "Signal (AU)", color = color)
        # Add the filtered data plots
        self.addFilteredPlots(axes, (yLimLow, yLimHigh), color = color)
        # Add the feature plots
        self.addFeaturePlots(axes, (-1, 1), color = color)
        
# -------------------------------------------------------------------------- #
