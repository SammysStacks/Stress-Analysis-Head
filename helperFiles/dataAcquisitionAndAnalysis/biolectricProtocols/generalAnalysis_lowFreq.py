import numpy as np
import scipy

# Import Files
from .globalProtocol import globalProtocol


class generalProtocol_lowFreq(globalProtocol):
    
    def __init__(self, numPointsPerBatch = 2000, moveDataFinger = 200, channelIndices=(), plottingClass = None, readData = None):
        # Feature collection parameters
        self.startFeatureTimePointer = []  # The start pointer of the feature window interval.
        self.featureTimeWindow = None         # The duration of time that each feature considers
        self.minPointsPerBatch = None       # The minimum number of points in a batch to analyze.

        # High Pass Filter Parameters
        self.cutOffFreq = [None, 25]        # Optimal LPF Cutoff in Literature is 6-8 or 20 Hz (Max 35 or 50); I Found 25 Hz was the Best, but can go to 15 if noisy (small amplitude cutoff)

        # Reset analysis variables
        super().__init__("general_lf", numPointsPerBatch, moveDataFinger, channelIndices, plottingClass, readData)
        self.resetAnalysisVariables()

    def resetAnalysisVariables(self):
        # General parameters
        self.startFeatureTimePointer = [0 for _ in range(self.numChannels)]    # The start pointer of the feature window interval.
        self.featureTimeWindow = self.featureTimeWindow_lowFreq  # The duration of time that each feature considers
        self.minPointsPerBatch = None       # The minimum number of points in a batch to analyze.

    def checkParams(self):
        # Assert that the buffer is large enough for the feature window.
        assert self.featureTimeWindow < self.dataPointBuffer, "The buffer does not include enough points for the feature window"
        
    def setSamplingFrequencyParams(self):
        maxBufferSeconds = max(self.featureTimeWindow, 15)
        # Set Parameters
        self.lastAnalyzedDataInd[:] = int(self.samplingFreq*self.featureTimeWindow)
        self.minPointsPerBatch = int(self.samplingFreq*self.featureTimeWindow/2)
        self.dataPointBuffer = max(self.dataPointBuffer, int(self.samplingFreq*maxBufferSeconds))
        
    # ----------------------------------------------------------------------- #
    # ------------------------- Data Analysis Begins ------------------------ #

    def analyzeData(self, dataFinger):
                
        # Add incoming Data to Each Respective Channel's Plot
        for channelIndex in range(self.numChannels):
            
            # ---------------------- Filter the Data ----------------------- #    
            # Find the starting/ending points of the data to analyze
            startFilterPointer = max(dataFinger - self.dataPointBuffer, 0)
            dataBuffer = np.asarray(self.channelData[channelIndex][startFilterPointer:dataFinger + self.numPointsPerBatch])
            timepoints = np.asarray(self.timepoints[startFilterPointer:dataFinger + self.numPointsPerBatch])

            # Get the Sampling Frequency from the First Batch (If Not Given)
            if not self.samplingFreq:
                self.setSamplingFrequency(startFilterPointer)
                
            # Filter the data and remove bad indices
            filteredTime, filteredData, goodIndicesMask = self.filterData(timepoints, dataBuffer)
            # --------------------------------------------------------------- #
            
            # ---------------------- Feature Extraction --------------------- #
            if self.collectFeatures:
                # Initialize the new raw features and times.
                newFeatureTimes, newRawFeatures = [], []

                # Extract features across the dataset
                while self.lastAnalyzedDataInd[channelIndex] < len(self.timepoints):
                    featureTime = self.timepoints[self.lastAnalyzedDataInd[channelIndex]]
                                        
                    # Find the start window pointer
                    self.startFeatureTimePointer[channelIndex] = self.findStartFeatureWindow(self.startFeatureTimePointer[channelIndex], featureTime, self.featureTimeWindow)
                    # Compile the good data in the feature interval.
                    intervalTimes, intervalData = self.compileBatchData(filteredTime, filteredData, goodIndicesMask, startFilterPointer, self.startFeatureTimePointer[channelIndex], channelIndex)
                    
                    # Only extract features if enough information is provided.
                    if self.minPointsPerBatch < len(intervalTimes):
                        # Calculate and save the features in this window.
                        finalFeatures = self.extractFeatures(intervalTimes, intervalData)

                        # Keep track of the new features.
                        newRawFeatures.append(finalFeatures)
                        newFeatureTimes.append(featureTime)
                
                    # Keep track of which data has been analyzed 
                    self.lastAnalyzedDataInd[channelIndex] += int(self.samplingFreq*self.secondsPerFeature)

                # Compile the new raw features into a smoothened (averaged) feature.
                self.readData.compileContinuousFeatures(newFeatureTimes, newRawFeatures, self.rawFeatureTimes[channelIndex], self.rawFeatures[channelIndex], self.compiledFeatures[channelIndex], self.featureAverageWindow)

            # -------------------------------------------------------------- #  
        
            # ------------------- Plot Biolectric Signals ------------------- #
            if self.plotStreamedData:
                # Format the raw data:.
                timepoints = timepoints[dataFinger - startFilterPointer:] # Shared axis for all signals
                rawData = dataBuffer[dataFinger - startFilterPointer:]
                # Format the filtered data
                filterOffset = (goodIndicesMask[0:dataFinger - startFilterPointer]).sum(axis = 0, dtype=int)

                # Plot Raw Bioelectric Data (Slide Window as Points Stream in)
                self.plottingMethods.bioelectricDataPlots[channelIndex].set_data(timepoints, rawData)
                self.plottingMethods.bioelectricPlotAxes[channelIndex].set_xlim(timepoints[0], timepoints[-1])
                                            
                # Plot the Filtered + Digitized Data
                self.plottingMethods.filteredBioelectricDataPlots[channelIndex].set_data(filteredTime[filterOffset:], filteredData[filterOffset:])
                self.plottingMethods.filteredBioelectricPlotAxes[channelIndex].set_xlim(timepoints[0], timepoints[-1]) 
                
                # Plot a single feature.
                if len(self.compiledFeatures[channelIndex]) != 0:
                    self.plottingMethods.featureDataPlots[channelIndex].set_data(self.rawFeatureTimes[channelIndex], np.asarray(self.compiledFeatures[channelIndex])[:, 9])
                    self.plottingMethods.featureDataPlotAxes[channelIndex].legend(["Signal Slope"], loc="upper left")

            # -------------------------------------------------------------- #   
            
    def filterData(self, timepoints, data):
        # Filter the Data: Low pass Filter and Savgol Filter
        filteredData = self.filteringMethods.bandPassFilter.butterFilter(data, self.cutOffFreq[1], self.samplingFreq, order = 3, filterType = 'low', fastFilt = True)
        filteredTime = timepoints.copy()
        # print(len(filteredTime), len(filteredData))
        return filteredTime, filteredData, np.ones(len(filteredTime))

    def findStartFeatureWindow(self, timePointer, currentTime, timeWindow):
        # Loop through until you find the first time in the window 
        while self.timepoints[timePointer] < currentTime - timeWindow:
            timePointer += 1
            
        return timePointer
    
    def compileBatchData(self, filteredTime, filteredData, goodIndicesMask, startFilterPointer, startFeatureTimePointer, channelIndex):
        assert len(goodIndicesMask) >= len(filteredData) == len(filteredTime), print(len(goodIndicesMask), len(filteredData), len(filteredTime))
        
        # Accounts for the missing points (count the number of viable points within each pointer).
        startReferenceFinger = (goodIndicesMask[0:startFeatureTimePointer - startFilterPointer]).sum(axis = 0, dtype=int)
        endReferenceFinger = startReferenceFinger + (goodIndicesMask[startFeatureTimePointer - startFilterPointer:self.lastAnalyzedDataInd[channelIndex]+1 - startFilterPointer]).sum(axis = 0, dtype=int)
        # Compile the information in the interval.
        intervalTimes = filteredTime[startReferenceFinger:endReferenceFinger]
        intervalData = filteredData[startReferenceFinger:endReferenceFinger]

        return intervalTimes, intervalData
    
    # --------------------- Feature Extraction Methods --------------------- #
    
    def extractFeatures(self, timepoints, data):

        # ----------------------- Data Preprocessing ----------------------- #
        
        # Normalize the data
        standardized_data = self.universalMethods.standardizeData(data)
        if all(standardized_data == 0):
            return [0 for _ in range(8)]
        
        # Get the baseline data
        baselineX = timepoints - timepoints[0]

        # Calculate the derivatives
        firstDerivative = np.gradient(standardized_data, timepoints)

        # ------------------------------------------------------------------ #  
        # ----------------------- Features from Data ----------------------- #
        
        # General Shape Parameters
        signalPower = scipy.integrate.simpson(y=data ** 2, x=timepoints) / (baselineX[-1] - baselineX[0])
        signalArea = scipy.integrate.simpson(y=data, x=timepoints) / (baselineX[-1] - baselineX[0])
        signalRange = max(data) - min(data)
        standardDeviation = np.std(data, ddof=1)
        mean = np.mean(data)

        # -------------------- Features from Derivatives ------------------- #
        
        # First derivative features
        firstDerivativeMean = np.mean(firstDerivative)
        firstDerivativeStdDev = np.std(firstDerivative, ddof=1)
        firstDerivativePower = scipy.integrate.simpson(y=firstDerivative ** 2, x=timepoints) / (baselineX[-1] - baselineX[0])

        # ----------------------- Organize Features ------------------------ #
        
        finalFeatures = []
        # Add peak shape parameters
        finalFeatures.extend([mean, standardDeviation])
        finalFeatures.extend([signalRange, signalPower, signalArea])
        # Add derivative features
        finalFeatures.extend([firstDerivativeMean, firstDerivativeStdDev, firstDerivativePower])

        return finalFeatures
    
    