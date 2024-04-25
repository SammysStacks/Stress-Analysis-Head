
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic modules
import re
import scipy
import numpy as np

# -------------------------------------------------------------------------- #
# ------------------------- Universal Methods Class ------------------------ #

class universalMethods:
    
    # ---------------------------------------------------------------------- #
    # --------------------- Feature Extraction Methods --------------------- #
    
    def hjorthParameters(self, timePoints, data, firstDeriv = None, secondDeriv = None):
        # Assert the correct data format.
        data = np.asarray(data)
        
        # If no derivatives given, calculate the derivatives
        if firstDeriv == None:
            if np.min(np.diff(timePoints)) < 1e-10: print("ALERT: Values too small for gradient: {np.min(np.diff(timePoints))}")
            firstDeriv = np.gradient(data, timePoints)
        if secondDeriv == None:
            secondDeriv = np.gradient(firstDeriv, timePoints)
        
        # Calculat the hjorthActivity
        activity = np.var(data)
        
        # If activity is zero, complexity and mobility are zero.
        if activity == 0 and len(data.shape) == 1:
            return activity, 0, 0, 0, 0
        
        # Calculate the hjorthMobility
        firstDerivVariance = np.var(firstDeriv, ddof = 1)
        mobility = np.sqrt(firstDerivVariance / activity)
        
        # Calculate the hjorthComplexity
        secondDerivVariance = np.var(secondDeriv, ddof = 1)
        complexity = np.sqrt(secondDerivVariance / firstDerivVariance) / mobility

        return activity, mobility, complexity, firstDerivVariance, secondDerivVariance
        
    def bandPower(self, powerSpectrumDensity, powerSpectrumDensityFreqs, bands):    
        bandPowers = []
        for freqBand in bands:
            # Find the indices corresponding to the band of interest
            idx_band = np.logical_and(powerSpectrumDensityFreqs >= freqBand[0], powerSpectrumDensityFreqs <= freqBand[1])
            assert idx_band.sum() != 0, f"You do not have enough sampling frequency to view this band: {freqBand}"
            
            # Calculate the power in the band of interest
            power = scipy.integrate.simps(y = powerSpectrumDensity[idx_band], x = powerSpectrumDensityFreqs[idx_band])
            
            bandPowers.append(power)
        
        return bandPowers
    
    # ---------------------------------------------------------------------- #
    # ------------------------- Formatting Arrays  ------------------------- #
    
    def getEvenlySampledArray(self, sampling_frequency, num_points, start_time = 0):
        end_time = (num_points - 1) / sampling_frequency
        time_array = np.arange(start_time, end_time + 1/sampling_frequency, 1/sampling_frequency)
        return time_array


    # ---------------------------------------------------------------------- #
    # ------------------------ Numeric Information  ------------------------ #
    
    def isNumber(self, string):
        return bool(re.match(r'^[-+]?\d*\.?\d+$', string))
    
    def convertToOddInt_Positive(self, x):
        return self.convertToOddInt(x, minInt = 1)
    
    def convertToOddInt(self, x, minInt = None, maxInt = None):
        # Round to nearest odd integer.
        oddInt = 2*round((x+1)/2) - 1
        
        # Bound the odd integer.
        if minInt != None: 
            assert minInt % 2 == 1
            oddInt = max(oddInt, minInt)
        if minInt != None:
            assert maxInt % 2 == 1
            oddInt = min(oddInt, maxInt)
        
        return oddInt
    
    # ---------------------------------------------------------------------- #
    # -------------------------- Formatting Lines -------------------------- #
    
    def findPointCrossing(self, array, threshold = 0):
        # Shift the threshold to zero
        array = np.array(array) - threshold
        
        # Return the value right BEFORE the zero crossing.
        return np.where(np.diff(np.sign(array) >= 0))[0]
    
    def findLineIntersectionPoint(self, leftLineParams, rightLineParams):
        """
        Parameters
        ----------
        leftLineParams: A list of length 2 representing the [slope, y-intercept] ([m, b] in y = mx + b)
        rightLineParams: A list of length 2 representing the [slope, y-intercept] ([m, b] in y = mx + b)
        """
        xPoint = (rightLineParams[1] - leftLineParams[1])/(leftLineParams[0] - rightLineParams[0])
        yPoint = leftLineParams[0]*xPoint + leftLineParams[1]
        return xPoint, yPoint
    
    # ---------------------------------------------------------------------- #
    # ------------------------ Find Peak Information ----------------------- #
    
    def findNearbyMinimum(self, data, xPointer, binarySearchWindow = 5, maxPointsSearch = 10000):
        """
        Search Right: binarySearchWindow > 0
        Search Left: binarySearchWindow < 0
        """
        # Base Case
        if abs(binarySearchWindow) < 1 or maxPointsSearch == 0:
            searchSegment = data[max(0,xPointer-1):min(xPointer+2, len(data))]
            xPointer -= np.where(searchSegment==data[xPointer])[0][0]
            return xPointer + np.argmin(searchSegment) 
        
        maxHeightPointer = xPointer
        maxHeight = data[xPointer]; searchDirection = binarySearchWindow//abs(binarySearchWindow)
        # Binary Search Data to Find the Minimum (Skip Over Minor Fluctuations)
        for dataPointer in range(max(xPointer, 0), max(0, min(xPointer + searchDirection*maxPointsSearch, len(data))), binarySearchWindow):
            # If the Next Point is Greater Than the Previous, Take a Step Back
            if data[dataPointer] >= maxHeight and xPointer != dataPointer:
                return self.findNearbyMinimum(data, dataPointer - binarySearchWindow, round(binarySearchWindow/4), maxPointsSearch - searchDirection*(abs(dataPointer - binarySearchWindow)) - xPointer)
            # Else, Continue Searching
            else:
                maxHeightPointer = dataPointer
                maxHeight = data[dataPointer]

        # If Your Binary Search is Too Large, Reduce it
        return self.findNearbyMinimum(data, maxHeightPointer, round(binarySearchWindow/2), maxPointsSearch-1)
    
    def findNearbyMaximum(self, data, xPointer, binarySearchWindow = 5, maxPointsSearch = 10000):
        """
        Search Right: binarySearchWindow > 0
        Search Left: binarySearchWindow < 0
        """
        # Base Case
        xPointer = min(max(xPointer, 0), len(data)-1)
        if abs(binarySearchWindow) < 1 or maxPointsSearch == 0:
            searchSegment = data[max(0,xPointer-1):min(xPointer+2, len(data))]
            xPointer -= np.where(searchSegment==data[xPointer])[0][0]
            return xPointer + np.argmax(searchSegment)
        
        minHeightPointer = xPointer; minHeight = data[xPointer];
        searchDirection = binarySearchWindow//abs(binarySearchWindow)
        # Binary Search Data to Find the Minimum (Skip Over Minor Fluctuations)
        for dataPointer in range(xPointer, max(0, min(xPointer + searchDirection*maxPointsSearch, len(data))), binarySearchWindow):
            # If the Next Point is Greater Than the Previous, Take a Step Back
            if data[dataPointer] < minHeight and xPointer != dataPointer:
                return self.findNearbyMaximum(data, dataPointer - binarySearchWindow, round(binarySearchWindow/2), maxPointsSearch - searchDirection*(abs(dataPointer - binarySearchWindow)) - xPointer)
            # Else, Continue Searching
            else:
                minHeightPointer = dataPointer
                minHeight = data[dataPointer]

        # If Your Binary Search is Too Large, Reduce it
        return self.findNearbyMaximum(data, minHeightPointer, round(binarySearchWindow/2), maxPointsSearch-1)
    
    def findRightMinMax(self, data, xPointer, binarySearchWindow = 5, maxPointsSearch = 10000):
        rightMinumum = self.universalMethods.findNearbyMinimum(data, xPointer, binarySearchWindow, maxPointsSearch)
        rightMaximum = self.universalMethods.findNearbyMaximum(data, xPointer, binarySearchWindow, maxPointsSearch)

        if rightMinumum < rightMaximum: 
            return rightMaximum, 1
        else:
            return rightMinumum, -1
    
    def localOptimization(self, data, points, localType, binarySearchWindow, maxPointsSearch):
        optimizedPoints = []
        if localType == 'min':
            for point in points:
                left = self.universalMethods.findNearbyMinimum(data, point, binarySearchWindow * -1, maxPointsSearch)
                right = self.universalMethods.findNearbyMinimum(data, point, binarySearchWindow, maxPointsSearch)
                if data[left] < data[right]:
                    optimizedPoints.append(left)
                else:
                    optimizedPoints.append(right)
        elif localType == 'max':
            for point in points:
                left = self.universalMethods.findNearbyMaximum(data, point, binarySearchWindow * -1, maxPointsSearch)
                right = self.universalMethods.findNearbyMaximum(data, point, binarySearchWindow, maxPointsSearch)
                if data[left] > data[right]:
                    optimizedPoints.append(left)
                else:
                    optimizedPoints.append(right)
        else:
            return points
        
        return optimizedPoints
    
    def findPrevBaselinePointer(self, data, xPointer, binarySearchWindow):
        currBaseLinePointer = xPointer
        newBaseLinePointer = self.universalMethods.findNearbyMinimum(data, currBaseLinePointer, binarySearchWindow, maxPointsSearch = int(self.samplingFreq * 30))
        print(binarySearchWindow)
        # print(currBaseLinePointer, newBaseLinePointer)
        # print(data[currBaseLinePointer] - data[newBaseLinePointer])
        
        while newBaseLinePointer < currBaseLinePointer and data[currBaseLinePointer] - data[newBaseLinePointer] > self.peakHeight_Threshold and newBaseLinePointer > 0:
            currBaseLinePointer = newBaseLinePointer
            newBaseLinePointer = self.universalMethods.findNearbyMinimum(data, currBaseLinePointer, binarySearchWindow, maxPointsSearch = int(self.samplingFreq * 30))   
            # print('walking')
            # print(data[currBaseLinePointer] - data[newBaseLinePointer])
        return currBaseLinePointer
    
    def findLocalMax(self, data, xPointer, binarySearchWindow = 5, maxPointsSearch = 10000):
        # Find the local maximum to the left and the right.
        localMax = self.findNearbyMaximum(data, xPointer, binarySearchWindow, maxPointsSearch)
        localMax = self.findNearbyMaximum(data, localMax, -binarySearchWindow, maxPointsSearch)
        
        return localMax
    

    
    
    
    