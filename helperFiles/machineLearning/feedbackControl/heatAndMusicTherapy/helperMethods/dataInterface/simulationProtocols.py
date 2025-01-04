# General
import torch
import torch.nn.functional as F

# Import helper files.
from helperFiles.machineLearning.feedbackControl.heatAndMusicTherapy.helperMethods.therapyProtcols.helperTherapyMethods.generalMethods import generalMethods
from helperFiles.machineLearning.feedbackControl.heatAndMusicTherapy.helperMethods.dataInterface.dataInterface import dataInterface

class simulationProtocols:
    def __init__(self, allParameterBins, allPredictionBins, predictionBinWidths, modelParameterBounds, numPredictions, numParameters, predictionWeights, optimalNormalizedState, initialParameterBounds, unNormalizedParameterBins, simulationParameters,
                 therapySelection):
        # ================================================General Parameters================================================
        self.unNormalizedParamBins = unNormalizedParameterBins
        self.optimalNormalizedState = optimalNormalizedState
        self.initialParameterBounds = initialParameterBounds
        self.modelParameterBounds = modelParameterBounds
        self.simulationParameters = simulationParameters
        self.predictionBinWidths = predictionBinWidths
        self.allPredictionBins = allPredictionBins
        self.predictionWeights = predictionWeights
        self.allParameterBins = allParameterBins
        self.numPredictions = numPredictions
        self.numParameters = numParameters

        # ===============================================Hardcoded Parameters===============================================
        self.therapySelection = therapySelection
        if self.therapySelection == 'Heat':
            self.startingPoints = [47, 0.5966159, 0.69935307, 0.91997683]
        elif self.therapySelection == 'BinauralBeats':
            self.startingPoints = [[426, 430], [0.5966159, 0.69935307, 0.91997683]]
        self.initialTimePoint = 0
        self.initialPoints = 1
        self.timeDelay = 10

        # ======================================Simulation Parameters Initializations=======================================
        self.UnNormalizedUniformParamSampler = torch.distributions.uniform.Uniform(torch.FloatTensor([initialParameterBounds.squeeze()[0]]), torch.FloatTensor([initialParameterBounds.squeeze()[1]]))
        self.uniformPredictionSampler = torch.distributions.uniform.Uniform(torch.FloatTensor([modelParameterBounds[0]]), torch.FloatTensor([modelParameterBounds[1]]))
        self.uniformParamSampler = torch.distributions.uniform.Uniform(torch.FloatTensor([modelParameterBounds[0]]), torch.FloatTensor([modelParameterBounds[1]]))
        self.numSimulationHeuristicSamples = simulationParameters['numSimulationHeuristicSamples']
        self.numSimulationTrueSamples = simulationParameters['numSimulationTrueSamples']
        self.heuristicMapType = simulationParameters['heuristicMapType']
        self.simulatedMapType = simulationParameters['simulatedMapType']

        # ===============================================Simulated Parameters===============================================
        self.startingTimes, self.startingParams, self.startingPredictions = self.randomlySamplePoints(numPoints=self.initialPoints, lastTimePoint=self.initialTimePoint - self.timeDelay)
        # print(f'startingTimes, straringParams, startingPredictions: {self.startingTimes, self.startingParams, self.startingPredictions}')

        # ==============================================Uninitialize Parameters=============================================
        self.simulatedMapPA = None
        self.simulatedMapNA = None
        self.simulatedMapSA = None
        self.simulatedMapCompiledLoss = None
        self.simulatedMapCompiledLossUnNormalized = None

        # ============================================Real Data for simulations=============================================
        self.realSimMapPA = None
        self.realSimMapNA = None
        self.realSimMapSA = None
        self.realSimMapCompiledLoss = None
        self.realSimMapCompiledLossUnNormalized = None

        # ===================================================Helper Class===================================================
        self.dataInterface = dataInterface(predictionWeights, optimalNormalizedState)
        self.generalMethods = generalMethods()

    """Simulate individual times"""
    def getSimulatedTimes(self, numPoints, lastTimePoint=None):
        # If lastTimePoint is not provided, start from 0
        startTime = lastTimePoint if lastTimePoint is not None else 0

        # Generate the tensor with the specified number of points, starting from startTime, incremented by timeDelay
        simulatedTimes = torch.arange(start=startTime + self.timeDelay, end=startTime + numPoints * self.timeDelay + self.timeDelay, step=self.timeDelay)
        return simulatedTimes
        # potential alternative:
        # # If no time is given, start over.
        # lastTimePoint = lastTimePoint or -self.timeDelay
        # # Simulate the time points.
        # currentTimePoint = lastTimePoint + self.timeDelay
        # start = currentTimePoint + self.timeDelay
        # end = currentTimePoint + numPoints * self.timeDelay
        # if start >= end:
        #     print("Start value is greater than or equal to end value, returning empty tensor.")
        #     return torch.tensor([], dtype=torch.int64)
        # simulatedTimes = torch.arange(start, end, self.timeDelay)
        # # simulatedTimes dimension: numPoints
        # print('simulatedTimes:', simulatedTimes)
        # return simulatedTimes

    """Define the initialStates"""
    def getInitialState(self):
        # TODO: prediction is for loss, parameter is the whatever we can control/change
        return self.startingTimes, self.startingParams, self.startingPredictions  # (initialPoints), (initialPoints, numParams), (initialPoints, predictions = emotion states)

    """Randomly sample states"""
    def randomlySamplePoints(self, numPoints=1, lastTimePoint=None):
        # --------------------------------------Dimensions--------------------------------------
        # sampledPredictions dimension: numPoints, numPredictions
        # sampledParameters dimension: numPoints, numParameters
        # simulatedTimes dimension: numPoints
        # --------------------------------------------------------------------------------------

        # Generate a random temperature within the bounds.
        sampledPredictions = self.uniformPredictionSampler.sample(torch.Size([numPoints, self.numPredictions])).unsqueeze(-1)  # torch.size ([numPoints, numPredictions, 1, 1]) ([1,3,1,1])
        sampledParameters = self.uniformParamSampler.sample(torch.Size([numPoints, self.numParameters])).unsqueeze(-1)  # torch.size ([numPoints, numParameters, 1, 1]) ([1,1,1,1]) for heat therapy; ([1, 2, 1, 1]) for binaural beats
        simulatedTimes = self.getSimulatedTimes(numPoints, lastTimePoint)  # torch.size([0]) at the beginning. tensor([], dtype=torch.int64)
        return simulatedTimes, sampledParameters, sampledPredictions


    """Simulation interface"""
    def initializeSimulatedMaps(self, lossWeights, gausParamSTDs, gausLossSTDs, applyGaussianFilter):
        # Convert gausSTD to a 2D tensor by combining paramSTDs and lossSTDs
        # Get the simulated data points.
        # --------------------------------------Dimensions--------------------------------------
        # sampledPredictions dimension: numSimulationTrueSamples, numPoints, numPredictions, 1, 1; torch.Size([30, 3, 1, 1])
        # sampledParameters dimension: numSimulationTrueSamples, numPoints, numParameters, 1, 1; torch.Size([30, 1, 1, 1]) or torch.Size([30, 2, 1, 1])
        # simulatedTimes dimension: numSimulationTrueSamples with concatination of 0 at the beginning torch([30]) otherwise numSimulationTrueSamples - 1
        # Get the simulated matrix from the simulated points.
        # --------------------------------------------------------------------------------------
        simulatedTimes, sampledParameters, sampledPredictions = self.generateSimulatedMap()

        simulatedCompiledLoss = self.dataInterface.calculateCompiledLoss(sampledPredictions)  # SampledPredictions: numPoints, (PA, NA, SA); 2D array

        # Compiling the simulated Data for generating probability matrix
        initialSimulatedData = torch.cat((sampledParameters, sampledPredictions, simulatedCompiledLoss), dim=1) # numPoints, (Parameter, emotionPrediction, compiledLoss), 1, 1;
        if self.therapySelection == 'Heat':
            initialSimulatedData_PA = initialSimulatedData[:, [0, 1], :, :]  # Shape: [30, 2, 1, 1]

            # Extract for NA: Parameter and second emotion prediction
            initialSimulatedData_NA = initialSimulatedData[:, [0, 2], :, :]  # Shape: [30, 2, 1, 1]

            # Extract for SA: Parameter and third emotion prediction
            initialSimulatedData_SA = initialSimulatedData[:, [0, 3], :, :]  # Shape: [30, 2, 1, 1]
            resampledParameterBins, resampledPredictionBins = self.generalMethods.resampleBins(self.allParameterBins, self.allPredictionBins, eventlySpacedBins=False)

            """VERY IMPORTANT: simulated maps are all probability matrices, meaning the probability of possessing certain parameter prediction pairs on the map"""
            self.simulatedMapPA = self.generalMethods.getProbabilityMatrix(initialSimulatedData_PA, resampledParameterBins, resampledPredictionBins[0], gausParamSTDs, gausLossSTDs[0], noise=0.1, applyGaussianFilter=applyGaussianFilter)
            self.simulatedMapNA = self.generalMethods.getProbabilityMatrix(initialSimulatedData_NA, resampledParameterBins, resampledPredictionBins[1], gausParamSTDs, gausLossSTDs[1], noise=0.1, applyGaussianFilter=applyGaussianFilter)
            self.simulatedMapSA = self.generalMethods.getProbabilityMatrix(initialSimulatedData_SA, resampledParameterBins, resampledPredictionBins[2], gausParamSTDs, gausLossSTDs[2], noise=0.1, applyGaussianFilter=applyGaussianFilter)

        elif self.therapySelection == 'BinauralBeats':
            initialSimulatedData_PA = initialSimulatedData[:, [0, 1, 2], :, :]  # Shape: [30, 3, 1, 1] 3 for 2 parameters and 1 PA value

            # Extract for NA: Parameter and second emotion prediction
            initialSimulatedData_NA = initialSimulatedData[:, [0, 1, 3], :, :]  # Shape: [30, 3, 1, 1]

            # Extract for SA: Parameter and third emotion prediction
            initialSimulatedData_SA = initialSimulatedData[:, [0, 1, 4], :, :]  # Shape: [30, 3, 1, 1]
            # resample the data bins

            resampledParameterBins_single, resampledPredictionBins = self.generalMethods.resampleBins(self.allParameterBins[0], self.allPredictionBins, eventlySpacedBins=False)
            resampledParameterBins = [resampledParameterBins_single, resampledParameterBins_single]

            """VERY IMPORTANT: simulated maps are all probability matrices, meaning the probability of possessing certain parameter prediction pairs on the map"""
            self.simulatedMapPA = self.generalMethods.get3DProbabilityMatrix(initialSimulatedData_PA, resampledParameterBins, resampledPredictionBins[0], gausParamSTDs, gausLossSTDs[0], noise=0.1, applyGaussianFilter=applyGaussianFilter)
            self.simulatedMapNA = self.generalMethods.get3DProbabilityMatrix(initialSimulatedData_NA, resampledParameterBins, resampledPredictionBins[1], gausParamSTDs, gausLossSTDs[1], noise=0.1, applyGaussianFilter=applyGaussianFilter)
            self.simulatedMapSA = self.generalMethods.get3DProbabilityMatrix(initialSimulatedData_SA, resampledParameterBins, resampledPredictionBins[2], gausParamSTDs, gausLossSTDs[2], noise=0.1, applyGaussianFilter=applyGaussianFilter)

        # Weigh the maps
        self.simulatedMapCompiledLoss = (lossWeights[0]*self.simulatedMapPA + lossWeights[1]*self.simulatedMapNA + lossWeights[2]*self.simulatedMapSA)
        self.simulatedMapCompiledLossUnNormalized = self.simulatedMapCompiledLoss
        self.simulatedMapCompiledLoss = self.simulatedMapCompiledLoss / torch.sum(self.simulatedMapCompiledLoss) # Normalization
        # self.simulatedMapCompiledLoss = self.apply_gaussian_smoothing(self.simulatedMapCompiledLoss, kernel_size=3, sigma=0.5)
        # Smoothen the map

    """Get the simulated compiled loss"""
    def getSimulatedCompiledLoss(self, currentParam, currentUserState, newParamValues=None, therapyMethod=None):
        if self.therapySelection == 'Heat':
            # Unpack the current user state.
            currentUserTemp = currentParam
            currentUserLoss = self.dataInterface.calculateCompiledLoss(currentUserState) # torch.Size([1, 1, 1, 1])
            newUserTemp = currentUserTemp if newParamValues is None else newParamValues

            # Resample for specific protocols
            if therapyMethod == 'aStarTherapyProtocol' or 'basicTherapyProtocol' or 'hmmTherapyProtocol':
                resampledParameterBins, resampledPredictionBins = self.generalMethods.resampleBins(self.allParameterBins, self.allPredictionBins, eventlySpacedBins=False)

                currentTempBinIndex = self.dataInterface.getBinIndex(resampledParameterBins[0], currentUserTemp)
                currentLossIndex = self.dataInterface.getBinIndex(resampledPredictionBins[0], currentUserLoss)
                newTempBinIndex = self.dataInterface.getBinIndex(resampledParameterBins[0], newUserTemp)
                newUserLoss, PA, NA, SA = self.sampleNewLoss(currentUserLoss, currentLossIndex, currentTempBinIndex, newTempBinIndex, currentUserState, therapyMethod, bufferZone=0.01)
                newUserLoss = torch.tensor(newUserLoss).view(1, 1, 1, 1)
                PA = torch.tensor(PA).view(1, 1, 1, 1)
                NA = torch.tensor(NA).view(1, 1, 1, 1)
                SA = torch.tensor(SA).view(1, 1, 1, 1)
                return newUserLoss, PA, NA, SA

        elif self.therapySelection == 'BinauralBeats':
            # Unpack the current user state.
            currentUserParam_1 = currentParam[:, 0]
            currentUserParam_2 = currentParam[:, 1]
            currentUserLoss = self.dataInterface.calculateCompiledLoss(currentUserState)
            newParamValues = currentParam if newParamValues is None else newParamValues

            # Resample for specific protocols
            if therapyMethod == 'aStarTherapyProtocol' or 'basicTherapyProtocol' or 'hmmTherapyProtocol':
                resampledParameterBins_single, resampledPredictionBins = self.generalMethods.resampleBins(self.allParameterBins[0], self.allPredictionBins, eventlySpacedBins=False)
                resampledParameterBins = [resampledParameterBins_single, resampledParameterBins_single]
                currentLossIndex = self.dataInterface.getBinIndex(resampledPredictionBins[0], currentUserLoss)
                currentParamBinIndex_1 = self.dataInterface.getBinIndex(resampledParameterBins[0], currentUserParam_1)
                currentParamBinIndex_2 = self.dataInterface.getBinIndex(resampledParameterBins[0], currentUserParam_2)
                newParamBinIndex_1 = self.dataInterface.getBinIndex(resampledParameterBins[0], newParamValues[:, 0])
                newParamBinIndex_2 = self.dataInterface.getBinIndex(resampledParameterBins[0], newParamValues[:, 1])

                currentParamBinIndex = [currentParamBinIndex_1, currentParamBinIndex_2]
                newParamBinIndex = [newParamBinIndex_1, newParamBinIndex_2]

                newUserLoss, PA, NA, SA = self.sampleNewLoss3D(currentUserLoss, currentLossIndex, currentParamBinIndex, newParamBinIndex, currentUserState, therapyMethod, bufferZone=0.01)
                newUserLoss = torch.tensor(newUserLoss).view(1, 1, 1, 1)
                PA = torch.tensor(PA).view(1, 1, 1, 1)
                NA = torch.tensor(NA).view(1, 1, 1, 1)
                SA = torch.tensor(SA).view(1, 1, 1, 1)
                return newUserLoss, PA, NA, SA
        else:
            # for other conditions or protocols
            # Calculate the bin indices for the current and new user states.
            currentLossIndex = self.dataInterface.getBinIndex(self.allPredictionBins, currentUserLoss)
            newTempBinIndex = self.dataInterface.getBinIndex(self.allParameterBins, newUserTemp)
            #newUserLoss, PA, NA, SA = self.sampleNewLoss(currentUserLoss, currentLossIndex, currentTempBinIndex, newTempBinIndex, currentUserState, therapyMethod, bufferZone=0.01)
        # Simulate a new user loss.
            newUserLoss = None
            PA = None
            NA = None
            SA = None
            return newUserLoss, PA, NA, SA

    """Sample new loss"""
    def sampleNewLoss(self, currentUserLoss, currentLossIndex, currentParamBinIndex, newParamIndex, currentUserState, therapyMethod=None, bufferZone=0.01, gausSTD=0.05):
        simulatedMapPA = torch.tensor(self.simulatedMapPA, dtype=torch.float32)
        simulatedMapNA = torch.tensor(self.simulatedMapNA, dtype=torch.float32)
        simulatedMapSA = torch.tensor(self.simulatedMapSA, dtype=torch.float32)
        simulatedMapCompiledLoss = torch.tensor(self.simulatedMapCompiledLoss, dtype=torch.float32)

        # Resample for specific protocols
        if therapyMethod == 'aStarTherapyProtocol' or 'basicTherapyProtocol' or 'hmmTherapyProtocol':
            resampledParameterBins, resampledPredictionBins = self.generalMethods.resampleBins(self.allParameterBins, self.allPredictionBins, eventlySpacedBins=False)
            if newParamIndex != currentParamBinIndex or torch.rand(1).item() < 0.1:

                # Calculate new loss probabilities and Gaussian boost
                newLossProbabilities = simulatedMapCompiledLoss[newParamIndex] / torch.sum(simulatedMapCompiledLoss[newParamIndex])
                gaussian_boost = self.generalMethods.createGaussianArray(inputData=newLossProbabilities.numpy(), gausMean=currentLossIndex, gausSTD=gausSTD, torchFlag=True)

                # Combine the two distributions and normalize
                newLossProbabilities = newLossProbabilities + gaussian_boost
                newLossProbabilities = newLossProbabilities / torch.sum(newLossProbabilities)

                # Sample a new loss from the distribution
                newLossBinIndex = torch.multinomial(newLossProbabilities, 1).item()
                newUserLoss = resampledPredictionBins[0][newLossBinIndex]

                # Sample distribution of loss at a certain temperature for PA, NA, SA
                simulatedSpecificMaps = {'PA': simulatedMapPA, 'NA': simulatedMapNA, 'SA': simulatedMapSA}
                specificLossProbabilities = {}
                specificUserLosses = {}

                for key in simulatedSpecificMaps.keys():
                    specificLossProbabilities[key] = simulatedSpecificMaps[key][newParamIndex] / torch.sum(simulatedSpecificMaps[key][newParamIndex])
                    gaussian_boost = self.generalMethods.createGaussianArray(inputData=specificLossProbabilities[key].numpy(), gausMean=currentLossIndex, gausSTD=gausSTD, torchFlag=True)
                    specificLossProbabilities[key] = specificLossProbabilities[key] + gaussian_boost
                    specificLossProbabilities[key] = specificLossProbabilities[key] / torch.sum(specificLossProbabilities[key])

                    newSpecificLossBinIndex = torch.multinomial(specificLossProbabilities[key], 1).item()
                    specificUserLosses[key] = resampledPredictionBins[0][newSpecificLossBinIndex]
                    print(f"{key} specific user loss: {specificUserLosses[key]}")

                # SpecificLossProbabilities contains the normalized loss probabilities for PA, NA, and SA
                return newUserLoss, specificUserLosses['PA'], specificUserLosses['NA'], specificUserLosses['SA']
            else:
                newUserLoss = currentUserLoss + torch.normal(mean=0.0, std=0.01, size=currentUserLoss.size())
                newUserLossPA = currentUserState[0][0][0][0] + torch.normal(mean=0.0, std=0.01, size=currentUserState[0][0][0][0].size())
                newUserLossNA = currentUserState[0][1][0][0] + torch.normal(mean=0.0, std=0.01, size=currentUserState[0][1][0][0].size())
                newUserLossSA = currentUserState[0][2][0][0] + torch.normal(mean=0.0, std=0.01, size=currentUserState[0][2][0][0].size())
                return newUserLoss, newUserLossPA, newUserLossNA, newUserLossSA

    """For BinauralBeats therapy specifically"""
    def sampleNewLoss3D(self, currentUserLoss, currentLossIndex, currentParamBinIndex, newParamIndex, currentUserState, therapyMethod=None, bufferZone=0.01, gausSTD=0.05):

        simulatedMapPA = torch.tensor(self.simulatedMapPA, dtype=torch.float32)
        simulatedMapNA = torch.tensor(self.simulatedMapNA, dtype=torch.float32)
        simulatedMapSA = torch.tensor(self.simulatedMapSA, dtype=torch.float32)
        simulatedMapCompiledLoss = torch.tensor(self.simulatedMapCompiledLoss, dtype=torch.float32)

        if therapyMethod in ['aStarTherapyProtocol', 'basicTherapyProtocol', 'hmmTherapyProtocol']:
            resampledParameterBins_single, resampledPredictionBins = self.generalMethods.resampleBins(self.allParameterBins[0], self.allPredictionBins, eventlySpacedBins=False)
            resampledParameterBins = [resampledParameterBins_single, resampledParameterBins_single]
            if newParamIndex != currentParamBinIndex or torch.rand(1).item() < 0.1:
                # Extract the slice of the compiled loss map for the new parameter index
                newLossProbabilities = simulatedMapCompiledLoss[newParamIndex[0], newParamIndex[1]]
                newLossProbabilities = newLossProbabilities / torch.sum(newLossProbabilities)

                # Apply Gaussian boost
                gaussian_boost = self.generalMethods.createGaussianArray(
                    inputData=newLossProbabilities.numpy(),
                    gausMean=currentLossIndex,
                    gausSTD=gausSTD,
                    torchFlag=True
                )
                newLossProbabilities = newLossProbabilities + gaussian_boost
                newLossProbabilities = newLossProbabilities / torch.sum(newLossProbabilities)

                # Sample a new loss
                newLossBinIndex = torch.multinomial(newLossProbabilities, 1).item()
                newUserLoss = resampledPredictionBins[0][newLossBinIndex]

                # Repeat the sampling process for PA, NA, SA
                specificUserLosses = {}
                for key, simMap in zip(['PA', 'NA', 'SA'], [simulatedMapPA, simulatedMapNA, simulatedMapSA]):
                    specificLossProbabilities = simMap[newParamIndex[0], newParamIndex[1]]
                    specificLossProbabilities = specificLossProbabilities / torch.sum(specificLossProbabilities)

                    # Apply Gaussian boost
                    gaussian_boost = self.generalMethods.createGaussianArray(
                        inputData=specificLossProbabilities.numpy(),
                        gausMean=currentLossIndex,
                        gausSTD=gausSTD,
                        torchFlag=True
                    )
                    specificLossProbabilities = specificLossProbabilities + gaussian_boost
                    specificLossProbabilities = specificLossProbabilities / torch.sum(specificLossProbabilities)

                    specificBinIndex = torch.multinomial(specificLossProbabilities, 1).item()
                    specificUserLosses[key] = resampledPredictionBins[0][specificBinIndex]

                return newUserLoss, specificUserLosses['PA'], specificUserLosses['NA'], specificUserLosses['SA']
            else:
                # Add small noise if no resampling
                return (
                    currentUserLoss + torch.normal(mean=0.0, std=0.01, size=currentUserLoss.size()),
                    currentUserState[0][0][0][0] + torch.normal(mean=0.0, std=0.01, size=currentUserState[0][0][0][0].size()),
                    currentUserState[0][1][0][0] + torch.normal(mean=0.0, std=0.01, size=currentUserState[0][1][0][0].size()),
                    currentUserState[0][2][0][0] + torch.normal(mean=0.0, std=0.01, size=currentUserState[0][2][0][0].size())
                )

    # =================================================Sampling Methods=================================================
    def generateSimulatedMap(self):
        """ Final dimension: numSimulationSamples, (T, PA, NA, SA); 2D array """
        if self.simulatedMapType == "uniformSampling":
            return self.uniformSampling(self.numSimulationTrueSamples)
        elif self.simulatedMapType == "linearSampling":
            return self.linearSampling(self.numSimulationTrueSamples)
        elif self.simulatedMapType == "parabolicSampling":
            return self.parabolicSampling(self.numSimulationTrueSamples)
        else:
            raise Exception()

    def uniformSampling(self, numSimulationSamples, lastTimePoint=None):
        simulatedTimes, sampledParameters, sampledPredictions = self.randomlySamplePoints(numPoints=numSimulationSamples, lastTimePoint=lastTimePoint)
        # sampledPredictions dimension: numSimulationSamples, numPredictions
        # sampledParameters dimension: numSimulationSamples, numParameters
        # simulatedTimes dimension: numSimulationSamples

        return simulatedTimes, sampledParameters, sampledPredictions

    def linearSampling(self, numSimulationSamples, lastTimePoint=None):
        # Randomly generate (uniform sampling) the times, temperature, PA, NA, SA for each data point.
        simulatedTimes, sampledParameters, sampledPredictions = self.uniformSampling(numSimulationSamples=numSimulationSamples, lastTimePoint=lastTimePoint)
        # sampledPredictions dimension: numSimulationSamples, numPredictions
        # sampledParameters dimension: numSimulationSamples, numParameters
        # simulatedTimes dimension: numSimulationSamples

        # Add a bias towards higher values.
        sampledPredictions = sampledPredictions.pow(2)
        sampledParameters = sampledParameters.pow(2)
        simulatedTimes = simulatedTimes.pow(2)

        return simulatedTimes, sampledParameters, sampledPredictions

    def parabolicSampling(self, numSimulationSamples, lastTimePoint=None):
        # Randomly generate (uniform sampling) the times, temperature, PA, NA, SA for each data point.
        simulatedTimes, sampledParameters, sampledPredictions = self.uniformSampling(numSimulationSamples=numSimulationSamples, lastTimePoint=lastTimePoint)
        # sampledPredictions dimension: numSimulationSamples, numPredictions
        # sampledParameters dimension: numSimulationSamples, numParameters
        # simulatedTimes dimension: numSimulationSamples

        # Add a bias towards higher values.
        sampledPredictions = sampledPredictions.pow(2)
        sampledParameters = sampledParameters.pow(2)
        simulatedTimes = simulatedTimes.pow(2)

        return simulatedTimes, sampledParameters, sampledPredictions

    def apply_gaussian_smoothing(self, input_map, kernel_size=5, sigma=0.1):
        """
        Apply Gaussian smoothing to the input map.

        Parameters:
        input_map (torch.Tensor): The input probability map.
        kernel_size (int): The size of the Gaussian kernel. Default is 5.
        sigma (float): The standard deviation of the Gaussian kernel. Default is 1.0.

        Returns:
        torch.Tensor: The smoothed probability map with the same size as the input.
        """

        # Create a 1D Gaussian kernel
        def gaussian_kernel_1d(kernel_size, sigma):
            half_size = (kernel_size - 1) // 2
            x = torch.arange(-half_size, half_size + 1, dtype=torch.float32)
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel = kernel / kernel.sum()
            return kernel

        # Create a 2D Gaussian kernel from the 1D kernel
        def gaussian_kernel_2d(kernel_size, sigma):
            kernel_1d = gaussian_kernel_1d(kernel_size, sigma)
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            return kernel_2d

        kernel_2d = gaussian_kernel_2d(kernel_size, sigma)
        kernel_2d = kernel_2d.expand(1, 1, -1, -1)

        # Apply the Gaussian filter
        input_map = input_map.unsqueeze(0).unsqueeze(0)
        padding = kernel_size // 2
        smoothed_map = F.conv2d(input_map, kernel_2d, padding=padding)
        smoothed_map = smoothed_map.squeeze(0).squeeze(0)

        return smoothed_map

    # TODO: deleted after real-time streaming is implemented
    def getSimulatedCompiledLoss_empatch(self, currentParam, currentUserState, newUserTemp=None, therapyMethod=None):
        # Unpack the current user state.
        currentUserTemp = currentParam
        print('entering loss calculation in the getNextStates')
        currentUserLoss = self.dataInterface.calculateCompiledLoss(currentUserState) # torch.Size([1, 1, 1, 1])

        newUserTemp = currentUserTemp if newUserTemp is None else newUserTemp

        # if it is aStarProtocol, resample
        if therapyMethod == 'aStarTherapyProtocol' or 'basicTherapyProtocol' or 'hmmTherapyProtocol':
            resampledParameterBins, resampledPredictionBins = self.generalMethods.resampleBins(self.allParameterBins, self.allPredictionBins, eventlySpacedBins=False)
            # Calculate the bin indices for the current and new user states.

            # doesn't matter to index resampledPrediciton bins or not, it's the same anyway if we are resampling
            currentLossIndex = self.dataInterface.getBinIndex(resampledPredictionBins[0], currentUserLoss)
            currentTempBinIndex = self.dataInterface.getBinIndex(resampledParameterBins[0], currentUserTemp)
            newTempBinIndex = self.dataInterface.getBinIndex(resampledParameterBins[0], newUserTemp)
            newUserLoss, PA, NA, SA = self.sampleNewLoss_empatch(currentUserLoss, currentLossIndex, currentTempBinIndex, newTempBinIndex, currentUserState, therapyMethod, bufferZone=0.01)
            newUserLoss = torch.tensor(newUserLoss).view(1, 1, 1, 1)
            PA = torch.tensor(PA).view(1, 1, 1, 1)
            NA = torch.tensor(NA).view(1, 1, 1, 1)
            SA = torch.tensor(SA).view(1, 1, 1, 1)
            return newUserLoss, PA, NA, SA
        else:
            # for other conditions or protocols
            # Calculate the bin indices for the current and new user states.
            currentLossIndex = self.dataInterface.getBinIndex(self.allPredictionBins, currentUserLoss)
            newTempBinIndex = self.dataInterface.getBinIndex(self.allParameterBins, newUserTemp)
            #newUserLoss, PA, NA, SA = self.sampleNewLoss(currentUserLoss, currentLossIndex, currentTempBinIndex, newTempBinIndex, currentUserState, therapyMethod, bufferZone=0.01)
            # Simulate a new user loss.
            newUserLoss = None
            PA = None
            NA = None
            SA = None
            return newUserLoss, PA, NA, SA

    "Sample a new loss for real data interface"
    def sampleNewLoss_empatch(self, currentUserLoss, currentLossIndex, currentParamBinIndex, newParamIndex, currentUserState, therapyMethod=None, bufferZone=0.01, gausSTD=0.05):
        simulatedMapPA = torch.tensor(self.realSimMapPA, dtype=torch.float32)
        simulatedMapNA = torch.tensor(self.realSimMapNA, dtype=torch.float32)
        simulatedMapSA = torch.tensor(self.realSimMapSA, dtype=torch.float32)
        simulatedMapCompiledLoss = torch.tensor(self.realSimMapCompiledLoss, dtype=torch.float32)
        if therapyMethod == 'aStarTherapyProtocol' or 'basicTherapyProtocol' or 'hmmTherapyProtocol':
            # resampling
            resampledParameterBins, resampledPredictionBins = self.generalMethods.resampleBins(self.allParameterBins, self.allPredictionBins, eventlySpacedBins=False)
            if newParamIndex != currentParamBinIndex or torch.rand(1).item() < 0.1:

                # Calculate new loss probabilities and Gaussian boost
                newLossProbabilities = simulatedMapCompiledLoss[newParamIndex] / torch.sum(simulatedMapCompiledLoss[newParamIndex])

                gaussian_boost = self.generalMethods.createGaussianArray(inputData=newLossProbabilities.numpy(), gausMean=currentLossIndex, gausSTD=gausSTD, torchFlag=True)
                # Combine the two distributions and normalize
                newLossProbabilities = newLossProbabilities + gaussian_boost
                newLossProbabilities = newLossProbabilities / torch.sum(newLossProbabilities)

                # Sample a new loss from the distribution
                newLossBinIndex = torch.multinomial(newLossProbabilities, 1).item()
                newUserLoss = resampledPredictionBins[0][newLossBinIndex]  # doesn't matter which prediction bins, after resampling they are the same.
                # Sample distribution of loss at a certain temperature for PA, NA, SA
                simulatedSpecificMaps = {'PA': simulatedMapPA, 'NA': simulatedMapNA, 'SA': simulatedMapSA}
                specificLossProbabilities = {}
                specificUserLosses = {}

                for key in simulatedSpecificMaps.keys():
                    specificLossProbabilities[key] = simulatedSpecificMaps[key][newParamIndex] / torch.sum(simulatedSpecificMaps[key][newParamIndex])
                    gaussian_boost = self.generalMethods.createGaussianArray(inputData=specificLossProbabilities[key].numpy(), gausMean=currentLossIndex, gausSTD=gausSTD, torchFlag=True)
                    specificLossProbabilities[key] = specificLossProbabilities[key] + gaussian_boost
                    specificLossProbabilities[key] = specificLossProbabilities[key] / torch.sum(specificLossProbabilities[key])

                    # can also sample a new loss for each specific map if needed
                    newSpecificLossBinIndex = torch.multinomial(specificLossProbabilities[key], 1).item()
                    specificUserLosses[key] = resampledPredictionBins[0][newSpecificLossBinIndex]
                    print(f"{key} specific user loss: {specificUserLosses[key]}")

                # specificLossProbabilities contains the normalized loss probabilities for PA, NA, and SA
                return newUserLoss, specificUserLosses['PA'], specificUserLosses['NA'], specificUserLosses['SA']
            else:
                newUserLoss = currentUserLoss + torch.normal(mean=0.0, std=0.01, size=currentUserLoss.size())
                newUserLossPA = currentUserState[0][0][0][0] + torch.normal(mean=0.0, std=0.01, size=currentUserState[0][0][0][0].size())
                newUserLossNA = currentUserState[0][1][0][0] + torch.normal(mean=0.0, std=0.01, size=currentUserState[0][1][0][0].size())
                newUserLossSA = currentUserState[0][2][0][0] + torch.normal(mean=0.0, std=0.01, size=currentUserState[0][2][0][0].size())
                return newUserLoss, newUserLossPA, newUserLossNA, newUserLossSA