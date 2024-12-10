# General
import torch
import abc

# Import helper files.
from .....modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from .plottingProtocols.plottingProtocolsMain import plottingProtocolsMain
from ..dataInterface.simulationProtocols import simulationProtocols
from .helperTherapyMethods.generalMethods import generalMethods
from ..dataInterface.empatchProtocols import empatchProtocols
from ..dataInterface.dataInterface import dataInterface


class generalTherapyProtocol(abc.ABC):
    def __init__(self, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapySelection, therapyMethod):
        # ================================================General Parameters================================================
        self.unNormalizedParameterBinWidths = unNormalizedParameterBinWidths  # The parameter bounds for the therapy.
        self.simulateTherapy = simulationParameters['simulateTherapy']  # Whether to simulate the therapy.
        self.initialParameterBounds = initialParameterBounds  # The parameter bounds for the therapy.
        self.modelParameterBounds = [0, 1]  # The model parameter bounds for the therapy.
        self.applyGaussianFilter = False  # Whether to apply a Gaussian filter on the discrete maps.
        self.finishedTherapy = False  # Whether the therapy has finished.

        # ====================================================Model Class===================================================
        self.compileModelInfoClass = compileModelInfo()  # The class for compiling model information.

        # ====================================================General Info==================================================
        self.predictionBinWidths = self.compileModelInfoClass.standardErrorMeasurements  # using the SEM as bin width for the losses (PA, NA, SA)
        self.optimalPredictions = self.compileModelInfoClass.optimalPredictions  # The bounds for the mental health predictions.
        self.predictionWeights = self.compileModelInfoClass.predictionWeights  # The weights for the loss function. [PA, NA, SA]
        self.predictionBounds = self.compileModelInfoClass.predictionBounds  # The bounds for the mental health predictions.
        self.predictionOrder = self.compileModelInfoClass.predictionOrder  # The order of the mental health predictions.

        # =================================================Tensor Interface================================================
        self.unNormalizedParameterBinWidths = torch.tensor(self.unNormalizedParameterBinWidths)  # Dimensions: numParameters
        self.initialParameterBounds = torch.tensor(self.initialParameterBounds)  # Dimensions: (numParameters, 2) --> i.e.(lower and upper bounds): tensor([low, high]) depending on therapy
        self.modelParameterBounds = torch.tensor(self.modelParameterBounds)  # Dimensions: 2 --> tensor([0, 1]) normalized
        self.predictionBinWidths = torch.tensor(self.predictionBinWidths)  # Dimensions: numPredictions
        self.optimalPredictions = torch.tensor(self.optimalPredictions)  # Dimensions: numPredictions
        self.predictionBounds = torch.tensor(self.predictionBounds)  # Dimensions: numPredictions, 2

        # ====================================================Data Format==================================================
        if self.initialParameterBounds.ndim == 1: self.initialParameterBounds = torch.unsqueeze(self.initialParameterBounds, dim=0)  # tensor([[low, high]]) depending on therapy
        if self.predictionBounds.ndim == 1: self.predictionBounds = torch.unsqueeze(self.predictionBounds, dim=0)  # Hardcoded: ([[5, 25], [5, 25], [20, 80]])
        self.numParameters = len(self.initialParameterBounds)  # The number of parameters.

        # ============================================Specific Calculated Params===========================================
        self.predictionBinWidths = dataInterface.normalizeParameters(currentParamBounds=self.predictionBounds - self.predictionBounds[:, 0:1], normalizedParamBounds=self.modelParameterBounds, currentParamValues=self.predictionBinWidths)
        self.optimalNormalizedState = dataInterface.normalizeParameters(currentParamBounds=self.predictionBounds, normalizedParamBounds=self.modelParameterBounds, currentParamValues=self.optimalPredictions)
        self.numPredictions = len(self.optimalNormalizedState)  # The number of losses to predict.
        self.therapySelection = therapySelection

        # ==========================================Helper Method Initializations==========================================
        self.dataInterface = dataInterface(self.predictionWeights, self.optimalNormalizedState)
        self.generalMethods = generalMethods()

        # ==============================================Calculated Parameters==============================================
        if self.therapySelection == 'Heat':
            # Interfacing with real data
            self.empatchProtocols = empatchProtocols(self.predictionOrder, self.predictionBounds, self.modelParameterBounds, therapyExpMethod='HeatingPad')
            self.parameterBinWidths = dataInterface.normalizeParameters(currentParamBounds=self.initialParameterBounds - self.initialParameterBounds[:, 0:1], normalizedParamBounds=self.modelParameterBounds,
                                                                        currentParamValues=self.unNormalizedParameterBinWidths)

            # Initialize the loss and parameter bins.
            self.unNormalizedAllParameterBins = dataInterface.initializeAllBins(self.initialParameterBounds, self.unNormalizedParameterBinWidths.unsqueeze(0))  # UNEVEN 2D list. [[parameter]] bin list
            self.allParameterBins = dataInterface.initializeAllBins(self.modelParameterBounds, self.parameterBinWidths)  # UNEVEN 2D list. [[parameter]] bin list
            self.allPredictionBins = dataInterface.initializeAllBins(self.modelParameterBounds, self.predictionBinWidths)  # UNEVEN 2D list. [[PA], [NA], [SA]] bin list

            # Initialize the number of bins for the parameter and loss.
            self.allNumParameterBins = [len(self.allParameterBins[parameterInd]) for parameterInd in range(self.numParameters)]  # Parameter number of Bins in the list

        elif self.therapySelection == 'BinauralBeats':
            # Interfacing with real data
            self.empatchProtocols = empatchProtocols(self.predictionOrder, self.predictionBounds, self.modelParameterBounds, therapyExpMethod='BinauralBeats')

            # Calculate separate Param Bounds
            selectedInitialParamBounds_1 = self.initialParameterBounds[0].unsqueeze(0)
            selectedInitialParamBounds_2 = self.initialParameterBounds[1].unsqueeze(0)
            self.parameterBinWidths_1 = dataInterface.normalizeParameters(currentParamBounds=selectedInitialParamBounds_1 - selectedInitialParamBounds_1[:, 0:1], normalizedParamBounds=self.modelParameterBounds,
                                                                          currentParamValues=self.unNormalizedParameterBinWidths)
            self.parameterBinWidths_2 = dataInterface.normalizeParameters(currentParamBounds=selectedInitialParamBounds_2 - selectedInitialParamBounds_2[:, 0:1], normalizedParamBounds=self.modelParameterBounds,
                                                                          currentParamValues=self.unNormalizedParameterBinWidths)
            self.unNormalizedAllParameterBins_1 = dataInterface.initializeAllBins(self.initialParameterBounds[0], self.unNormalizedParameterBinWidths.unsqueeze(0))
            self.unNormalizedAllParameterBins_2 = dataInterface.initializeAllBins(self.initialParameterBounds[1], self.unNormalizedParameterBinWidths.unsqueeze(0))
            self.parameterBinWidths = torch.cat((self.parameterBinWidths_1.unsqueeze(1), self.parameterBinWidths_2.unsqueeze(1)), dim=0)
            self.allParameterBins_1 = dataInterface.initializeAllBins(self.modelParameterBounds, self.parameterBinWidths[0])
            self.allParameterBins_2 = dataInterface.initializeAllBins(self.modelParameterBounds, self.parameterBinWidths[1])
            self.allParameterBins = [self.allParameterBins_1, self.allParameterBins_2]
            self.allPredictionBins = dataInterface.initializeAllBins(self.modelParameterBounds, self.predictionBinWidths)

            # Concatenation
            self.unNormalizedAllParameterBins = [self.unNormalizedAllParameterBins_1, self.unNormalizedAllParameterBins_2]
            # Initialize the number of bins for the parameter and loss.
            self.allNumParameterBins = [len(self.allParameterBins[parameterInd][0]) for parameterInd in range(self.numParameters)]

        # ===============================================Gaussian Adjustment===============================================
        self.gausParameterSTDs = self.parameterBinWidths.clone()  # The standard deviation for the Gaussian distribution for parameters.
        self.gausLossSTDs = self.predictionBinWidths.clone()  # The standard deviation for the Gaussian distribution for losses.
        self.allNumPredictionBins = [len(self.allPredictionBins[lossInd]) for lossInd in range(self.numPredictions)]  # PA, NA, SA number of bins in the list

        # ===========================================Helper Class for Exp. Params==========================================
        """For Binaural Beats therapy, simulation is done in the same dimension as Heat therapy due to the Binaural Beats constraints"""
        self.simulationProtocols = simulationProtocols(self.allParameterBins, self.allPredictionBins, self.predictionBinWidths, self.modelParameterBounds, self.numPredictions, self.numParameters, self.predictionWeights, self.optimalNormalizedState, self.initialParameterBounds[0], self.unNormalizedAllParameterBins[0], simulationParameters, therapySelection)
        self.plottingProtocolsMain = plottingProtocolsMain(self.initialParameterBounds, self.modelParameterBounds, self.allNumParameterBins, self.parameterBinWidths, self.predictionBounds, self.allNumPredictionBins, self.predictionBinWidths)

        # ================================================Reset Parameters=================================================
        self.userMentalStatePath = None
        self.paramStatePath = None
        self.timepoints = None
        self.userMentalStateCompiledLoss = None
        self.userName = None
        self.unNormalizedParameter = None
        self.unNormalizedParam_1 = None
        self.unNormalizedParam_2 = None
        self.resetTherapy(therapySelection)

    """Reset Parameters"""
    def resetTherapy(self, therapySelection):
        self.userMentalStatePath = []  # The path of the user's mental state: PA, NA, SA
        self.finishedTherapy = False
        self.paramStatePath = []  # The path of the therapy parameters: numParameters
        self.timepoints = []  # The time points for the therapy.
        self.userMentalStateCompiledLoss = []  # The compiled loss for the user's mental state.
        self.userName = []  # The user's name.
        self.unNormalizedParameter = []

        # Only for BinauralBeats Therapy
        self.unNormalizedParam_1 = []
        self.unNormalizedParam_2 = []

        # Reset the therapy maps.
        self.initializeMaps(therapySelection)

    """Track User States"""
    def initializeMaps(self, therapySelection):
        if self.simulateTherapy:
            if therapySelection == "Heat":
                self.simulationProtocols.initializeSimulatedMaps(self.predictionWeights, self.gausParameterSTDs, self.gausLossSTDs, self.applyGaussianFilter)
            elif therapySelection == "BinauralBeats":
                self.simulationProtocols.initializeSimulatedMaps(self.predictionWeights, self.gausParameterSTDs, self.gausLossSTDs, self.applyGaussianFilter)
        else:
            # Real data points
            temperature, pa, na, sa = self.empatchProtocols.getTherapyData()

            # Sort the Param, PA, NA, SA into correct format to generate initialSimulatedData
            initialSimulatedStates = torch.stack([temperature, pa, na, sa], dim=1)
            extracted_param = initialSimulatedStates[0, 0, :]

            # Normalize the Params
            extracted_param = dataInterface.normalizeParameters(currentParamBounds=self.initialParameterBounds, normalizedParamBounds=self.modelParameterBounds, currentParamValues=extracted_param)
            extracted_pa = initialSimulatedStates[0, 1, :]
            extracted_na = initialSimulatedStates[0, 2, :]
            extracted_sa = initialSimulatedStates[0, 3, :]

            # Stack the PA, NA, and SA into the desired shape
            realDataInitialEmotionStates = torch.stack([extracted_pa, extracted_na, extracted_sa], dim=1).unsqueeze(2).unsqueeze(3)  # [48, 3, 1, 1]

            initialRealData_compiledLoss = self.dataInterface.calculateCompiledLoss(realDataInitialEmotionStates)  # initialSimulatedData dimension: numSimulationTrueSamples, (T, L).
            realDataInitialCompiledStates = torch.stack([extracted_param, extracted_pa, extracted_na, extracted_sa, initialRealData_compiledLoss.flatten()], dim=1).unsqueeze(2).unsqueeze(3)  # [48, 5, 1, 1]

            # data preprocessing for generating intial Maps
            initialRealData_PA = realDataInitialCompiledStates[:, [0, 1], :, :]  # Shape: [30, 2, 1, 1]

            # Extract for NA: Parameter and second emotion prediction
            initialRealData_NA = realDataInitialCompiledStates[:, [0, 2], :, :]  # Shape: [30, 2, 1, 1]

            # Extract for SA: Parameter and third emotion prediction
            initialRealData_SA = realDataInitialCompiledStates[:, [0, 3], :, :]  # Shape: [30, 2, 1, 1]

            # Resample the data bins to address unevenness
            resampledParameterBins, resampledPredictionBins = self.generalMethods.resampleBins(self.allParameterBins, self.allPredictionBins, eventlySpacedBins=False)

            # Generate Prob Matrix
            self.simulationProtocols.realSimMapPA = self.generalMethods.getProbabilityMatrix(initialRealData_PA, resampledParameterBins, resampledPredictionBins[0], self.gausParameterSTDs, self.gausLossSTDs[0], noise=0.1, applyGaussianFilter=self.applyGaussianFilter)
            self.simulationProtocols.realSimMapNA = self.generalMethods.getProbabilityMatrix(initialRealData_NA, resampledParameterBins, resampledPredictionBins[1], self.gausParameterSTDs, self.gausLossSTDs[1], noise=0.1, applyGaussianFilter=self.applyGaussianFilter)
            self.simulationProtocols.realSimMapSA = self.generalMethods.getProbabilityMatrix(initialRealData_SA, resampledParameterBins, resampledPredictionBins[2], self.gausParameterSTDs, self.gausLossSTDs[2], noise=0.1, applyGaussianFilter=self.applyGaussianFilter)

            # Weigh the Maps
            self.simulationProtocols.realSimMapCompiledLoss = 0.3 * self.simulationProtocols.realSimMapPA + 0.3 * self.simulationProtocols.realSimMapNA + 0.4 * self.simulationProtocols.realSimMapSA

    """Initialize the User information"""
    def initializeUserState(self, userName, initialTime, initialParam, initialPredicitons):
        # --------------------------------------Dimensions--------------------------------------
        # timePints: a tensor
        # parameter: tensor of size 1, 1, 1, 1 for heat and tensor of size 1, 2, 1, 1 for Binaural Beats
        # emotionStates: tensor of size 1, 3, 1, 1
        # --------------------------------------------------------------------------------------

        timepoints, parameters, emotionStates = self.getInitialSate(initialTime, initialParam, initialPredicitons)

        # Check if any of the initial state components are None
        if timepoints is None or parameters is None or emotionStates is None:
            return

        # Track the user state and time delay.
        startTimePoint = timepoints

        # Update the Holders
        self.userMentalStatePath.append(emotionStates)  # emotionstates: list of tensor: torch.Size([1, 3, 1, 1])
        self.timepoints.append(startTimePoint)  # timepoints: list of tensor: [tensor(0)]
        self.paramStatePath.append(parameters)  # self.paramStatePath: list of tensor: [torch.Size([1, 1, 1, 1])

        # Calculate the initial user loss.
        compiledLoss = self.dataInterface.calculateCompiledLoss(emotionStates[-1])  # compile the loss state for the current emotion state; torch.Size([1, 1, 1, 1])
        self.userMentalStateCompiledLoss.append(compiledLoss)  # list of tensor torch.Size([1, 1, 1, 1])
        self.userName.append(userName)  # list: username

        # Therapy Specific Calculations
        if self.therapySelection == 'Heat':
            initialUserParamBinIndex = self.dataInterface.getBinIndex(self.allParameterBins, parameters)
            initialUserParam = self.unNormalizedAllParameterBins[0][initialUserParamBinIndex]  # bound the initial temperature (1D)
            initialUserParam = self.boundNewTemperature(initialUserParam, bufferZone=0.01)  # bound the initial temperature (1D)
            self.unNormalizedParameter.append(initialUserParam)  # list of tensor torch.Size([1, 1, 1, 1])

        elif self.therapySelection == 'BinauralBeats':
            initialUserParamBinIndex_1 = self.dataInterface.getBinIndex(self.allParameterBins[0], parameters[:, 0])
            initialUserParamBinIndex_2 = self.dataInterface.getBinIndex(self.allParameterBins[1], parameters[:, 1])
            initialUserParam_1 = self.unNormalizedAllParameterBins[0][0][initialUserParamBinIndex_1]
            initialUserParam_2 = self.unNormalizedAllParameterBins[1][0][initialUserParamBinIndex_2]
            initialUserParam_1 = self.boundNewTemperature(initialUserParam_1, bufferZone=0.01)
            initialUserParam_2 = self.boundNewTemperature(initialUserParam_2, bufferZone=0.01)
            initialUserParam = [initialUserParam_1, initialUserParam_2]
            self.unNormalizedParameter.append(initialUserParam)
            self.unNormalizedParam_1.append(initialUserParam_1)
            self.unNormalizedParam_2.append(initialUserParam_2)

    """Initialize the state information"""
    def getInitialSate(self, initialTime, initialParam, initialPredicitons):
        if self.simulateTherapy:
            # Simulate a new time point by adding a constant delay factor.
            currentTime, currentParam, currentPredictions = self.simulationProtocols.getInitialState() # currentTime: tensor(0); currentParam: torch.Size([1, 1, 1, 1]); currentPredictions: torch.Size([1, 3, 1, 1]) predefined.
            print('currentTime, currentParam, currentPredictions', currentTime, currentParam, currentPredictions)
            return currentTime, currentParam, currentPredictions
        else:
            # Real-time interfacing
            # Assumption: random start state
            currentTime, currentParam, currentPredictions = self.simulationProtocols.getInitialState()
            #currentTime, currentParam, currentPredictions = initialTime, initialParam, initialPredicitons  # currentTime: tensor(0); currentParam: torch.Size([1, 1, 1, 1]); currentPredictions: torch.Size([1, 3, 1, 1]) predefined.
            # print('currentTime, currentParam, currentPredictions', currentTime, currentParam, currentPredictions)
            return currentTime, currentParam, currentPredictions

    """Therapy state updates"""
    def getNextState(self, newParamValues, therapyMethod):
        if self.simulateTherapy:
            # Simulate a new time.
            lastTimePoint = self.timepoints[-1] if len(self.timepoints) != 0 else 0

            # Convert tensor to int
            lastTimePoint = int(lastTimePoint)
            newTimePoint = self.simulationProtocols.getSimulatedTimes(self.simulationProtocols.initialPoints, lastTimePoint)

            # Get the current user state
            currentParam = self.paramStatePath[-1]
            currentEmotionStates = self.userMentalStatePath[-1]

            # Sample the new loss form a pre-simulated map.
            newUserLoss, PA, NA, SA = self.simulationProtocols.getSimulatedCompiledLoss(currentParam, currentEmotionStates, newParamValues, therapyMethod)

            # Combined to mentalState
            combinedMentalState = torch.cat((PA, NA, SA), dim=1)

            # Therapy-specific calculations
            if self.therapySelection == 'Heat':
                param_state_index = self.dataInterface.getBinIndex(self.allParameterBins[0], newParamValues)
                param_state_unbound = self.unNormalizedAllParameterBins[0][param_state_index]
                param_state_unbound = self.boundNewTemperature(param_state_unbound, bufferZone=0.01)

                # User state update
                self.timepoints.append(newTimePoint)
                self.paramStatePath.append(newParamValues)
                self.userMentalStatePath.append(combinedMentalState)
                self.userMentalStateCompiledLoss.append(newUserLoss)
                self.unNormalizedParameter.append(param_state_unbound)

            elif self.therapySelection == 'BinauralBeats':
                param_state_index_1 = self.dataInterface.getBinIndex(self.allParameterBins[0], newParamValues[:, 0])
                param_state_index_2 = self.dataInterface.getBinIndex(self.allParameterBins[0], newParamValues[:, 1])

                param_state_unbound_1 = self.unNormalizedAllParameterBins[0][0][param_state_index_1]
                param_state_unbound_2 = self.unNormalizedAllParameterBins[1][0][param_state_index_2]

                param_state_unbound_1 = self.boundNewTemperature(param_state_unbound_1, bufferZone=0.01)
                param_state_unbound_2 = self.boundNewTemperature(param_state_unbound_2, bufferZone=0.01)
                param_state_unbound = [param_state_unbound_1, param_state_unbound_2]

                # User state update
                self.unNormalizedParameter.append(param_state_unbound)
                self.unNormalizedParam_1.append(param_state_unbound_1)
                self.unNormalizedParam_2.append(param_state_unbound_2)
                self.userMentalStatePath.append(combinedMentalState)
                self.userMentalStateCompiledLoss.append(newUserLoss)
                self.paramStatePath.append(newParamValues)
                self.timepoints.append(newTimePoint)

        else:
            # TODO: eventually, this will be the real-time user state update during experiment
            # Simulate a new time.
            lastTimePoint = self.timepoints[-1] if len(self.timepoints) != 0 else 0

            # Convert tensor to int
            lastTimePoint = int(lastTimePoint)
            newTimePoint = self.simulationProtocols.getSimulatedTimes(self.simulationProtocols.initialPoints, lastTimePoint)

            # Get the current user state
            currentParam = self.paramStatePath[-1]
            currentEmotionStates = self.userMentalStatePath[-1]

            # Sample the new loss form a pre-simulated map.
            newUserLoss, PA, NA, SA = self.simulationProtocols.getSimulatedCompiledLoss_empatch(currentParam, currentEmotionStates, newParamValues, therapyMethod)

            # Combined to mentalState
            combinedMentalState = torch.cat((PA, NA, SA), dim=1)

            # Unbound temperature:
            param_state_index = self.dataInterface.getBinIndex(self.allParameterBins[0], newParamValues)
            param_state_unbound = self.unNormalizedAllParameterBins[0][param_state_index]
            param_state_unbound = self.boundNewTemperature(param_state_unbound, bufferZone=0.01)

            # User state update
            self.unNormalizedParameter.append(param_state_unbound)
            self.userMentalStatePath.append(combinedMentalState)
            self.userMentalStateCompiledLoss.append(newUserLoss)
            self.paramStatePath.append(newParamValues)
            self.timepoints.append(newTimePoint)

    """Bound the temperatures"""
    def boundNewTemperature(self, newUserParam, bufferZone=0.01):
        # Bound the new temperature.
        newUserTemp = max((self.initialParameterBounds[0][0]).numpy() + bufferZone, min((self.initialParameterBounds[0][1]).numpy() - bufferZone, newUserParam))
        newUserTemp = torch.tensor(newUserTemp).view(1, 1, 1, 1)
        return newUserTemp

    """Convergence check"""
    def checkConvergence(self, maxIterations):
        # Check if the therapy has converged.
        if maxIterations is not None:
            if len(self.userMentalStatePath) >= maxIterations:
                self.finishedTherapy = True
                print(f'Therapy converged at iterations: {maxIterations}')
        else:
            currentParam = self.paramStatePath[-1].item()
            currentCompiledLoss = self.userMentalStateCompiledLoss[-1].item()

            # Get the corresponding bin index for currentCompiledLoss and currentParam
            currentParamIndex = self.dataInterface.getBinIndex(self.allParameterBins[0], currentParam)
            currentLossIndex = self.dataInterface.getBinIndex(self.allPredictionBins[0], currentCompiledLoss)
            if self.simulateTherapy:
                probabilityMap = self.simulationProtocols.simulatedMapCompiledLoss
            else:
                probabilityMap = self.simulationProtocols.realSimMapCompiledLoss
            currentProbability = probabilityMap[currentParamIndex][currentLossIndex]

            # Check if currentProbability is greater than 90% of other probabilities under currentParam
            param_probs = probabilityMap[currentParamIndex]
            param_quantile = torch.quantile(param_probs, 0.9)

            # Check if currentProbability is greater than 90% of probabilities for all parameters under the same loss bin
            loss_probs = probabilityMap[:, currentLossIndex]
            loss_quantile = torch.quantile(loss_probs, 0.9)

            if currentCompiledLoss < 0.2:
                if currentProbability > param_quantile and currentProbability > loss_quantile:
                    self.finishedTherapy = True
                    print(f'Therapy converged at currentProbability {currentProbability}')

    """Convergence Check"""
    def checkConvergence_hmm(self, maxIterations):
        # Check if the therapy has converged.
        if maxIterations is not None:
            if len(self.userMentalStatePath) >= maxIterations:
                self.finishedTherapy = True
        else:
            currentParam = self.paramStatePath[-1].item()
            currentCompiledLoss = self.userMentalStateCompiledLoss[-1].item()

            # Get the corresponding bin index for currentCompiledLoss and currentParam
            currentParamIndex = self.dataInterface.getBinIndex(self.allParameterBins[0], currentParam)
            currentLossIndex = self.dataInterface.getBinIndex(self.allPredictionBins[0], currentCompiledLoss)
            if self.simulateTherapy:
                probabilityMap = self.simulationProtocols.simulatedMapCompiledLoss
            else:
                probabilityMap = self.simulationProtocols.realSimMapCompiledLoss
            currentProbability = probabilityMap[currentParamIndex][currentLossIndex]

            # Check if currentProbability is greater than 80% of other probabilities under currentParam
            param_probs = probabilityMap[currentParamIndex]
            param_quantile = torch.quantile(param_probs, 0.8)

            # Check if currentProbability is greater than 80% of probabilities for all parameters under the same loss bin
            loss_probs = probabilityMap[:, currentLossIndex]
            loss_quantile = torch.quantile(loss_probs, 0.8)

            if currentCompiledLoss < 0.2:
                if currentProbability > param_quantile and currentProbability > loss_quantile:
                    self.finishedTherapy = True


    # ===============================================Child Class Contract================================================
    @abc.abstractmethod
    def updateTherapyState(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")
