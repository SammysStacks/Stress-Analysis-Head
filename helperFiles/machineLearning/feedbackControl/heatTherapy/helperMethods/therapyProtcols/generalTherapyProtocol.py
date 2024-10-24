# General
import torch
import abc

# Import helper files.
from .....modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from .plottingProtocols.plottingProtocolsMain import plottingProtocolsMain
from ..dataInterface.simulationProtocols import simulationProtocols
from ..dataInterface.empatchProtocols import empatchProtocols
from .helperTherapyMethods.generalMethods import generalMethods
from ..dataInterface.dataInterface import dataInterface


class generalTherapyProtocol(abc.ABC):
    def __init__(self, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapyMethod):
        # General parameters.
        self.unNormalizedParameterBinWidths = unNormalizedParameterBinWidths  # The parameter bounds for the therapy.
        self.simulateTherapy = simulationParameters['simulateTherapy']  # Whether to simulate the therapy.
        self.initialParameterBounds = initialParameterBounds  # The parameter bounds for the therapy.
        self.modelParameterBounds = [0, 1]  # The model parameter bounds for the therapy.
        self.applyGaussianFilter = True  # Whether to apply a Gaussian filter on the discrete maps.
        self.finishedTherapy = False  # Whether the therapy has finished.

        # Initialize the hard-coded survey information.
        self.compileModelInfoClass = compileModelInfo()  # The class for compiling model information.
        # Get information from the hard-coded survey information.
        self.predictionBinWidths = self.compileModelInfoClass.standardErrorMeasurements  # using the SEM as bin width for the losses (PA, NA, SA)

        self.optimalPredictions = self.compileModelInfoClass.optimalPredictions  # The bounds for the mental health predictions.
        self.predictionWeights = self.compileModelInfoClass.predictionWeights  # The weights for the loss function. [PA, NA, SA]
        self.predictionBounds = self.compileModelInfoClass.predictionBounds  # The bounds for the mental health predictions.
        self.predictionOrder = self.compileModelInfoClass.predictionOrder  # The order of the mental health predictions.

        # Convert to torch tensors.
        self.unNormalizedParameterBinWidths = torch.tensor(self.unNormalizedParameterBinWidths)  # Dimensions: numParameters
        self.initialParameterBounds = torch.tensor(self.initialParameterBounds)  # Dimensions: numParameters, 2 #i.e.(lower and upper bounds): tensor([35, 50])
        self.modelParameterBounds = torch.tensor(self.modelParameterBounds)  # Dimensions: 2 # tensor([0, 1]) normalized already
        self.predictionBinWidths = torch.tensor(self.predictionBinWidths)  # Dimensions: numPredictions
        self.optimalPredictions = torch.tensor(self.optimalPredictions)  # Dimensions: numPredictions
        self.predictionBounds = torch.tensor(self.predictionBounds)  # Dimensions: numPredictions, 2
        # Get the parameters in the correct data format.
        if self.initialParameterBounds.ndim == 1: self.initialParameterBounds = torch.unsqueeze(self.initialParameterBounds, dim=0) # tensor([[35, 50]])
        if self.predictionBounds.ndim == 1: self.predictionBounds = torch.unsqueeze(self.predictionBounds, dim=0) # ([[5, 25], [5, 25], [20, 80]])
        self.numParameters = len(self.initialParameterBounds)  # The number of parameters. # 1 for now

        # Calculated parameters.
        self.parameterBinWidths = dataInterface.normalizeParameters(currentParamBounds=self.initialParameterBounds - self.initialParameterBounds[:, 0:1], normalizedParamBounds=self.modelParameterBounds, currentParamValues=self.unNormalizedParameterBinWidths) #torch.tensor([0.1])
        self.predictionBinWidths = dataInterface.normalizeParameters(currentParamBounds=self.predictionBounds - self.predictionBounds[:, 0:1], normalizedParamBounds=self.modelParameterBounds, currentParamValues=self.predictionBinWidths)

        self.optimalNormalizedState = dataInterface.normalizeParameters(currentParamBounds=self.predictionBounds, normalizedParamBounds=self.modelParameterBounds, currentParamValues=self.optimalPredictions)  # The optimal normalized state for the therapy. tensor([1, 0, 0])
        self.gausParameterSTDs = self.parameterBinWidths.clone()  # The standard deviation for the Gaussian distribution for parameters.
        self.numPredictions = len(self.optimalNormalizedState)  # The number of losses to predict.
        self.gausLossSTDs = self.predictionBinWidths.clone()  # The standard deviation for the Gaussian distribution for losses.

        # Initialize the loss and parameter bins.
        self.allParameterBins = dataInterface.initializeAllBins(self.modelParameterBounds, self.parameterBinWidths)    # Note this is an UNEVEN 2D list. [[parameter]] bin list
        self.allPredictionBins = dataInterface.initializeAllBins(self.modelParameterBounds, self.predictionBinWidths)  # Note this is an UNEVEN 2D list. [[PA], [NA], [SA]] bin list TODO: the binwidth could be the same, but we can unNormalize differently?
        self.unNormalizedAllParameterBins = dataInterface.initializeAllBins(self.initialParameterBounds, self.unNormalizedParameterBinWidths.unsqueeze(0))  # Note this is an UNEVEN 2D list. [[parameter]] bin list
        # print('allPredictionBins: ', self.allPredictionBins)

        # Initialize the number of bins for the parameter and loss.
        self.allNumParameterBins = [len(self.allParameterBins[parameterInd]) for parameterInd in range(self.numParameters)]  # Parameter number of Bins in the list
        self.allNumPredictionBins = [len(self.allPredictionBins[lossInd]) for lossInd in range(self.numPredictions)]  #PA, NA, SA number of bins in the list

        # Define a helper class for experimental parameters.
        self.simulationProtocols = simulationProtocols(self.allParameterBins, self.allPredictionBins, self.predictionBinWidths, self.modelParameterBounds, self.numPredictions, self.numParameters, self.predictionWeights, self.optimalNormalizedState, self.initialParameterBounds, self.unNormalizedAllParameterBins, simulationParameters)
        self.plottingProtocolsMain = plottingProtocolsMain(self.initialParameterBounds, self.modelParameterBounds, self.allNumParameterBins, self.parameterBinWidths, self.predictionBounds, self.allNumPredictionBins, self.predictionBinWidths)
        self.empatchProtocols = empatchProtocols(self.predictionOrder, self.predictionBounds, self.modelParameterBounds, therapyExpMethod='HeatingPad')
        self.dataInterface = dataInterface(self.predictionWeights, self.optimalNormalizedState)
        self.generalMethods = generalMethods()


        # Reset the therapy parameters.
        self.userMentalStatePath = None
        self.paramStatePath = None
        self.timepoints = None
        self.userMentalStateCompiledLoss = None
        self.userName = None
        self.unNormalizedParameter = None
        self.resetTherapy()



    def resetTherapy(self):
        # Reset the therapy parameters.
        self.userMentalStatePath = []  # The path of the user's mental state: PA, NA, SA
        self.finishedTherapy = False  # Whether the therapy has finished.
        self.paramStatePath = []  # The path of the therapy parameters: numParameters
        self.timepoints = [] # The time points for the therapy.
        self.userMentalStateCompiledLoss = []  # The compiled loss for the user's mental state.
        self.userName = []  # The user's name.
        self.unNormalizedParameter = []
        # Reset the therapy maps.
        self.initializeMaps()

    # ------------------------ Track User States ------------------------ #
    def initializeMaps(self):
        if self.simulateTherapy:
            self.simulationProtocols.initializeSimulatedMaps(self.predictionWeights, self.gausParameterSTDs, self.gausLossSTDs, self.applyGaussianFilter)
        else:
            # real data points
            temperature, pa, na, sa = self.empatchProtocols.getTherapyData()
            # sort the temperature, pa, na, sa into correct format passed to generate initialSimulatedData
            initialSimulatedStates = torch.stack([temperature, pa, na, sa], dim=1)
            extracted_param = initialSimulatedStates[0, 0, :]
            # normalize the temperature
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

            # resample the data bins
            resampledParameterBins, resampledPredictionBins = self.generalMethods.resampleBins(self.allParameterBins, self.allPredictionBins, eventlySpacedBins=False)

            self.simulationProtocols.realSimMapPA = self.generalMethods.getProbabilityMatrix(initialRealData_PA, resampledParameterBins, resampledPredictionBins[0], self.gausParameterSTDs, self.gausLossSTDs[0], noise=0.1, applyGaussianFilter=self.applyGaussianFilter)
            self.simulationProtocols.realSimMapNA = self.generalMethods.getProbabilityMatrix(initialRealData_NA, resampledParameterBins, resampledPredictionBins[1], self.gausParameterSTDs, self.gausLossSTDs[1], noise=0.1, applyGaussianFilter=self.applyGaussianFilter)
            self.simulationProtocols.realSimMapSA = self.generalMethods.getProbabilityMatrix(initialRealData_SA, resampledParameterBins, resampledPredictionBins[2], self.gausParameterSTDs, self.gausLossSTDs[2], noise=0.1, applyGaussianFilter=self.applyGaussianFilter)

            # say that state anxiety has a slightly higher weight
            self.simulationProtocols.realSimMapCompiledLoss = 0.3 * self.simulationProtocols.realSimMapPA + 0.3 * self.simulationProtocols.realSimMapNA + 0.4 * self.simulationProtocols.realSimMapSA

    def initializeUserState(self, userName):
        # Get the user information.
        # timePints: a tensor
        # parameter: tensor of size 1, 1, 1, 1
        # emotionStates: tensor of size 1, 3, 1, 1
        timepoints, parameters, emotionStates = self.getInitialSate()  # dim: numPoints, timePoint: t; emotionStates: (PA, NA, SA); prediction: predict the next state; Note these are actual state values
        # Track the user state and time delay.
        startTimePoint = timepoints

        self.timepoints.append(startTimePoint) # timepoints: list of tensor: [tensor(0)]
        self.paramStatePath.append(parameters) # self.paramStatePath: list of tensor: [torch.Size([1, 1, 1, 1])
        self.userMentalStatePath.append(emotionStates) # emotionstates: list of tensor: torch.Size([1, 3, 1, 1])
        # Calculate the initial user loss.
        compiledLoss = self.dataInterface.calculateCompiledLoss(emotionStates[-1])  # compile the loss state for the current emotion state; torch.Size([1, 1, 1, 1])
        self.userMentalStateCompiledLoss.append(compiledLoss) #  list of tensor torch.Size([1, 1, 1, 1])
        self.userName.append(userName) # list: username
        initialUserParamBinIndex = self.dataInterface.getBinIndex(self.allParameterBins, parameters)
        initialUserParam = self.unNormalizedAllParameterBins[0][initialUserParamBinIndex]  # bound the initial temperature (1D)
        initialUserParam = self.boundNewTemperature(initialUserParam, bufferZone=0.01)  # bound the initial temperature (1D)
        self.unNormalizedParameter.append(initialUserParam) # list of tensor torch.Size([1, 1, 1, 1])





    def getInitialSate(self):
        if self.simulateTherapy:
            # Simulate a new time point by adding a constant delay factor.
            currentTime, currentParam, currentPredictions = self.simulationProtocols.getInitialState() # currentTime: tensor(0); currentParam: torch.Size([1, 1, 1, 1]); currentPredictions: torch.Size([1, 3, 1, 1]) predefined.
            print('currentTime, currentParam, currentPredictions', currentTime, currentParam, currentPredictions)
            return currentTime, currentParam, currentPredictions
        else:
            # TODO: !!! Implement a method to get the current user state.
            # TODO: right now just simulate a random start state, eventually this has to be real time start state
            currentTime, currentParam, currentPredictions = self.simulationProtocols.getInitialState()  # currentTime: tensor(0); currentParam: torch.Size([1, 1, 1, 1]); currentPredictions: torch.Size([1, 3, 1, 1]) predefined.
            print('currentTime, currentParam, currentPredictions', currentTime, currentParam, currentPredictions)
            return currentTime, currentParam, currentPredictions

        # Returning timePoint, (T, PA, NA, SA)

    def getNextState(self, newParamValues, therapyMethod):
        if self.simulateTherapy:
            # Simulate a new time.
            lastTimePoint = self.timepoints[-1] if len(self.timepoints) != 0 else 0
            # convert tensor to int
            lastTimePoint = int(lastTimePoint)
            newTimePoint = self.simulationProtocols.getSimulatedTimes(self.simulationProtocols.initialPoints, lastTimePoint)
            # get the current user state
            currentParam = self.paramStatePath[-1]
            currentEmotionStates = self.userMentalStatePath[-1]
            # Sample the new loss form a pre-simulated map.
            newUserLoss, PA, NA, SA = self.simulationProtocols.getSimulatedCompiledLoss(currentParam, currentEmotionStates, newParamValues, therapyMethod)
            # combined mentalstate
            print('newUserLoss, PA, NA, SA', newUserLoss, PA, NA, SA)
            combinedMentalState = torch.cat((PA, NA, SA), dim=1)
            print(f'newuSerparam,newuserLoss {newParamValues, newUserLoss}')
            # unbound temperature:
            param_state_index = self.dataInterface.getBinIndex(self.allParameterBins[0], newParamValues)
            param_state_unbound = self.unNormalizedAllParameterBins[0][param_state_index]
            param_state_unbound = self.boundNewTemperature(param_state_unbound, bufferZone=0.01)  # newUserParam = torch.Size([1, 1, 1, 1])
            print('param_state_unbound', param_state_unbound)
            # User state update

            self.timepoints.append(newTimePoint)
            self.paramStatePath.append(newParamValues)
            self.userMentalStatePath.append(combinedMentalState)
            self.userMentalStateCompiledLoss.append(newUserLoss)
            self.unNormalizedParameter.append(param_state_unbound)
        else:
            # TODO !!! eventually, this will be the real-time user state update during experiment
            # Simulate a new time.
            lastTimePoint = self.timepoints[-1] if len(self.timepoints) != 0 else 0
            # convert tensor to int
            lastTimePoint = int(lastTimePoint)
            newTimePoint = self.simulationProtocols.getSimulatedTimes(self.simulationProtocols.initialPoints, lastTimePoint)
            # get the current user state
            currentParam = self.paramStatePath[-1]
            currentEmotionStates = self.userMentalStatePath[-1]
            # Sample the new loss form a pre-simulated map.
            newUserLoss, PA, NA, SA = self.simulationProtocols.getSimulatedCompiledLoss_empatch(currentParam, currentEmotionStates, newParamValues, therapyMethod)
            # combined mentalstate
            print('newUserLoss, PA, NA, SA', newUserLoss, PA, NA, SA)
            combinedMentalState = torch.cat((PA, NA, SA), dim=1)
            print(f'newuSerparam,newuserLoss {newParamValues, newUserLoss}')
            # unbound temperature:
            param_state_index = self.dataInterface.getBinIndex(self.allParameterBins[0], newParamValues)
            param_state_unbound = self.unNormalizedAllParameterBins[0][param_state_index]
            param_state_unbound = self.boundNewTemperature(param_state_unbound, bufferZone=0.01)  # newUserParam = torch.Size([1, 1, 1, 1])
            print('param_state_unbound', param_state_unbound)
            # User state update

            self.timepoints.append(newTimePoint)
            self.paramStatePath.append(newParamValues)
            self.userMentalStatePath.append(combinedMentalState)
            self.userMentalStateCompiledLoss.append(newUserLoss)
            self.unNormalizedParameter.append(param_state_unbound)

    def boundNewTemperature(self, newUserParam, bufferZone=0.01):
        # Bound the new temperature.
        # TODO: current implementation only for heat therapy (1D), so we extract the parameter bounds at the 1st dimension. For music therapy, we need both dimensions
        newUserTemp = max((self.initialParameterBounds[0][0]).numpy() + bufferZone, min((self.initialParameterBounds[0][1]).numpy() - bufferZone, newUserParam))
        newUserTemp = torch.tensor(newUserTemp).view(1, 1, 1, 1)
        return newUserTemp

    def checkConvergence(self, maxIterations):
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

            # Check if currentProbability is greater than 90% of other probabilities under currentParam
            param_probs = probabilityMap[currentParamIndex]
            param_quantile = torch.quantile(param_probs, 0.9)

            # Check if currentProbability is greater than 90% of probabilities for all parameters under the same loss bin
            loss_probs = probabilityMap[:, currentLossIndex]
            loss_quantile = torch.quantile(loss_probs, 0.9)

            if currentCompiledLoss < 0.2:
                if currentProbability > param_quantile and currentProbability > loss_quantile:
                    self.finishedTherapy = True

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


    # ------------------------ Child Class Contract ------------------------ #

    @abc.abstractmethod
    def updateTherapyState(self):
        """ Create contract for child class method """
        raise NotImplementedError("Must override in child")
