# Import the necessary libraries.
from helperFiles.machineLearning.feedbackControl.heatAndMusicTherapy.therapyHelpers import therapyHelpers


class heatTherapyControl(therapyHelpers):
    def __init__(self, userName, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapySelection, therapyMethod, plotResults=False):
        super().__init__(userName=userName, initialParameterBounds=initialParameterBounds, unNormalizedParameterBinWidths=unNormalizedParameterBinWidths,
                         simulationParameters=simulationParameters, therapySelection=therapySelection,therapyMethod=therapyMethod, plotResults=plotResults)

    def runTherapyProtocol(self, maxIterations=None):
        # Initialize holder parameters such as the user maps.
        self.therapyProtocol.initializeUserState(userName=self.userName,  initialTime=None, initialParam=None, initialPredicitons=None)
        print('Finished initialize UserState')
        iterationCounter = 0
        if self.therapyMethod == 'hmmTherapyProtocol':
            self.therapyProtocol.trainHMM()
            while not self.therapyProtocol.finishedTherapy:
                currentParam, outputMap = self.therapyProtocol.updateTherapyState()
                print('currentParam:', currentParam)
                self.therapyProtocol.getNextState(currentParam, self.therapyMethod)
                print('self.therpyprotocol.paramStatePath:', self.therapyProtocol.paramStatePath)
                combinedStates = [[param_state, user_compiled_mental] for param_state, user_compiled_mental in zip(self.therapyProtocol.unNormalizedParameter, self.therapyProtocol.userMentalStateCompiledLoss)]
                print('combinedStates:', combinedStates)
                if self.plotResults:
                    self.therapyProtocol.plottingProtocolsMain.plotTherapyResults_hmm(self.therapyProtocol.hmmModel, combinedStates, outputMap)
                self.therapyProtocol.checkConvergence_hmm(maxIterations)
                iterationCounter += 1
                print('iterationCounter:', iterationCounter)
        # Until the therapy converges.
        while not self.therapyProtocol.finishedTherapy:
            if self.therapyMethod == "aStarTherapyProtocol":
                # Get the next states for the therapy.
                therapyState, allMaps = self.therapyProtocol.updateTherapyState() # therapy state newuserParam

                # normalize the therapy state
                # therapyState = self.therapyProtocol.normalizeParameter(therapyState) # normalizeTherapyState = torch.Size([1, 1, 1, 1])

                self.therapyProtocol.getNextState(therapyState, self.therapyMethod)

                # Preparation for plotting
                combinedStates = [[param_state, user_compiled_mental] for param_state, user_compiled_mental in zip(self.therapyProtocol.unNormalizedParameter, self.therapyProtocol.userMentalStateCompiledLoss)]
                print('combinedStates', combinedStates)

                if self.plotResults:
                    self.therapyProtocol.plottingProtocolsMain.plotTherapyResults(combinedStates, allMaps)
                    print(f"Alpha after iteration: {self.therapyProtocol.percentHeuristic}\n")
                elif self.therapyMethod == "basicTherapyProtocol":
                    self.therapyProtocol.plotTherapyResults_basic(allMaps)  # For basic protocol, allMaps is the simulated map (only 1)
            elif self.therapyMethod == 'basicTherapyProtocol':
                therapyState, basicMap = self.therapyProtocol.updateTherapyState()
                self.therapyProtocol.getNextState(therapyState, self.therapyMethod)
                # Preparation for plotting
                combinedStates = [[param_state, user_compiled_mental] for param_state, user_compiled_mental in zip(self.therapyProtocol.unNormalizedParameter, self.therapyProtocol.userMentalStateCompiledLoss)]
                if self.plotResults:
                    self.therapyProtocol.plottingProtocolsMain.plotTherapyResults_basic(combinedStates, basicMap)

            # Check if the therapy has converged.
            self.therapyProtocol.checkConvergence(maxIterations)
            iterationCounter += 1
            print('iterationCounter:', iterationCounter)

    def therapyModelOutputUpdate(self):
        pass


if __name__ == "__main__":
    therapySelection = "Heat"
    # User parameters.
    userTherapyMethod = "aStarTherapyProtocol"  # The therapy algorithm to run. Options: "aStarTherapyProtocol", "basicTherapyProtocol", "nnTherapyProtocol", "hmmTherapyProtocol"
    testingUserName = "Squirtle"  # The username for the therapy.
    temperatureBounds = (30, 50)  # The temperature bounds for the therapy.
    temperatureBinWidth = 1.5  # The temperature bounds for the therapy.
    plotTherapyResults = True  # Whether to plot the results.

    # Simulation parameters.
    currentSimulationParameters = {
        'heuristicMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'simulatedMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'numSimulationHeuristicSamples': 50,  # The number of simulation samples to generate.
        'numSimulationTrueSamples': 30,  # The number of simulation samples to generate.
        'simulateTherapy': True,  # Whether to simulate the therapy.
    }

    # Initialize the therapy protocol
    therapyProtocol = heatTherapyControl(testingUserName, temperatureBounds, temperatureBinWidth, currentSimulationParameters, therapySelection=therapySelection, therapyMethod=userTherapyMethod, plotResults=plotTherapyResults)
    # Run the therapy protocol.
    therapyProtocol.runTherapyProtocol(maxIterations=None)
