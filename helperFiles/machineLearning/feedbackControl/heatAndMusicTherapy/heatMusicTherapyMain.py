# Import the necessary libraries.
from helperFiles.machineLearning.feedbackControl.heatAndMusicTherapy.therapyHelpers import therapyHelpers


class heatMusicTherapyControl(therapyHelpers):
    def __init__(self, userName, initialParameterBounds, unNormalizedParameterBinWidths, simulationParameters, therapySelection, therapyMethod, plotResults=False):
        super().__init__(userName=userName, initialParameterBounds=initialParameterBounds, unNormalizedParameterBinWidths=unNormalizedParameterBinWidths,
                         simulationParameters=simulationParameters, therapySelection=therapySelection, therapyMethod=therapyMethod, plotResults=plotResults)

    def runTherapyProtocol(self, maxIterations=None):
        # Initialize holder parameters such as the user maps.
        self.therapyProtocol.initializeUserState(userName=self.userName)
        print('Finished initialize UserState')

        iterationCounter = 0
        # Until the therapy converges.
        while not self.therapyProtocol.finishedTherapy:
            if self.therapyMethod == "aStarTherapyProtocol":
                if self.therapySelection == 'Heat':
                    # Get the next states for the therapy.
                    therapyState, allMaps = self.therapyProtocol.updateTherapyState() # Therapy state newuserParam
                    self.therapyProtocol.getNextState(therapyState, self.therapyMethod)
                    # Preparation for plotting
                    combinedStates = [[param_state, user_compiled_mental] for param_state, user_compiled_mental in zip(self.therapyProtocol.unNormalizedParameter, self.therapyProtocol.userMentalStateCompiledLoss)]
                    if self.plotResults:
                        self.therapyProtocol.plottingProtocolsMain.plotTherapyResults(combinedStates, allMaps)
                        print(f"Alpha after iteration: {self.therapyProtocol.percentHeuristic}\n")
                elif self.therapySelection == 'BinauralBeats':
                    therapyState, allMaps = self.therapyProtocol.updateTherapyState3D()

                    self.therapyProtocol.getNextState(therapyState, self.therapyMethod)
                    combinedStates = [
                        [
                            unNormalizedParam_1,  # First parameter
                            unNormalizedParam_2,  # Second parameter
                            user_compiled_mental  # Compiled loss
                        ]
                        for unNormalizedParam_1, unNormalizedParam_2, user_compiled_mental in zip(
                            self.therapyProtocol.unNormalizedParam_1, self.therapyProtocol.unNormalizedParam_2,
                            self.therapyProtocol.userMentalStateCompiledLoss
                        )
                    ]

                    if self.plotResults:
                        self.therapyProtocol.plottingProtocolsMain.plotTherapyResults3D(combinedStates, allMaps)
                        print(f"Alpha after iteration: {self.therapyProtocol.percentHeuristic}\n")

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


if __name__ == "__main__":
    # Hard-coded initialization
    parameterBounds = (0, 10)
    parameterBinWidth = 1

    # Therapy method initialziation
    therapyChoices = "BinauralBeats"  # 'Heat or 'BinauralBeats'

    # User parameters.
    userTherapyMethod = "aStarTherapyProtocol"
    testingUserName = "Subject XYZ"  # The username for the therapy.
    if therapyChoices == "BinauralBeats":
        baseFrequency = 424  # Select different base frequency to stimulate;
        parameterBounds = ((baseFrequency+8, baseFrequency+15), (baseFrequency+8, baseFrequency+15))  # The parameter bounds for the therapy.
        parameterBinWidth = 1  # The parameter bounds for the therapy.
    elif therapyChoices == "Heat":
        parameterBounds = (30, 50)  # The temperature bounds for the therapy.
        parameterBinWidth = 1.5  # The temperature bounds for the therapy.
    plotTherapyResults = True  # Whether to plot the results.

    # Simulation parameters.
    currentSimulationParameters = {
        'heuristicMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'simulatedMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'numSimulationHeuristicSamples': 50,  # The number of simulation samples to generate.
        'numSimulationTrueSamples': 30,  # The number of simulation samples to generate.
        'simulateTherapy': False,  # Whether to simulate the therapy.
    }

    # Add assertion to disallow certain combinations
    assert not (therapyChoices == "BinauralBeats" and userTherapyMethod == "basicTherapyProtocol"), \
        "BinauralBeats with basicTherapyProtocol is not efficient please choose aStarTherapyProtocol."

    # Initialize the therapy protocol
    therapyProtocol = heatMusicTherapyControl(testingUserName, parameterBounds, parameterBinWidth, currentSimulationParameters, therapySelection=therapyChoices, therapyMethod=userTherapyMethod, plotResults=plotTherapyResults)
    # Run the therapy protocol.
    therapyProtocol.runTherapyProtocol(maxIterations=100)
