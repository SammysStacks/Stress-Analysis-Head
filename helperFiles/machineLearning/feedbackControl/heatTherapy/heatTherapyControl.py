from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.aStarProtocol import aStarProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.basicProtocol import basicProtocol
from helperFiles.machineLearning.feedbackControl.heatTherapy.helperMethods.nnProtocol import nnProtocol
import time


class heatTherapyControl:
    def __init__(self, temperatureBounds, tempBinWidth, simulationParameters, therapyMethod="aStarProtocol", plotResults=False):
        # General parameters.
        self.simulationParameters = simulationParameters
        self.temperatureBounds = temperatureBounds
        self.tempBinWidth = tempBinWidth
        self.plotResults = plotResults

        # Therapy parameters.
        self.therapyProtocol = None
        self.therapyMethod = None

        # Set up the therapy protocols.
        self.setupTherapyProtocols(therapyMethod)

    def setupTherapyProtocols(self, therapyMethod):
        # Change the therapy method.
        self.therapyMethod = therapyMethod

        if self.therapyMethod == "aStarProtocol":
            self.therapyProtocol = aStarProtocol(self.temperatureBounds, self.tempBinWidth, self.simulationParameters, learningRate=2)
        elif self.therapyMethod == "basicProtocol":
            self.therapyProtocol = basicProtocol(self.temperatureBounds, self.tempBinWidth, self.simulationParameters)
        elif self.therapyMethod == "nnProtocol":
            self.therapyProtocol = nnProtocol(self.temperatureBounds, self.tempBinWidth, self.simulationParameters, modelName="2024-04-12 heatTherapyModel", onlineTraining=False)
        else:
            raise ValueError("Invalid therapy method provided.")

    def runTherapyProtocol(self, maxIterations=None):
        # Initialize holder parameters.
        self.therapyProtocol.initializeUserState()

        # Until the therapy converges.
        while not self.therapyProtocol.finishedTherapy:
            # Get the next states for the therapy.
            newUserTemp, allMaps = self.therapyProtocol.updateTherapyState()
            self.therapyProtocol.getNextState(newUserTemp)

            if self.plotResults:
                if self.therapyMethod == "aStarProtocol":
                    self.therapyProtocol.plotTherapyResults(allMaps)
                    print(f"Alpha after iteration: {self.therapyProtocol.percentHeuristic}\n")
                elif self.therapyMethod == "basicProtocol":
                    self.therapyProtocol.plotTherapyResults_basic(allMaps) # For basic protocol, allMaps is the simulated map (only 1)
                    time.sleep(0.1)
            # Check if the therapy has converged.
            self.therapyProtocol.checkConvergence(maxIterations)


if __name__ == "__main__":
    # User parameters.
    userTherapyMethod = "aStarProtocol"  # The therapy algorithm to run. Options: "aStarProtocol", "basicProtocol"
    userTemperatureBounds = (30, 50)  # The temperature bounds for the therapy.
    plotTherapyResults = True  # Whether to plot the results.
    userTempBinWidth = 2  # The temperature bin width for the therapy.

    # Simulation parameters.
    currentSimulationParameters = {
        'heuristicMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'simulatedMapType': 'uniformSampling',  # The method for generating the simulated map. Options: 'uniformSampling', 'linearSampling', 'parabolicSampling'
        'numSimulationHeuristicSamples': 10,  # The number of simulation samples to generate.
        'numSimulationTrueSamples': 50,  # The number of simulation samples to generate.
        'simulateTherapy': True,  # Whether to simulate the therapy.
    }

    # Initialize the therapy protocol
    therapyProtocol = heatTherapyControl(userTemperatureBounds, userTempBinWidth, currentSimulationParameters, therapyMethod=userTherapyMethod, plotResults=plotTherapyResults)

    # Run the therapy protocol.
    therapyProtocol.runTherapyProtocol(maxIterations=1000)
