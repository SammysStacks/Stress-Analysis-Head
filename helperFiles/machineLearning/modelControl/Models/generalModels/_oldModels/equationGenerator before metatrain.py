
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic modules
import os
import sys
import random
import joblib
import warnings
import numpy as np
from copy import deepcopy

from sklearn.model_selection import train_test_split

# Import files
sys.path.append(os.path.dirname(__file__) + "/../")
import _globalModel # Global model class

# Import metrics folder
sys.path.append(os.path.dirname(__file__) + "/../Model Helpers/Metrics/")
import _evaluationMetrics

# Import expression tree folder
sys.path.append(os.path.dirname(__file__) + "/../Model Helpers/Expression Tree/")
import expressionTreeModel

# -------------------------------------------------------------------------- #
# -------------------------- Recommendation Model -------------------------- #

class equationGenerator(nn.Module):        
    
    def __init__(self, modelPath, modelType, allFeatureNames, overwriteModel): 
        # Specify state boundaries.
        self.minValueConsidered, self.maxValueConsidered = -2, 2
        self.minStateValue, self.maxStateValue = -1, 1
        self.missingStateValue = np.nan
        self.missingStateIndex = -1
        self.rootNodeState = None
        
        # Specify initial parameters.
        numInternalStates = 3  # The number of internal states between [-minState, maxState]. Does not include the initial, bad overflow, and final state.
        evaluationWeights = [1, 0.75, 0.5, 0.25, 1.2] # A list of metric weights and the exploration weight.
        evaluationWeights = [1, 0.75, 0.5, 0.25, 1.2] # A list of metric weights and the exploration weight.
        metricTypes = ["cosSimilarity", "R2", "pearson", "spearman"] # A list of all metrics.
        # Parameters that should be optimized during training.
        self.setOptimizedParameters(metricTypes, evaluationWeights, numInternalStates)
                
        # Initialize classes
        self.metricsClass = _evaluationMetrics.evaluationMetrics() # Define evaluation metrics to score the states.
        self.expressionTreeModel = expressionTreeModel.expressionTreeModel() # Define the tree structure interface.
        # Initialize default states for evaluation.
        self.metricsClass._setMissingStateValue(self.missingStateValue)
        self.metricsClass._setBoundaryValues(self.minValueConsidered, self.maxValueConsidered)
        
        # Initialize common model class.
        super().__init__(modelPath, modelType, allFeatureNames, overwriteModel)
        
    def setOptimizedParameters(self, metricTypes, evaluationWeights, numInternalStates):
        # Setup variables as numpy arrays.
        self.metricTypes = np.asarray(metricTypes)
        self.evaluationWeights = np.asarray(evaluationWeights)
        
        self.evalLearningRate = 0.0
        
        # Segment off the metric weights.
        self.metricWeights = self.evaluationWeights[0:-1]
        assert len(self.metricTypes) == len(self.metricWeights)
        # Specify state information.
        self.numMetrics = len(self.metricTypes)        
        self.metricIndices = np.arange(0, self.numMetrics, 1)
        assert len(self.metricIndices) == self.numMetrics
        
        # Normalize the weights.
        self.metricWeights /= self.metricWeights.sum()
        self.evaluationWeights /= self.evaluationWeights.sum()
        
        # Setup state information
        self.numInternalStates = numInternalStates        
        # Setup root node statet information
        self.rootNodeStateIndices = np.full(self.numMetrics, self.rootNodeState)
        self.rootNodeStateValues = np.full(self.numMetrics, self.rootNodeState) 

    def _loadModel(self):
        with open(self.modelPath, 'rb') as handle:
            self.model = joblib.load(handle, mmap_mode ='r')
        
    def _saveModel(self):
        # Save the model
        joblib.dump(self, self.modelPath)
        
    # ---------------------------------------------------------------------- #
    # -------------------------- Initialize Model -------------------------- #
        
    def createModel(self):          
        # Reset variable model parameters.
        self._resetModel()
                
    def _resetModel(self):  
        # Reset training parameters.
        self.numExpandedFeatures = None
        
        # Initialize state information.
        self.segmentStateSpace()
        
    def changeTrainingData(self, Training_Data, Testing_Data, Training_Labels):
        # Expand the input data to include common transformations.
        Testing_Data = self.expandInputs(Testing_Data, [])
        Training_Data = self.expandInputs(Training_Data, [])
        
        # Initialize state-action pair information.
        self.setupStateActionInfo()
        
        # Initialize base state expression tree.
        self.initializeBaseState(Training_Data, Training_Labels.flatten(), baseStringEquation = "")
        
        return Training_Data, Testing_Data
                        
    def expandInputs(self, newInputs, collectedTransformations = []):
        inputShape = newInputs.shape
        # Initialize holder for the transformations.
        if len(collectedTransformations) == 0:
            collectedTransformations = np.zeros((inputShape[0], inputShape[1]*len(self.expressionTreeModel.functions_onceApplied)))
        
        # For each potental transformation.
        for transformerInd in range(len(self.expressionTreeModel.functions_onceApplied)):
            transformer = self.expressionTreeModel.functions_onceApplied[transformerInd]
            startColumnInd = inputShape[1]*transformerInd
            
            # While ignoring warnings.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Apply and store the transformed inputs.
                collectedTransformations[:, startColumnInd:startColumnInd + inputShape[1]] = transformer(newInputs)
        
        # Set the number of transformed features.
        if self.numExpandedFeatures != None:
            assert self.numExpandedFeatures == collectedTransformations.shape[1], "The number of features are inconsistent?"
            assert self.numExpandedFeatures > 0, "We need at least 1 feature."
        self.numExpandedFeatures = collectedTransformations.shape[1]
        
        return collectedTransformations
        
    # ---------------------------------------------------------------------- #
    # ----------------------- State-Action Interface ----------------------- # 
    
    def updateEvaluationWeights(self, nextStateValues):
        # Remove bad values.
        metricScores = nextStateValues.copy()
        metricScores[self.getStateIndices(metricScores) == self.missingStateIndex] = self.minValueConsidered
            
        # Calculate the regret of each metric
        metricRegrets = np.maximum(self.maxStateValue - metricScores, 0)
        metricRegrets = np.insert(metricRegrets, len(metricRegrets), np.min(metricRegrets)) 
        # Normalize the regret.
        metricRegrets /= metricRegrets.sum()
                
        # Update each evaluation weight.
        for evaluationInd in range(len(self.evaluationWeights)):
            self.evaluationWeights[evaluationInd] += self.evalLearningRate * metricRegrets[evaluationInd]
        # Update the metric weights as well
        self.metricWeights = self.evaluationWeights[0:-1]
        
        # Normalize the final weights.
        self.metricWeights /= self.metricWeights.sum()
        self.evaluationWeights /= self.evaluationWeights.sum()
    
    def segmentStateSpace(self):
        # Assert the validity of segmented the state space.
        assert 1 <= self.numInternalStates, "Must have at least one slice of the state space."
        assert isinstance(self.numInternalStates, int), "Please input an intger number of state spaces."
        
        # Segment the state space.
        self.stateSpaces = np.linspace(self.minStateValue, self.maxStateValue, num=self.numInternalStates + 1, endpoint=True)
        # Add boundary states: root node and end state.
        self.stateSpaces = np.insert(self.stateSpaces, 0, -np.inf) # Allow for overflow bad states.
        self.stateSpaces = np.insert(self.stateSpaces, len(self.stateSpaces), np.inf) # Add a boundary for the end state.

        # Get the state information.
        self.numStates = len(self.stateSpaces) - 1 # We add one state for the root node. However, the last state is there for boundary purposes.
        assert self.numStates == self.numInternalStates + 2, "Inequality: {} != {} + 2".format(self.numStates, self.numInternalStates)
        
    def initializeBaseState(self, transformedInputs, trueVals, baseStringEquation):
        # Initialize starting tree.
        self.baseExpressionTree = self.expressionTreeModel.equationInterface(baseStringEquation, [])
        
        # Initialize the starting state.
        self.baseStateValues = self.getStateValues(self.baseExpressionTree, transformedInputs, trueVals)
        self.baseStateIndices = self.getStateIndices(self.baseStateValues)
        
    def setupStateActionInfo(self):
        # Record how many actions are possible.
        self.numRootActions = self.numExpandedFeatures
        self.numActions = self.expressionTreeModel.numOperators*self.numExpandedFeatures + self.expressionTreeModel.numFunctions
        
        numActionBuffer = 10E-7
        # Initialize state action-pair information.
        self.stateActionRewards = np.zeros((self.numMetrics, self.numStates, self.numActions))
        self.numActionsTaken_perState = np.full((self.numMetrics, self.numStates, self.numActions), numActionBuffer)
        # Initialize Thompson sampling parameters.
        self.alpha = np.zeros((self.numMetrics, self.numStates, self.numActions))
        self.beta = np.zeros((self.numMetrics, self.numStates, self.numActions))
        
        # Initialize root state action-pair information.
        self.stateActionRewards_Root = np.zeros((self.numMetrics, self.numRootActions))
        self.numActionsTaken_Root = np.full((self.numMetrics, self.numRootActions), numActionBuffer)
        # Initialize Thompson sampling parameters.
        self.alpha_Root = np.zeros((self.numMetrics, self.numRootActions))
        self.beta_Root = np.zeros((self.numMetrics, self.numRootActions))
                        
    def getBaseStates(self):
        return deepcopy(self.baseExpressionTree), self.baseStateValues, self.baseStateIndices
            
    def getStateIndices(self, stateValues):
        # Base case: we are at the root node.
        if self.isRootState(stateValues):
            return self.rootNodeStateIndices
        
        # The stateSpaces is 1-indexed, as we are ignoring the root node state.
        stateIndices = np.searchsorted(self.stateSpaces, stateValues, side='right') - 1
        stateIndices[np.isnan(stateValues)] = self.missingStateIndex # In searchsorted, np.nan index is len(stateValues)
        
        return stateIndices
            
    def isRootState(self, stateInfo):
        assert self.rootNodeState == None
        return (self.rootNodeState == stateInfo).all()
        
    def isUnknownStateValue(self, stateInfo):
        # TODO: np.isnan fails with None values.
        assert np.isnan(self.missingStateValue)
        return np.isnan(stateInfo).all()
    
    def isUnknownStateIndex(self, stateInfo):
        return (self.missingStateIndex == stateInfo).all()
            
    def getStateBoundaries(self, stateIndex):
        # Base case: we are at the root node.
        if self.isRootState(stateIndex):
            return self.rootNodeState, self.rootNodeState
        # Base case: we cant calculate the state.
        elif self.isUnknownStateIndex(stateIndex):
            return self.missingStateValue, self.missingStateValue
        
        # Returns two states. Only the left state is inclusive.
        return self.stateSpaces[stateIndex], self.stateSpaces[stateIndex + 1]
    
    def getStateValues(self, expressionTree, transformedInputs, trueVals, metricTypes = None):
        # Base case: I ONLY have a root node, no equation to compare.
        if self.expressionTreeModel.isStubTree(expressionTree):
            return self.rootNodeStateValues
        
        # Find the predicted equation's points.
        predictedVals = self.predict(expressionTree, transformedInputs)
        assert len(predictedVals) == len(trueVals)
        
        # Score the prediction with each metric provided.
        metricTypes = self.metricTypes if metricTypes == None else metricTypes
        finalStateValues = self.metricsClass.evaluateMetrics(metricTypes, trueVals, predictedVals, normalizeData = False, numParams = 0) 
        
        # print("finalStateValues:", finalStateValues)
        return finalStateValues
    
    def findOptimalAction(self, stateIndices):         
        isRootState = self.isRootState(stateIndices)
        assert not self.isUnknownStateIndex(stateIndices)
        # Specify the weights of each metric
        metricWeights = self.metricWeights.copy()
        evaluationWeights = self.evaluationWeights.copy()
                        
        # Calculate the expected reward for the action.
        if isRootState:
            # Using root node values.
            numStateActionPairs = self.numActionsTaken_Root
            expectedActionRewards = self.stateActionRewards_Root / numStateActionPairs
        else:
            # Create a mask of good states.
            goodStateMask = stateIndices != self.missingStateIndex
            metricIndices = self.metricIndices[goodStateMask]
            goodStateIndices = stateIndices[goodStateMask]
            # Specify the weights of each metric
            metricWeights = metricWeights[goodStateMask]
            evaluationWeights[0:-1] = evaluationWeights[0:-1][goodStateMask]
            
            # Using the spectrum of states.
            numStateActionPairs = self.numActionsTaken_perState[metricIndices, goodStateIndices, :]
            expectedActionRewards = self.stateActionRewards[metricIndices, goodStateIndices, :] / numStateActionPairs
            
        print("")
        print(numStateActionPairs.shape)
        print(numStateActionPairs)
        print(expectedActionRewards.shape)
        print(expectedActionRewards)
        if not isRootState:
            sys.exit()
        
        
        # Perform a weighted average across each metric.
        averageNum_stateActionPairs = (numStateActionPairs.T * metricWeights / metricWeights.sum()).T.sum(axis = 0)
        finalStateActionRewards = (expectedActionRewards.T * evaluationWeights[0:-1] / evaluationWeights.sum()).T.sum(axis = 0)
        
        # Base case: we have no information about this state.
        if averageNum_stateActionPairs.sum() < 1:
            return self.findRandomAction(isRootState)
        
        # TODO: better exploration. Thompson?
        # Evaluate the rewards for exploring underpaved actions.
        exploringActionRewards = 1 / averageNum_stateActionPairs
        exploringActionRewards[self.maxValueConsidered < exploringActionRewards] = self.maxValueConsidered
        # Compile the final reward given all the state's metrics.
        finalStateActionRewards += exploringActionRewards * self.evaluationWeights[-1]
                        
        # Return the best action considering both reward types.
        return np.argmax(finalStateActionRewards)
    
    # ---------------------------------------------------------------------- #
    # ------------- Update Weights After Exploring an Equation ------------- # 
    
    def recordActionState(self, stateIndices, actionIndex, finalRewards):
        assert len(stateIndices) == self.numMetrics
        print("Updating States", stateIndices, actionIndex, finalRewards)
        
        updatingRootState = self.isRootState(stateIndices)
        # For each evaluation metric of the equation.
        for metricInd in range(self.numMetrics): 
            stateIndex = stateIndices[metricInd]
            finalReward = finalRewards[metricInd]
            # TODO: we are still favoring for us to go into missing states.

            if updatingRootState:
                # Record taking the action in this state.
                self.numActionsTaken_Root[metricInd][actionIndex] += 1
                self.stateActionRewards_Root[metricInd][actionIndex] += finalReward
            else:
                # We cannot update if there is no state for this metric.
                if self.isUnknownStateIndex(stateIndex): continue
                # Record taking the action in this state.
                self.numActionsTaken_perState[metricInd][stateIndex][actionIndex] += 1
                self.stateActionRewards[metricInd][stateIndex][actionIndex] += finalReward
        
    def backPropogation(self, stateItinerary, actionItinerary, stateRewards):
        # Assert back propogation is 1:1 state-action.
        assert len(stateItinerary) == len(actionItinerary)
        
        # Perform back propogation.
        for iterationInd in range(len(actionItinerary)):
            # Get the iteration's action-state pair
            stateIndices = stateItinerary[iterationInd] # Each initial state for each metric.
            actionIndex = actionItinerary[iterationInd] # The action taken in these states.
            
            # Inform each state of the final reward for the action-state pair.
            self.recordActionState(stateIndices, actionIndex, stateRewards)
    
    # ---------------------------------------------------------------------- #
    # ------------------ Applying Action to the Expression ----------------- #  
    
    def findRandomAction(self, isRootNode = False):
        if isRootNode:
            return random.randint(0, self.numRootActions - 1)
        return random.randint(0, self.numActions - 1)
    
    def decodeActionIndex(self, actionIndex):
        print("actionIndex", actionIndex)
        # Get the transformer index associated with the action.
        transformerIndex = actionIndex // self.numExpandedFeatures  # The index of the transformers: [operators, functions]
        # If we have a function, adjust the index to account for no variables.
        if self.expressionTreeModel.isFunctionIndex(transformerIndex):
            transformerIndex = actionIndex - self.expressionTreeModel.numOperators*(self.numExpandedFeatures-1)
            return transformerIndex, None
        
        # Get the feature associated with the action.
        expandedFeatureInd = actionIndex % self.numExpandedFeatures   # The index of expanded feature index.
        featureInd = expandedFeatureInd % len(self.finalFeatureNames) # The true feature index.
        # expandedTransformerIndex = expandedFeatureInd // len(self.finalFeatureNames) # The index of the function once applied.
        
        return transformerIndex, featureInd
    
    def createActionNodes(self, transformerIndex, featureInd = None):
        # Get the transformer
        transformer = self.expressionTreeModel.getTransformer(transformerIndex)
        # Get the transformer information.
        transformerSymbol = self.expressionTreeModel.mapTransformer_toString[transformer]
        transformerSymbol = transformerSymbol.replace(" ", "").replace("(", "")
        maxChildren = self.expressionTreeModel.findMaxChildren_Transformer(transformerIndex)
        
        # Create a tree node for the transformer.
        transformerNode = expressionTreeModel.treeNodeClass(transformer, stringValue = transformerSymbol, 
                                                            parentNode = None, maxChildren = maxChildren)
        # If we have a function.
        if featureInd == None:
            # We dont have an associate variable.
            print(transformerSymbol)
            return None, transformerNode
        
        # Get the variable information.
        variableName = self.finalFeatureNames[featureInd]
        # Create a tree node for the variable.
        variableNode = expressionTreeModel.treeNodeClass(featureInd, stringValue = variableName, 
                                                         parentNode = None, maxChildren = 0)
        
        print(transformerSymbol, variableName)
        return variableNode, transformerNode
        
    def isBetterModel(self, previousStates, newStates):
        # Base case: we are still at a root state.
        if self.isRootState(newStates): return False
        # Base case: we cannot score this new state,
        if self.isUnknownStateValue(newStates): return False
        # Base case: we moved from root to good state.
        if self.isRootState(previousStates): return True
        
        # Score the previous state.
        goodStateMask = ~np.isnan(newStates)
        newScore = np.average(newStates[goodStateMask], weights = self.metricWeights[goodStateMask])
        # Score the new state.
        goodStateMask = ~np.isnan(previousStates)
        previousScore = np.average(previousStates[goodStateMask], weights = self.metricWeights[goodStateMask])
        
        # Return if the new state is better
        return previousScore < newScore
    
    def applyOptimalAction(self, initialExpressionTree, stateValues, actionIndex, transformedInputs, trueVals):
        # Organize input information.
        expressionTree = deepcopy(initialExpressionTree)
        
        # Prepare the nodes associated with the action.
        transformerIndex, featureInd = self.decodeActionIndex(actionIndex)
        variableNode, transformerNode = self.createActionNodes(transformerIndex, featureInd)
        
        # Base case: I ONYL have a root node.
        if self.expressionTreeModel.isStubTree(expressionTree):
            # I can only place a single feature in the tree.
            self.expressionTreeModel.insertChildNode(expressionTree, variableNode)
            stateValues = self.getStateValues(expressionTree, transformedInputs, trueVals)

            return expressionTree, stateValues
        
        # Setup breadth first search.
        queuedNodes = expressionTree.children.copy()
        # Initialize the best modified expressionTree.
        bestExpressionTree = None
        bestStateValues = self.rootNodeStateValues.copy()
                                
        # While there are unexplored nodes.
        while len(queuedNodes) != 0:
            # Explore a new child node.
            actionNode = queuedNodes.pop(0)
            
            # Check if its worth evaluating this transformer/variable.         
            if self.usefulEvaluation(actionNode, variableNode, transformerNode):
                # print("Before Mod")
                # expressionTree.prettyPrint()
                # Apply the action to this node.
                self.expressionTreeModel.insertParentNode(actionNode, transformerNode)
                # expressionTree.prettyPrint()
                if variableNode != None:
                    self.expressionTreeModel.insertChildNode(transformerNode, variableNode)
                # expressionTree.prettyPrint()
                

                # Evaluate the benefit of this action.
                stateValues = self.getStateValues(expressionTree, transformedInputs, trueVals)
                 
                # print("After Mod")
                # print(stateValues)
                # expressionTree.prettyPrint()
                
                if self.isBetterModel(bestStateValues, stateValues):
                    bestStateValues = stateValues.copy()
                    bestExpressionTree = deepcopy(expressionTree)
                # Remove the action from this node.
                if variableNode != None:
                    self.expressionTreeModel.removeNode(variableNode)
                self.expressionTreeModel.removeNode(transformerNode)
                # print("After Remove")
                # expressionTree.prettyPrint()
                # print("\tPossible State Value", stateValues)
            
            # Store the actionNode's children.
            queuedNodes.extend(actionNode.children.copy())
        
        return bestExpressionTree, bestStateValues
    
    def usefulEvaluation(self, actionNode, variableNode, transformerNode):
        # Get the parent node.
        parentNode = actionNode.parentNode
        # Get the values of the nodes.
        actionValue = actionNode.numericValue
        parentValue = parentNode.numericValue
        transformerValue = transformerNode.numericValue
            
        # If we have a function transformer.
        if variableNode == None:
            # Always consider this.
            return True

        # If we have a multiplication or addition.
        if transformerValue == parentValue and transformerValue in [self.expressionTreeModel.add, self.expressionTreeModel.multiply]:
            # We should only consider this action on the first child.
            return self.expressionTreeModel.isFirstChild(actionNode)
            
        return True
        
    # ---------------------------------------------------------------------- #
    # ----------------- Finding an Equation from Base State ---------------- #  

    def adjustStateValues(self, stateValues, scaleValues):
        return np.where(stateValues > 0, stateValues * scaleValues, stateValues / scaleValues)
        
    def expandExpressionTree(self, transformedInputs, trueVals):
        # Initialize a new base expression tree.
        expressionTree, stateValues, stateIndices = self.getBaseStates()
        # Initialize game state parameters.
        actionItinerary = []  # A list of actions in order.
        stateItinerary = []   # A list of states visited in order.
                
        while True: # While there is a good move to make. 
            # Modify the equation with the optimal action.
            actionIndex = self.findOptimalAction(stateIndices)
            modifiedExpressionTree, nextStateValues = self.applyOptimalAction(expressionTree, stateValues, actionIndex, transformedInputs, trueVals)
            
            # Adjust the reward value to prevent complex equations.
            numFeatures, numTransformers = self.expressionTreeModel.getDegreesOfFreedom(modifiedExpressionTree)
            scaleReward = (len(transformedInputs) - 1 - numFeatures - numTransformers) / (len(transformedInputs) - 1)
            assert (len(transformedInputs) - 1 - numFeatures - numTransformers) > 0
            # Scale the state rewards.
            nextStateValues = self.adjustStateValues(nextStateValues, scaleReward)
                
            # Base case: we never explored a state.
            if self.expressionTreeModel.isStubTree(modifiedExpressionTree):
                # Prevent this action from occuring and re-explore.
                self.recordActionState(stateIndices, actionIndex, -np.inf)
                return self.expandExpressionTree(transformedInputs, trueVals)
                
            # Base case: there is nothing good to explore (termination state).
            elif not self.isBetterModel(stateValues, nextStateValues) or numFeatures + numTransformers > 50:
                # Record a bad move for taking this action in this state.
                print("Bad Step:", stateIndices, actionIndex, nextStateValues)
                self.recordActionState(stateIndices, actionIndex, nextStateValues)
                # Perform backpropogation of the final reward.
                self.backPropogation(stateItinerary, actionItinerary, stateValues)
                break
        
            assert modifiedExpressionTree != None
            
            self.updateEvaluationWeights(nextStateValues)
            # Record the modification.
            actionItinerary.append(actionIndex)
            stateItinerary.append(stateIndices)
            # Reset the current state
            stateValues = nextStateValues.copy()
            stateIndices = self.getStateIndices(nextStateValues)
            # Normalize the equation
            # self.expressionTreeModel.normalizeEquation(modifiedExpressionTree, self.predict(modifiedExpressionTree, transformedInputs))
            # Prepare for the next round.
            expressionTree = self.expressionTreeModel.simplifyExpressionTree_org(modifiedExpressionTree)

        return expressionTree
    
    # ---------------------------------------------------------------------- #
    # ----------------------------- Apply Model ---------------------------- #
    
    def scoreModel(self, expressionTree, Testing_Data, Testing_Labels):
        if len(self.finalFeatureNames) == 0:
            print("Model Not Trained and cannot provide a score")
            return None
        
        return self.getStateValues(expressionTree, Testing_Data, Testing_Labels, metricTypes = ["R2"])

    def predictPoint(self, Ui, timePoint, userInd, itemInd):
        pass    
    
    def predict(self, expressionTree, transformedInputs):
        return self.expressionTreeModel.expressionTreeInterface(expressionTree, transformedInputs)
    
    def baseCasePredictions(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels):
        # Base case: the training labels are constant.
        if (Training_Labels == Training_Labels[0]).all():
            print("There is no variation in the training labels. Predicting a constant.")
            exit()        
        # Base case: the test labels are constant, but not the training labels.
        assert not (Testing_Labels == Testing_Labels[0]).all(), \
                "Your testing labels are constant, which does not track with your training labels."
        
        # Assert all base cases covered.
        assert np.linalg.norm(Testing_Labels) != 0
        assert np.linalg.norm(Training_Labels) != 0

    # ---------------------------------------------------------------------- #
    # ---------------------------- Train Model ----------------------------- #
    
    def metaTrain(batchData, trueBatchLabels):
        batchScores = []
        return batchScores
    
    def trainModel(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels, featureNames, maxEpochs = 300):          
        # Enforce variable types, save feature information, and reset the model.
        Training_Data, Training_Labels, Testing_Data, Testing_Labels = \
                    self.setupTraining(Training_Data, Training_Labels, Testing_Data, Testing_Labels, featureNames)
        # Alter the training data.
        Training_Data, Testing_Data = self.changeTrainingData(Training_Data, Testing_Data, Training_Labels)
        # Base case predictions.
        self.baseCasePredictions(Training_Data, Training_Labels, Testing_Data, Testing_Labels)

        # Keep track of error
        testingError = []
        trainingError = []
                
        for i in range(10000):
            print("\n\nEnter")
            Validation_Data, _, Validation_Labels, _ = train_test_split(Training_Data, Training_Labels, test_size=0.2, shuffle= True)
            finalExpressionTree = self.expandExpressionTree(Validation_Data, Validation_Labels.flatten())
            
            print("\nFinal:")
            finalExpressionTree.prettyPrint()
            # Keep track of error
            assert not self.expressionTreeModel.isStubTree(finalExpressionTree)
            if not self.expressionTreeModel.isStubTree(finalExpressionTree):
                trainingError.append(self.scoreModel(finalExpressionTree, Training_Data, Training_Labels))
                testingError.append(self.scoreModel(finalExpressionTree, Testing_Data, Testing_Labels))
            print(trainingError[-1], testingError[-1])
            
        
        minVal = min(min(testingError), min(trainingError))            
        import matplotlib.pyplot as plt
        plt.plot(trainingError, '-k', linewidth=2, label="Training Error")
        plt.plot(testingError, '-', c='tab:red', linewidth=2, label="Testing Error")
        plt.ylim(max(-5, minVal*0.9 if abs(minVal) == minVal else minVal*1.1), 1.05)
        plt.show()

        return None
                
        # Keep track of error
        testingError = []
        trainingError = []
        # For each training epoch.
        for epoch in range(maxEpochs):
            initialLoss = self.scoreModel(Training_Data, Training_Labels)
            
            finalEquations = []
            finalR2s = []
            
            numSplits = 10
            # For each split of the data.
            for splitInd in range(len(numSplits)):
                # Split the data into cross validation splits.
                Validation_Data, _, Validation_Labels, _ = train_test_split(self.inputData, Training_Labels, test_size=0.2, shuffle= True)
                
                # Calculate the R2 of the current equation
                predictedVal = self.expressionTreeModel(self.equation, Validation_Data, solveEquation = True)
                currentR2 = self.calculateR2(Validation_Labels, predictedVal, calculateAdjustedR2=True, numFeatures=0)
                # Find the best modification of this split.
                modifiedEquation, modifiedR2 = self.findBestModification(Validation_Data, Validation_Labels, self.equation, self.equation, currentR2)
                
                # Save the modification information
                finalEquations.append(modifiedEquation)
                finalR2s.append(modifiedR2)

            # Choose the most frequent and best equation modification.
            self.updateEquation(finalEquations, finalR2s)
            
            # Keep track of error
            trainingError.append(self.scoreModel(Training_Data, Training_Labels))
            testingError.append(self.scoreModel(Testing_Data, Testing_Labels))
        
            # Stopping condition.
            finalLoss = trainingError[-1]
            if initialLoss != 0 and abs((finalLoss - initialLoss)/initialLoss) < self.lossThreshold: # and 0.8 < finalLoss:
                print("\tStopped training early at epoch:", epoch)
                break
            
        import matplotlib.pyplot as plt
        plt.plot(trainingError, '-k', linewidth=2, label="Training Error")
        plt.plot(testingError, '-', c='tab:red', linewidth=2, label="Testing Error")
        plt.show()
        
        # Return the final score
        return testingError[-1]

    
# ---------------------------------------------------------------------------#

if __name__ == "__main__":
    # Set model paramters
    modelPath = "./EGM.pkl"
    modelType = "EGM"
    allFeatureNames = [""]
    overwriteModel = True
    
    numPoints = 1000
    # Specify input features.
    x = np.random.uniform(1, 10, numPoints)
    y = np.random.uniform(0, 10, numPoints)/10
    z = np.random.uniform(3, 6, numPoints)
    a = np.random.uniform(4, 7, numPoints)
    b = np.random.uniform(5, 8, numPoints)
    c = np.random.uniform(6, 4, numPoints)
    m = np.random.uniform(1, 9, numPoints)
    # True labels
    featureLabels_EGM = m*x*y + np.sin(x)*x
    featureLabels_EGM = m + y + z**y
    
    # Compile feature data.
    featureData_EGM = np.asarray([x, y, z, m, a, b, c]).T
    featureNames_EGM = np.asarray(['x', 'y', 'z', 'm', 'a', 'b', 'c'])
    maxEpochs = 5
    
    
    # featureData_EGM = np.asarray([c]).T
    # featureNames_EGM = np.asarray(['c'])
    # featureLabels_EGM = x + c


    # Instantiate class.
    modelClass = equationGenerator(modelPath, modelType, allFeatureNames, overwriteModel)
    # Train model
    Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(featureData_EGM, featureLabels_EGM, test_size=0.2, shuffle= True)
    modelClass.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels, featureNames_EGM, maxEpochs)
    
    
    f = m*c + x**y
    h = x
    
    modelClass.metricsClass.evaluateMetrics("cosSimilarity", f, h, normalizeData = False, numParams = 0) 
        
    # theta = np.linalg.inv((X.T @ X)) @ X.T @ Y
    # modelClass.calculateR2(trueOutput, (X@theta).T[0], calculateAdjustedR2=True, numFeatures=0)
    
    def findActionStateIndex(actionIndex):
        # Decode the action index.
        transformerIndex, featureInd = modelClass.decodeActionIndex(actionIndex)

        # Get the transformer associated with the action.
        transformer = modelClass.expressionTreeModel.getTransformer(transformerIndex)
        # Create a tree node for the transformer.
        transformerSymbol = modelClass.expressionTreeModel.mapTransformer_toString[transformer]
        transformerSymbol = transformerSymbol.replace(" ", "").replace("(", "")

        if featureInd is not None:
            # Create a tree node for the variable.
            variableName = modelClass.finalFeatureNames[featureInd]
            return transformerSymbol + " " + variableName
        else:
            return transformerSymbol

    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Get the variables from the model class.
    stateSpaces = modelClass.stateSpaces.copy()
    numMetrics = modelClass.numMetrics
    metricTypes = modelClass.metricTypes
    # Get the index with the highest probability for each state
    stateActionProbs = modelClass.stateActionRewards.copy() / modelClass.numActionsTaken_perState.copy()  # Array of probabilities for each state
    maxstateActionProbs_Indexes = np.argmax(stateActionProbs, axis=2)
    
    # Specify Figure aesthetics
    numColumns = 2; figWidth = 10; figHeight = 8
    fig, axes = plt.subplots(numMetrics // numColumns, numColumns, sharey=True, sharex=True, figsize=(figWidth, figHeight))
    if numMetrics == 1: axes = [axes]
    
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    # Add figure labels
    plt.suptitle('Co-optimal Policy Decisions from Adversarial Agents', fontsize=15, fontweight="bold")
    plt.subplots_adjust(top=0.925)
    
    # Assign a color to each index.
    indexes = np.unique(maxstateActionProbs_Indexes)
    indexColors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
                   'tab:cyan', 'tab:pink', 'tab:olive', 'tab:orange',
                   'mediumblue', 'darkorange', 'limegreen', 'firebrick', 'darkviolet',
                   'sienna', 'hotpink', 'dimgray', 'olivedrab', 'deepskyblue',
                   'lightblue', 'lightgreen', 'salmon', 'lavender', 'saddlebrown',
                   'aqua', 'violet', 'gold', 'tomato',
                   'steelblue', 'darkgoldenrod']
    indexDict = dict(zip(indexes, indexColors))
    
    # Loop over each metric
    for metricInd in range(numMetrics):
        # Remove the indexes where there wasn't any information.
        zero_rows = np.where((modelClass.stateActionRewards[metricInd] == 0).all(axis=1))[0]
        maxstateActionProbs_Indexes[metricInd][zero_rows] = -1
    
        # Create a subplot for the current metric
        ax = axes[metricInd // numColumns][metricInd % numColumns]
        ax.set_aspect('equal')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlabel("Arbitrary axis")
        ax.set_ylabel("Arbitrary axis")
        ax.set_title(metricTypes[metricInd])
    
        # Plot each segment of the circle with a different color based on the index with the highest probability
        for i in range(len(stateSpaces) - 2):
            # Get the angle.
            initialCosTheta = stateSpaces[i + 1]
            finalCosTheta = stateSpaces[i + 2]
            theta = np.linspace(np.arccos(initialCosTheta), np.arccos(finalCosTheta), 100)
            # Get the xi,yi coordinates
            xi = np.cos(theta)
            yi = np.sin(theta)
            # Fill the circle
            maxIndex = maxstateActionProbs_Indexes[metricInd][i + 1]
            if maxIndex != -1:
                ax.fill([0] + xi.tolist(), [0] + yi.tolist(), indexDict[maxIndex], alpha=1)
                ax.fill([0] + xi.tolist(), [0] + (-yi).tolist(), indexDict[maxIndex], alpha=1)
            else:
                ax.fill([0] + xi.tolist(), [0] + yi.tolist(), 'k', alpha=0.1)
                ax.fill([0] + xi.tolist(), [0] + (-yi).tolist(), 'k', alpha=0.1)
    
        # Adjusting the layout to make space for the legend
        ax.figure.subplots_adjust(right=0.8)
    
    # Add all indexes to the legend.
    legendInds = indexes[indexes >= 0]
    legendLabels = [findActionStateIndex(index) for index in legendInds]
    # Create a single legend for all metrics
    legendColors = list(indexDict.values())
    legend_patches = [Patch(color=color, label=name) for color, name in zip(legendColors, legendLabels)]
    # Add the legend outside the figure on the right
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove the empty subplots
    for i in range(numMetrics, len(axes)):
        fig.delaxes(axes[i])
    
    # Save the figure with extended bounding box to accommodate the legend
    plt.savefig('output.png', bbox_inches='tight', dpi=300)
    
    
    
    exit()
    
    


    
    
    numPoints = 10000
    xi = np.arange(-10, 10, 1/numPoints)
    
    # plt.plot(xi, xi**4 + 2*xi**3, 'k', linewidth = 2, label = "$xi^4 + 2x^3$")
    # plt.plot(xi, xi**4, 'tab:red', linewidth = 2, label = "$xi^4$")
    plt.plot(xi, xi**2 * np.sin(xi) / 50, 'k', linewidth = 2, label = "$xi^2 * sin(xi) / 50$")
    plt.plot(xi, np.sin(xi), 'tab:red', linewidth = 2, label = "$sin(xi)$")

    plt.title("How to Compare Equation Similarities")
    plt.xlabel("Input Points")
    plt.ylabel("Output Points")
    
    plt.xlim(-3, 2)
    plt.ylim(-3, 6)
    plt.legend()

        

