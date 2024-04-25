
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic modules
import random
import joblib
import warnings
import numpy as np
from copy import deepcopy

from sklearn.model_selection import train_test_split

# Import files
import _globalModel # Global model class
import expressionTreeModel

# -------------------------------------------------------------------------- #
# -------------------------- Recommendation Model -------------------------- #

class equationGenerator(_globalModel.globalModel):        
    
    def __init__(self, modelPath, modelType, allFeatureNames, overwriteModel): 
        # Parameters that should be optimized during training.
        self.setOptimizedParameters()

        # Initialize common model class.
        super().__init__(modelPath, modelType, allFeatureNames, overwriteModel)
        
    def setOptimizedParameters(self):
        self.explorationWeight = 0.5  # Typically between 0.1 and 1.
        self.numStateSpaces = 10
    
    def _loadModel(self):
        with open(self.modelPath, 'rb') as handle:
            self.model = joblib.load(handle, mmap_mode ='r')
        
    def _saveModel(self):
        # Save the model
        joblib.dump(self, self.modelPath)
        
    # ---------------------------------------------------------------------- #
    # -------------------------- Initialize Model -------------------------- #
        
    def createModel(self):        
        # Define the tree structure interface.
        self.treeInterfaceClass = expressionTreeModel.expressionTreeModel()
        
        # Reset variable model parameters.
        self._resetModel()
                
    def _resetModel(self):    
        # Reset training parameters.
        self.numExpandedFeatures = None
        
        # General model parameters.
        self.numFeaturesUsed = [] # The number of unique features used in generating the model equation
        self.modelEquations = []  # The   
        self.modelR2s = []  # The
        
    # ---------------------------------------------------------------------- #
    # ----------------------- State-Action Interface ----------------------- # 
    
    def segmentStateSpace(self, numStateSpaces):
        # Assert the validity of segmented the state space.
        assert 1 <= numStateSpaces, "Must have at least one slice of the state space."
        assert isinstance(numStateSpaces, int), "Please input an intger number of state spaces."
        
        # Segment the state space.
        self.stateSpaces = np.linspace(-1, 1, num=numStateSpaces + 1, endpoint=True)
        # Add root node start.
        self.rootNodeStateValue = -1.1
        self.stateSpaces = np.insert(self.stateSpaces, 0, self.rootNodeStateValue)

    def initializeBaseState(self, transformedInputs, trueVals, baseStringEquation):
        # Initialize starting tree.
        self.baseExpressionTree = self.treeInterfaceClass.equationInterface(baseStringEquation, [])
        
        # Initialize the starting state
        self.baseStateValue = self.getStateValue(self.baseExpressionTree, transformedInputs, trueVals)
        self.baseStateIndex = self.getStateIndex(self.baseStateValue)
        
    def setupStateActionInfo(self):
        # Initialize state action-pair information.
        self.stateActionRewards = np.zeros((len(self.stateSpaces), self.treeInterfaceClass.numTransformers*self.numExpandedFeatures))
        self.numActionsTaken_perState = np.zeros((len(self.stateSpaces), self.treeInterfaceClass.numTransformers*self.numExpandedFeatures))
        
    def getBaseState(self):
        return deepcopy(self.baseExpressionTree), self.baseStateValue, self.baseStateIndex
            
    def getStateIndex(self, stateValue):
        return np.searchsorted(self.stateSpaces, stateValue, side='left') - 1
    
    def getStateBoundaries(self, stateIndex):
        return self.stateSpaces[stateIndex:stateIndex + 2]
    
    def getStateValue(self, expressionTree, transformedInputs, trueVals):
        # Base case: I ONLY have a root node, no similarity.
        if self.treeInterfaceClass.isStubTree(expressionTree):
            return self.rootNodeStateValue
        
        # Find the state information of the equation.
        predictedVals = self.treeInterfaceClass.expressionTreeInterface(expressionTree, transformedInputs)
        stateValue = self.calculateCosineSimilarity(trueVals, predictedVals)

        return stateValue
    
    def findOptimalAction(self, stateIndex):
        # Find the rewards for following the algorithm.
        numStateActionPairs = self.numActionsTaken_perState[stateIndex] + 1
        expectedActionRewards = self.stateActionRewards[stateIndex] / numStateActionPairs
        # Calculate the reward for exploring.
        explorationRewards = 1 / numStateActionPairs

        # Calculate a weighted average of exploration and exploitation.
        finalStateActionRewards = (1 - self.explorationWeight) * expectedActionRewards + self.explorationWeight * explorationRewards
        
        # Return the best action considering both reward types.
        return np.argmax(finalStateActionRewards)
    
    # ---------------------------------------------------------------------- #
    # ------------- Update Weights After Exploring an Equation ------------- # 
    
    def recordActionState(self, stateIndex, actionIndex, rewardValue):
        # Record taking the action in this state.
        self.numActionsTaken_perState[stateIndex][actionIndex] += 1
        self.stateActionRewards[stateIndex][actionIndex] += rewardValue
        
    def backPropogation(self, stateItinerary, actionItinerary, finalReward):
        # Perform back propogation.
        for iterationInd in range(len(actionItinerary)):
            # Get the iteration's action-state pair
            actionIndex = actionItinerary[iterationInd] # Action taken in this state.
            stateIndex = stateItinerary[iterationInd]   # Initial state.
            
            # Inform each state of the final reward for the action-state pair.
            self.recordActionState(stateIndex, actionIndex, finalReward)
    
    # ---------------------------------------------------------------------- #
    # ----------------- Finding an Equation from Base State ---------------- #  
    
    def findRandomAction(self):
        return random.randint(0, self.stateActionRewards.shape[1] - 1)
    
    def applyOptimalAction(self, expressionTree, stateValue, actionIndex, transformedInputs, trueVals):
        # print("\tactionIndex:", actionIndex)
        # expressionTree.prettyPrint()
        
        # Get the transformer associated with the action.
        transformerIndex = actionIndex // self.numExpandedFeatures
        transformer = self.treeInterfaceClass.getTransformer(transformerIndex)
        # Get the feature associated with the action.
        expandedFeatureInd = actionIndex % self.numExpandedFeatures
        featureInd = expandedFeatureInd % len(self.treeInterfaceClass.functions_onceApplied)
        
        # Create a tree node for the transformer.
        transformerSymbol = self.treeInterfaceClass.mapTransformer_toString[transformer]
        maxChildren = self.treeInterfaceClass.findMaxChildren_Transformer(transformerIndex)
        transformerNode = expressionTreeModel.treeNodeClass(transformer, stringValue = transformerSymbol, 
                                                            parentNode = None, maxChildren = maxChildren)
        # Create a tree node for the variable.
        variableName = self.finalFeatureNames[featureInd]
        variableNode = expressionTreeModel.treeNodeClass(featureInd, stringValue = variableName, 
                                                         parentNode = None, maxChildren = 0)
        
        # Base case: I ONYL have a root node.
        if self.treeInterfaceClass.isStubTree(expressionTree):
            # Only addition makes sense at this point.
            if transformer != self.treeInterfaceClass.add:
                return None, -np.inf
            # I can only place a single feature to the tree.
            self.treeInterfaceClass.insertChildNode(expressionTree, variableNode)
            stateValue = self.getStateValue(expressionTree, transformedInputs, trueVals)

            return expressionTree, stateValue
        
        # Setup breadth first search.
        queuedNodes = expressionTree.children.copy()
        # Initialize the best modified expressionTree.
        bestExpressionTree = None
        bestStateValue = stateValue
                        
        # While there are unexplored nodes.
        while len(queuedNodes) != 0:
            # Explore a new child node.
            actionNode = queuedNodes.pop(0)
            
            # Check if this transformer/variable should be evaluated.
            modifyActionNode = self.usefulEvaluation(actionNode, variableNode, transformerNode)
                            
            if modifyActionNode:
                # Apply the action to this node.
                self.treeInterfaceClass.insertNodeBetween(actionNode, transformerNode, actionNode.parentNode)
                self.treeInterfaceClass.insertChildNode(transformerNode, variableNode)  
                # expressionTree.prettyPrint()
                # Evaluate the benefit of this action.
                stateValue = self.getStateValue(expressionTree, transformedInputs, trueVals)
                if bestStateValue < stateValue:
                    bestStateValue = stateValue
                    bestExpressionTree = deepcopy(expressionTree)
                # Remove the action from this node.
                self.treeInterfaceClass.removeNode(variableNode)
                self.treeInterfaceClass.removeNode(transformerNode)
                print("\tPossible State Value", stateValue)
            
            # Store the actionNode's children.
            queuedNodes.extend(actionNode.children.copy())
        
        return bestExpressionTree, bestStateValue
    
    def usefulEvaluation(self, actionNode, variableNode, transformerNode):
        # If we are trying to add the same node as the action node.
        if actionNode.stringValue == variableNode.stringValue:
            # I should never do this.
            return False
        
        return True
        
    def expandExpressionTree(self, transformedInputs, trueVals):
        # Initialize a new base expression tree.
        expressionTree, stateValue, stateIndex = self.getBaseState()
        nextStateValue = stateValue; nextStateIndex = stateIndex
        # Initialize game state parameters.
        actionItinerary = []  # A list of actions in order.
        stateItinerary = []   # A list of states visited in order.
                
        # While there is a good move to make.
        while stateValue <= nextStateValue:
            print("\n\tMake Modification:", stateValue, nextStateValue)
            # Reset the current state
            stateValue = nextStateValue
            stateIndex = self.getStateIndex(nextStateValue)
                
            # Modify the equation with the optimal action.
            actionIndex = self.findOptimalAction(stateIndex)
            modifiedExpressionTree, nextStateValue = self.applyOptimalAction(deepcopy(expressionTree), stateValue, actionIndex, transformedInputs, trueVals)

            # If we are at a terminating state.
            if modifiedExpressionTree == None or nextStateValue < stateValue:
                # Record a bad move for taking this action in this state.
                self.recordActionState(stateIndex, actionIndex, nextStateValue)
                # Perform backpropogation of the final reward.
                self.backPropogation(stateItinerary, actionItinerary, stateValue)
                break
            else:
                # Record the modification.
                actionItinerary.append(actionIndex)
                stateItinerary.append(stateIndex)
                # Prepare for the next round.
                expressionTree = modifiedExpressionTree
                
        print("FINAL")
        expressionTree.prettyPrint()


            
    def expandInputs(self, newInputs, collectedTransformations = []):
        inputShape = newInputs.shape
        # Initialize holder for the transformations.
        if len(collectedTransformations) == 0:
            collectedTransformations = np.zeros((inputShape[0], inputShape[1]*len(self.treeInterfaceClass.functions_onceApplied)))
        
        # For each potental transformation.
        for transformerInd in range(len(self.treeInterfaceClass.functions_onceApplied)):
            transformer = self.treeInterfaceClass.functions_onceApplied[transformerInd]
            startColumnInd = inputShape[1]*transformerInd
            
            # While ignoring warnings.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Apply and store the transformed inputs.
                collectedTransformations[:, startColumnInd:startColumnInd + inputShape[1]] = transformer(newInputs)
        
        # Set the number of transformed features.
        if self.numExpandedFeatures != None:
            assert self.numExpandedFeatures == collectedTransformations.shape[1], "The number of features are inconsistent?"
        self.numExpandedFeatures = collectedTransformations.shape[1]
        
        return collectedTransformations







                    
                
    # ---------------------------------------------------------------------- #
    # ----------------------- Update Model Parameters ---------------------- #
    
    # ---------------------------------------------------------------------- #
    # ----------------------------- Apply Model ---------------------------- #
    
    def scoreModel(self, Testing_Data, Testing_Labels):
        if len(self.finalFeatureNames) == 0:
            print("Model Not Trained and cannot provide a score")
            return None
        
        # Hold true and predicted user ratings
        userRatings = Testing_Labels.T[3]
        predictedRatings = []
        
        # For each user-item instance.
        for pointInd in range(len(Testing_Labels)):
            # Organize variables for prediction.
            Ui = Testing_Data[pointInd, :]
            timePoint, userInd, itemInd, userItemRating = Testing_Labels[pointInd]
            userInd = int(userInd); itemInd = int(itemInd)
            
            # Predit the user rating.
            predictedRatings.append(self.predictPoint(Ui, timePoint, userInd, itemInd))
        predictedRatings = np.array(predictedRatings)
        
        # Calculate the R2 correlation between the predicted and given ratings.
        R2 = self.calculateR2(userRatings, predictedRatings, calculateAdjustedR2=True, numFeatures=0)
        
        # import matplotlib.pyplot as plt
        # plt.plot(userRatings, predictedRatings, "o")
        # plt.show()

        return R2
    
    def calculateR2(self, trueVal, predictedVal, calculateAdjustedR2 = False, numFeatures = 0):
        # Setup variables as numpy arrays
        trueVal = np.asarray(trueVal)
        predictedVal = np.asarray(predictedVal)
        
        # Calculate the variance of the dataset
        unexplainedVariance =  ((trueVal - predictedVal)**2).sum(axis = 0)
        totalVariance = ((trueVal - trueVal.mean(axis = 0))**2).sum(axis = 0)
        
        # Special case: the totalVariance is zero, the trueOutput are constant.
        if totalVariance == 0:
            # If the model worsened the prediction (the average), then R2 = 0.
            if unexplainedVariance == 0:
                return 0
            # Else, the model fully explains the data by keeping the average.
            return 1

        # If considering the degrees of freedom lost from extra features
        if calculateAdjustedR2:
            # Calculate the adjusted R2
            adjustedR2 = 1 - unexplainedVariance * (len(trueVal) - 1) / (totalVariance * (len(trueVal) - numFeatures - 1))
            return adjustedR2
        else:
            # Calculate the R2 score.
            R2 = 1 - unexplainedVariance / totalVariance
            return R2
        
    def calculateCosineSimilarity(self, trueVals, predictedVals):
        # Calculate the length of the predicted values.
        predictedValueNorm = np.linalg.norm(predictedVals)
        # If we have no vector, there is no similarity.
        if predictedValueNorm == 0:
            return self.rootNodeStateValue
        
        # Return the cosine similarity
        return (trueVals*predictedVals).sum()/(predictedValueNorm * np.linalg.norm(trueVals))

    def predictPoint(self, Ui, timePoint, userInd, itemInd):
        # Gather rating contributions
        ratingMF = self.collabFiltRating(Ui, userInd, itemInd)
        ratingCB = self.contentBasedRating(Ui, timePoint, userInd, itemInd)
                
        # Predict label based on new Data
        predictedRatings = self.alpha * ratingMF + (1 - self.alpha) * ratingCB
        return predictedRatings     
    
    def predict(self, U, timePoints, userInds, itemInds):
        finalPredictions = []
        for pointInd in range(len(U)):
            Ui = U[pointInd]
            timePoint = timePoints[pointInd]
            userInd = userInds[pointInd] if type(userInds) not in [float, int, str] else userInds
            itemInd = itemInds[pointInd] if type(itemInds) not in [float, int, str] else itemInds
            
            if type(userInd) == str:
                userInd = np.where(self.userNames == userInd)[0][0]
            if type(itemInd) == str:
                itemInd = np.where(self.itemNames == itemInd)[0][0]
            userInd = int(userInd); itemInd = int(itemInd)
            
            finalPredictions.append(self.predictPoint(Ui, timePoint, userInd, itemInd))
        finalPredictions = np.array(finalPredictions)
        
        return finalPredictions

    # ---------------------------------------------------------------------- #
    # ---------------------------- Train Model ----------------------------- #
    
    def trainModel(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels, featureNames, maxEpochs = 300):  
        # Assert the integrity of the input variables.
        assert len(Testing_Data) == len(Testing_Labels), "Testing points have to map 1:1 to testing labels."
        assert len(Training_Data) == len(Training_Labels), "Training points have to map 1:1 to training labels."
        assert len(featureNames) == len(Training_Data[0]) == len(Testing_Data[0]), "The featureNames should have the same length as all the features in testing/training sets."
        # Setup variables as numpy arrays
        Training_Data = np.asarray(Training_Data.copy())
        Training_Labels = np.asarray(Training_Labels.copy())
        Testing_Data = np.asarray(Testing_Data.copy())
        Testing_Labels = np.asarray(Testing_Labels.copy())
        # Save the information we trained on.
        self.finalFeatureNames = np.asarray(featureNames)
        # Start model from scratch
        self._resetModel()
        
        # Expand the input data to include common transformatios.
        Testing_Data = self.expandInputs(Testing_Data, [])
        Training_Data = self.expandInputs(Training_Data, [])
        # Initialize state-action pair information.
        self.segmentStateSpace(self.numStateSpaces)
        self.setupStateActionInfo()
        # Initialize base state expression tree.
        self.initializeBaseState(Training_Data, Training_Labels.reshape(1,-1)[0], baseStringEquation = "")
                
        for i in range(1000):
            print("\n\nEnter")
            self.expandExpressionTree(Training_Data, Training_Labels.reshape(1,-1)[0])

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
    
    numPoints = 100000
    # Specify input features.
    x = np.random.uniform(1, 4, numPoints)
    y = np.random.uniform(2, 5, numPoints)
    z = np.random.uniform(3, 6, numPoints)
    a = np.random.uniform(4, 7, numPoints)
    b = np.random.uniform(5, 8, numPoints)
    c = np.random.uniform(6, 9, numPoints)
    m = np.random.uniform(1, 9, numPoints)
    # True labels
    featureLabels_EGM = m*x + y*x
    
    # Compile feature data.
    featureData_EGM = np.array([x, y, z, m, a, b, c]).T
    featureNames_EGM = np.array(['x', 'y', 'z', 'm', 'a', 'b', 'c'])
    maxEpochs = 5


    # Instantiate class.
    modelClass = equationGenerator(modelPath, modelType, allFeatureNames, overwriteModel)
    # Train model
    Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(featureData_EGM, featureLabels_EGM, test_size=0.2, shuffle= True)
    modelClass.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels, featureNames_EGM, maxEpochs)
    
    
    # numPoints = 100
    # x = np.random.uniform(2, 10, numPoints)
    # y = np.random.uniform(2, 10, numPoints)
    # a = np.random.uniform(2, 10, numPoints)
    # b = np.random.uniform(2, 10, numPoints)
    # ones = np.ones(len(x))    
    
    # xy = x*y
    # sin = np.sin(x)
    # sinY = np.sin(y)
    # sinY = np.sin(xy)
    
    # trueOutput = sinY + x*y + x**(y)
    # trueOutput = sinY + x*y
    
    # modelClass.calculateR2(trueOutput, x, calculateAdjustedR2=True, numFeatures=0)

    # X = np.array([x, y, a, b, ones*3.14, ones*9.81, np.sin(x), np.sin(y), np.sin(a), np.sin(b), np.cos(x), np.cos(y), np.cos(a), np.cos(b), np.exp(x), np.exp(y), np.exp(a), np.exp(b)]).T
    # Y = trueOutput[:, None]
    
    # theta = np.linalg.inv((X.T @ X)) @ X.T @ Y
    # modelClass.calculateR2(trueOutput, (X@theta).T[0], calculateAdjustedR2=True, numFeatures=0)
    


