
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic modules
import os
import sys
import joblib
import sklearn
import numpy as np
import sklearn.metrics

# Import files
import _globalModel # Global model class

# Import files
sys.path.append(os.path.dirname(__file__) + "/../../Feedback Control/Virtual Reality Control/")
import _imageSimilarities


# -------------------------------------------------------------------------- #
# -------------------------- Recommendation Model -------------------------- #

class recommendationModel(_globalModel.globalModel):
    
    def __init__(self, modelPath, modelType, allFeatureNames, overwriteModel, numUsers = 20, numBioFeatures = 84, numItems = 4):
        """
        Translation to mathmatical notation:
        ----------------------------------------------------------------------
            M = self.numUsers : The number of users who has seen the recomendations.
            N = self.numItems : The number of possible recomendations.
            K = self.numLatentFactors : The arbitrary dimension Users/Recommendation matrices are assumed to have.
        ----------------------------------------------------------------------
        """        
        # Matrix factorization parameters.
        self.numUsers = numUsers
        self.numItems = numItems
        self.numBioFeatures = numBioFeatures
        # General model parameters
        self.maxTimeRemember = 60*10
        self.W = np.identity(self.numItems) #self.getSimilarityMatrix() # Initialize weight matrix: similarity between items.
        
        # Parameters that should be optimized during training.
        self.setOptimizedParameters(learningRate = 0.001, regularization = 0.1, numLatentFactors = 20, lossThreshold = 0.001)

        # Initialize common model class.
        super().__init__(modelPath, modelType, allFeatureNames, overwriteModel)
        
    # ---------------------------------------------------------------------- #
    # -------------------------- Initialize Model -------------------------- #
        
    def createModel(self):
        """
        Initialize/reset the model parameters:
            𝑅(𝑡) = 𝛼(𝜇 + 𝛽_𝑢 + 𝛽_𝑣 + (𝑈(𝑡) - 𝛽_𝑏)𝐵𝑉) + (1 − 𝛼)∗(∑(P(𝑡) ∗ 𝑅(𝑡)))*W
        ----------------------------------------------------------------------
            𝑅 is the predicted rating of an item by a user. Each row is one user with all item ratings. Dim: numUsers x numItems.
            𝜇 is the overall average rating across all users, biological features, and items. Set to zero if no information. Dim: scalar.
            𝛽_𝑢  is the user-specific deviation from the average rating. Each element is one user's average rating. Dim: 1 x numUsers.
            𝛽_b is the feature-specific baseline for each user. Each row is one user with all the feature's baselines. Dim: numUsers x numFeatures.
            𝛽_𝑣 is the item-specific deviation from the average rating. Each element is one item's average rating. Dim: 1 x numItems.
            𝑈 is a matrix representing the user's biological state. Each row is one user with all biological features (at time t). Dim: numUsers x numFeatures.
            𝐵 is a matrix representing the biological influence on the latent factors. Each row is one feature with all latent factor influences. Dim: numFeatures x numLatentFactors.
            𝑉 is a vector representing the latent factor's influence on the items. Each row is a latent factor with all item influences. Dim: numLatentFactors x numItems.
            W is a similarity matrix between items (used to estimate rating of unseen items). Each column represents one item with its similarity to other items in the rows. Dim: numItems x numItems.
            𝛼 is the weight percent collaborative filtering has to the final rating (in comparison to content based rating).
        ----------------------------------------------------------------------
        """
        # Inform the user the model has been created.
        self.model = self
        
        # Reset the model's fixed parameters
        # self._resetModel()
        
        # Initialize matrix factorization parameters.
        self.B = np.random.uniform(-0.1, 0.1, (self.numBioFeatures, self.numLatentFactors)) 
        self.V = np.random.uniform(-0.1, 0.1, (self.numLatentFactors, self.numItems))
        # Normalize B and V
        self.B /= np.linalg.norm(self.B, ord="fro")
        self.V /= np.linalg.norm(self.V, ord="fro")
        # Initialize bias corrections.
        self.mu = 0
        self.userBias = np.zeros(self.numUsers)
        self.itemBias = np.zeros(self.numItems)
        self.bioBaseline = np.zeros((self.numUsers, self.numBioFeatures)) 
        # Initialize collaborative/content filtering weight fo the final rating.
        self.alpha = 1
        # Initialize content based parameters.
        minTimeDelay = 1
        self.decayRates = np.zeros(self.numUsers)
        self.numPastRatings = np.zeros(self.numUsers, dtype = int)
        self.pastRatingTimes = - np.ones((self.numUsers, int(self.maxTimeRemember/minTimeDelay)+1))
        self.pastUserItemRatings = - np.ones((self.numUsers, int(self.maxTimeRemember/minTimeDelay)+1, self.numItems)) 
        
        self.W = np.identity(self.numItems)
        
    def _resetModel(self):
        self.createModel()
        self.clearPastRatings()
        
        # Keep track of items and users we train on.
        self.userNames = None
        self.itemNames = None
        
    def summarizeModelParams(self):
        print("\nScalar Model Parameters")
        print("\t𝜇:", self.mu)
        print("\t𝛼:", self.alpha)
        
        print("\nBias Parameters")
        print("\t𝛽_𝑢:", self.userBias)
        print("\t𝛽_𝑣:", self.itemBias)
        print("\t𝛽_b:", self.bioBaseline)
        
        print("\t𝜆:", self.decayRates)
        
        print("\t𝐵:", self.B)
        print("\t𝑉:", self.V)
    
    def setOptimizedParameters(self, learningRate, regularization, numLatentFactors, lossThreshold):
        # Parameters that should be optimized.
        self.learningRate = learningRate
        self.numLatentFactors = numLatentFactors
        self.lossThreshold = lossThreshold
        self.regularization = regularization
        
        # Recreate the model with the new parameters.
        self.createModel()
        
    def getSimilarityMatrix(self):
        # Instantiate class.
        imageComparison = _imageSimilarities.imageSimilarities()
        # Collect the image files.
        imageFolder = os.path.dirname(__file__) + "/../../Feedback Control/Virtual Reality Control/Virtual Images/"
        imagePaths = imageComparison.getImagePaths(imageFolder)
        # Calculate similarity matrix
        similarity_matrix = imageComparison.get_similarity_matrix(imagePaths)
        
        return similarity_matrix
    
    def forgettingFuntion(self, decayRate, timeDelays):
        """ Returns a normalized probability for all timeDelays"""
        # Setup variables as numpy arrays
        timeDelays = np.asarray(timeDelays)
        
        # Calculate the decay weights and its normalization factor
        decayWeights = np.exp(-decayRate * timeDelays)
        normalization = decayWeights.sum() # Calculate the integral of the decay weights
        
        return decayWeights / normalization

    # ---------------------------------------------------------------------- #
    # --------------------- Update Past Rating Holders --------------------- #

    def resetPastRatings(self, userInd):
        # Remove all previous ratings.
        self.pastUserItemRatings[userInd][0:self.numPastRatings[userInd], :] = -1
        self.pastRatingTimes[userInd][0:self.numPastRatings[userInd]] = -1
        self.numPastRatings[userInd] = 0
        
    def clearPastRatings(self):
        # clear all ratings.
        self.numPastRatings[:] = 0
        self.pastRatingTimes[:] = -1
        self.pastUserItemRatings[:] = -1
    
    def addNewRating(self, Ui, timePoint, userInd, itemInd, userItemRating = None):
        assert timePoint >= 0
        
        # Shift the ratings in the holder to account for the new timePoint.
        self.updatePastRatingMatrix(timePoint, userInd) 

        # Predict the ratings if no ratn given.
        if userItemRating == None:
            userItemRating = self.predictPoint(Ui, timePoint, userInd, itemInd)

        # Update the records with the new item ratings.
        self.pastRatingTimes[userInd][self.numPastRatings[userInd]] = timePoint
        self.pastUserItemRatings[userInd][self.numPastRatings[userInd]][itemInd] = userItemRating
        self.numPastRatings[userInd] += 1
                
    def updatePastRatingMatrix(self, timePoint, userInd):
        # Special case: no ratings have been seen for this user.        
        if self.numPastRatings[userInd] == 0:
            return None
        # Special case: the new rating is less than the last rating.
        elif timePoint <= self.pastRatingTimes[userInd][max(0, self.numPastRatings[userInd]-1)]:
            # Then we finished a previous experiment and started a new one.
            self.resetPastRatings(userInd)
            return None
    
        # Get the oldest rating of the user.
        removeRatingPointer = 0; 
        previousRatingTime = self.pastRatingTimes[userInd][removeRatingPointer];
        
        # Find how many of the oldest ratings are out of bounds.
        while previousRatingTime < timePoint - self.maxTimeRemember:
            removeRatingPointer += 1
            # Only analyze the times in the array (not the buffer)
            if removeRatingPointer == self.numPastRatings[userInd]:
                break
            
            # Analyze the next oldest rating time
            previousRatingTime = self.pastRatingTimes[userInd][removeRatingPointer]
                            
        numRatingsKeeping = self.numPastRatings[userInd] - removeRatingPointer
        # Update the past rating holder to only include recent ratings.
        self.pastUserItemRatings[userInd][0:numRatingsKeeping] = self.pastUserItemRatings[userInd][removeRatingPointer:self.numPastRatings[userInd]]
        self.pastRatingTimes[userInd][0:numRatingsKeeping] = self.pastRatingTimes[userInd][removeRatingPointer:self.numPastRatings[userInd]]
        # Reset the remaining half of the matix.
        self.pastUserItemRatings[userInd][numRatingsKeeping:self.numPastRatings[userInd], :] = -1
        self.pastRatingTimes[userInd][numRatingsKeeping:self.numPastRatings[userInd]] = -1
        # Reset the poiner
        self.numPastRatings[userInd] = numRatingsKeeping
                
    # ---------------------------------------------------------------------- #
    # ----------------------- Update Model Parameters ---------------------- #
    
    def gradientUpdate(self):
        # Update the matrix factorization terms using gradient descent.
        self.B -= self.learningRate * (self.dE_dB + self.regularization * np.abs(self.B))
        self.V -= self.learningRate * (self.dE_dV + self.regularization * np.abs(self.V))
        # Update the bias terms using gradient descent.
        self.bioBaseline -= self.learningRate * (self.dE_dBb + self.regularization * np.abs(self.bioBaseline))
        self.userBias -= self.learningRate * (self.dE_dBu + self.regularization * np.abs(self.userBias))
        self.itemBias -= self.learningRate * (self.dE_dBv + self.regularization * np.abs(self.itemBias))
        self.mu -= self.learningRate * (self.dE_dMu + self.regularization * np.abs(self.mu))
        # Update content-based filtering parameters using gradient descent.
        self.decayRates -= self.learningRate * (self.dE_dLambda + self.regularization * np.abs(self.decayRates))
        self.alpha -= self.learningRate * (self.dE_dAlpha + self.regularization * np.abs(self.alpha))
        
        # Add constaints to the variables
        self.alpha = min(max(self.alpha, 0), 1) # Constrain the alpha weight between 0 and 1.
    
    def updateBioMatrix(self, Ui, timePoint, userInd, itemInd, userItemRating):
        """ Using regularized gradient descent, update the weight of B. """
        # Get the current parameters for the update.
        modelError = userItemRating - self.predictPoint(Ui, timePoint, userInd, itemInd)
        Vk = self.V[:, itemInd]

        # Calculate the gradient of the loss function.
        self.dE_dB += - modelError * self.alpha * np.outer(Ui - self.bioBaseline[userInd], Vk)
                
    def updateItemMatrix(self, Ui, timePoint, userInd, itemInd, userItemRating):
        """ Using regularized gradient descent, update the weight of V. """
        # Get the current parameters for the update.
        modelError = userItemRating - self.predictPoint(Ui, timePoint, userInd, itemInd)
        
        # Calculate the gradient of the loss function.
        dE_dV = - modelError * self.alpha * (Ui - self.bioBaseline[userInd]) @ self.B + self.regularization * np.abs(self.V[:, itemInd])

        # Update the term using gradient descent.
        self.V[:, itemInd] -= self.learningRate * dE_dV
        
    def updateBioBaselineMatrix(self, Ui, timePoint, userInd, itemInd, userItemRating):
        """ Using regularized gradient descent, update the weight of Bb. """
        # Get the current parameters for the update.
        modelError = userItemRating - self.predictPoint(Ui, timePoint, userInd, itemInd)
        Vk = self.V[:, itemInd]
        
        # Calculate the gradient of the loss function.
        dE_dBb = modelError * self.alpha * Vk @ self.B.T + self.regularization * np.abs(self.bioBaseline[userInd, :])
        
        # Update the term using gradient descent.
        self.bioBaseline[userInd] -= self.learningRate * dE_dBb
                        
    def updateUserBiasVector(self, Ui, timePoint, userInd, itemInd, userItemRating):
        """ Using regularized gradient descent, update the weight of Bu. """
        # Get the current parameters for the update.
        modelError = userItemRating - self.predictPoint(Ui, timePoint, userInd, itemInd)
                
        # Calculate the gradient of the loss function.
        dE_dBu = - modelError * self.alpha + self.regularization * np.abs(self.userBias[userInd])
    
        # Update the term using gradient descent.
        self.userBias[userInd] -= self.learningRate * dE_dBu
        
    def updateItemBiasVector(self, Ui, timePoint, userInd, itemInd, userItemRating):
        """ Using regularized gradient descent, update the weight of Bv. """
        # Get the current parameters for the update.
        modelError = userItemRating - self.predictPoint(Ui, timePoint, userInd, itemInd)
        
        # Calculate the gradient of the loss function.
        dE_dBv = - modelError * self.alpha + self.regularization * np.abs(self.itemBias[itemInd])
            
        # Update the term using gradient descent.
        self.itemBias[itemInd] -= self.learningRate * dE_dBv
        
    def updateMu(self, Ui, timePoint, userInd, itemInd, userItemRating):
        """ Using regularized gradient descent, update the weight of mu. """
        # Get the current parameters for the update.
        modelError = userItemRating - self.predictPoint(Ui, timePoint, userInd, itemInd)
        
        # Calculate the gradient of the loss function.
        dE_dMu = - modelError * self.alpha + self.regularization * np.abs(self.mu)
            
        # Update the term using gradient descent.
        self.mu -= self.learningRate * dE_dMu
    
    def updateAlpha(self, Ui, timePoint, userInd, itemInd, userItemRating):
        """ Using regularized gradient descent, update the weight of alpha. """
        # Get the current parameters for the update.
        modelError = userItemRating - self.predictPoint(Ui, timePoint, userInd, itemInd)
        
        # Calculate the gradient of the loss function.
        dE_dAlpha = - modelError * (self.collabFiltRating(Ui, userInd, itemInd) - self.contentBasedRating(Ui, timePoint, userInd, itemInd))
        
        # Update the term using gradient descent.
        self.alpha -= self.learningRate * dE_dAlpha
                    
    def updateDecayRate(self, Ui, timePoint, userInd, itemInd, userItemRating):
        """ Using regularized gradient descent, update the weight of lambda. """
        # Special case: we have no past ratings.
        if self.numPastRatings[userInd] == 0: return None
        
        # Get the model information.
        decayRate = self.decayRates[userInd]
        modelError = userItemRating - self.predictPoint(Ui, timePoint, userInd, itemInd)
                        
        # Get the recent ratings and deltaTime of the rating for the user.
        pastItemRatings = self.pastUserItemRatings[userInd][0:self.numPastRatings[userInd], :]
        ratingTimeDelays = timePoint - self.pastRatingTimes[userInd][0:self.numPastRatings[userInd]][:, None]
        # Get the normalized item weights.
        pastRatingWeights = self.forgettingFuntion(decayRate, ratingTimeDelays)
        
        # If there is a decay constant.
        if decayRate != 0:
            # Calculate the derivative of the rating weight with respect to the decay constant.
            allExpDecay = np.exp(-decayRate * ratingTimeDelays)      # exp(-𝜆t)
            lastExpDecay = np.exp(-decayRate * self.maxTimeRemember) # exp(-𝜆T), T = maximum time delay.
            dP_dLambda = allExpDecay * (1 - (decayRate*self.maxTimeRemember*lastExpDecay / (1-lastExpDecay)) - decayRate*ratingTimeDelays) / (1-lastExpDecay)
        else:
            # Special case derivative (1rst order taylor expansion) if the decay constant is zero.
            dP_dLambda = 0.5 - ratingTimeDelays/self.maxTimeRemember

        # Sum all the weights with item ratings. This should be 1 when an item is rated for all times.
        normRatingWeights = ((0 <= pastItemRatings) * pastRatingWeights).sum(axis=0)  # Do not consider non-rated item while looking at the sum of all weights.
        normRatingWeights[normRatingWeights == 0] = -1                                  # If all zeros, normalization doesnt matter as there were no ratings for that item.
        # Sum all the weights with item ratings. This should be 1 when an item is rated for all times.
        normRatingDerivWeights = ((0 <= pastItemRatings) * dP_dLambda).sum(axis=0)    # Do not consider non-rated item while looking at the sum of all weights.
        normRatingDerivWeights[normRatingDerivWeights == 0] = -1                        # If all zeros, normalization doesnt matter as there were no ratings for that item.

        # Calculate the average past rating (normalized integral of the weight * rating)
        expectedRating = (pastItemRatings * (0 <= pastItemRatings) * pastRatingWeights).sum(axis=0) / normRatingWeights
        expectedRating[normRatingWeights == -1] = -1 
        # Calculate the average past rating using derivative probability (integral of the dWeight/dDecay * rating)
        expectedRating_dP_dLamba = (pastItemRatings * (0 <= pastItemRatings) * dP_dLambda).sum(axis=0)
        expectedRating_dP_dLamba[normRatingDerivWeights == -1] = -1 
        # Calculate the average past rating derivative (integral of the weight * dRating/dDecay)
        # expectedRatingDerivative = (pastItemRatings * (0 <= pastItemRatings) * pastRatingWeights).sum(axis=0) / normRatingWeights
        # expectedRatingDerivative[normRatingDerivWeights == -1] = -1
        expectedRatingDerivative = 0
        
        # Calculate the derivative of the content based recommendation with respect to lambda for each item.                
        dCB_dLambda = expectedRating_dP_dLamba/normRatingWeights + expectedRatingDerivative - expectedRating*normRatingDerivWeights/normRatingWeights
        dCB_dLambda[normRatingWeights == -1] = -1 
        
        # Remove unrated items
        similarItemWeight = self.W[:, itemInd][0 <= dCB_dLambda]
        dCB_dLambda = dCB_dLambda[0 <= dCB_dLambda]
        # Add normalization factor to the item weights.
        normalizationItemWeight = similarItemWeight.sum()
        # Special case: no items are similar to the unrated item.
        if normalizationItemWeight == 0:
            return None
        
        # Calculate the gradient of the loss function.
        dE_dLambda = modelError * self.alpha * (dCB_dLambda @ similarItemWeight) / normalizationItemWeight + self.regularization * np.abs(self.decayRates[userInd])
        # Update the term using gradient descent.
        self.decayRates[userInd] -= self.learningRate * dE_dLambda
        
    # ---------------------------------------------------------------------- #
    # ----------------------------- Apply Model ---------------------------- #
    
    def collabFiltRating(self, Ui, userInd, itemInd):
        # Calculate ratingMF
        matrixFactorizationTerm = ((Ui - self.bioBaseline[userInd]) @ self.B) @ self.V[:, itemInd] 

        # print(self.mu , self.userBias[userInd] , self.itemBias[itemInd] , matrixFactorizationTerm)                       
        return self.mu + self.userBias[userInd] + self.itemBias[itemInd] + matrixFactorizationTerm
    
    def contentBasedRating(self, Ui, timePoint, userInd, itemInd):  
        # Special case: we have no past ratings.
        if self.numPastRatings[userInd] == 0:
            return self.collabFiltRating(Ui, userInd, itemInd)
        
        # Get the recent ratings and deltaTime of the rating for the user.
        pastItemRatings = self.pastUserItemRatings[userInd][0:self.numPastRatings[userInd], :]
        ratingTimeDelays = timePoint - self.pastRatingTimes[userInd][0:self.numPastRatings[userInd]]
        
        # Calculate the weight of each past rating
        decayRate = self.decayRates[userInd]
        pastRatingWeights = self.forgettingFuntion(decayRate, ratingTimeDelays)[:, None]
        
        # Sum all the weights with item ratings. This should be 1 when an item is rated for all times.
        normRatingWeights = ((0 <= pastItemRatings) * pastRatingWeights).sum(axis = 0)
        normRatingWeights[normRatingWeights == 0] = -1 # If all zeros, normalization doesnt matter.
        # Calculate the average past rating (normalized integral of the decay * rating)
        expectedRating = (pastItemRatings * (0 <= pastItemRatings) * pastRatingWeights).sum(axis = 0) / normRatingWeights
        expectedRating[normRatingWeights == -1] = -1 
        
        # Remove unrated items
        similarItemWeight = self.W[:, itemInd][0 <= expectedRating]
        expectedRating = expectedRating[0 <= expectedRating]
        # Add normalization factor to the item weights.
        normalizationItemWeight = similarItemWeight.sum(axis = 0)
        # Special case: no items are similar to the unrated item.
        if normalizationItemWeight == 0:
            return self.collabFiltRating(Ui, userInd, itemInd)

        # Smooth the ratings using content based filtering.
        return (expectedRating @ similarItemWeight) / normalizationItemWeight
            
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
            userInd = int(userInd)
            itemInd = int(itemInd)
            
            # Predit the user rating.
            predictedRatings.append(self.predictPoint(Ui, timePoint, userInd, itemInd))
        predictedRatings = np.array(predictedRatings)
        
        # Calculate the R2 correlation between the predicted and given ratings.
        R2 = self.calculateR2(userRatings, predictedRatings, calculateAdjustedR2 = False)
        
        # import matplotlib.pyplot as plt
        # plt.plot(userRatings, predictedRatings, "o")
        # plt.show()

        return R2
    
    def calculateR2(self, trueVal, predictedVal, calculateAdjustedR2 = False):
        # Setup variables as numpy arrays
        trueVal = np.asarray(trueVal)
        predictedVal = np.asarray(predictedVal)
        
        # Calculate the variance of the dataset
        unexplainedVariance =  ((trueVal - predictedVal)**2).sum(axis = 0)
        totalVariance = ((trueVal - trueVal.mean(axis = 0))**2).sum(axis = 0)
        
        # Special case: the totalVariance is zero, the trueVals are constant.
        if totalVariance == 0:
            # If the model worsened the prediction (the average), then R2 = 0.
            if unexplainedVariance == 0:
                return 0
            # Else, the model fully explains the data by keeping the average.
            return 1

        # If considering the degrees of freedom lost from extra features
        if calculateAdjustedR2:
            # Calculate the adjusted R2
            adjustedR2 = 1 - unexplainedVariance * (len(trueVal) - 1) / (totalVariance * (len(trueVal) - self.numBioFeatures - 1))
            return adjustedR2
        else:
            # Calculate the R2 score.
            R2 = 1 - unexplainedVariance / totalVariance
            return R2

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
    
    def trainModel(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels, featureNames, userNames, itemNames, maxEpochs = 300):  
        # Assert the integrity of the input variables.
        assert len(Testing_Data) == len(Testing_Labels), "Testing points have to map 1:1 to testing labels."
        assert len(Training_Data) == len(Training_Labels), "Training points have to map 1:1 to training labels."
        assert len(Training_Labels[0]) == len(Testing_Labels[0]) == 4, "Each label should have the following information: timePoint, userInd, itemInd, userItemRating."
        assert len(featureNames) == len(Training_Data[0]) == len(Testing_Data[0]), "The featureNames should have the same length as all the features in testing/training sets."
        # Setup variables as numpy arrays
        Training_Data = np.asarray(Training_Data)
        Training_Labels = np.asarray(Training_Labels)
        Testing_Data = np.asarray(Testing_Data)
        Testing_Labels = np.asarray(Testing_Labels)
        # Save the information we trained on.
        self.finalFeatureNames = featureNames
        self.userNames = np.asarray(userNames)
        self.itemNames = np.asarray(itemNames)
                
        if len(self.userNames) != self.numUsers:
            print("\tMore users given than expected. The model will be overwritten.")
            self.numUsers = len(self.userNames)
            self.createModel()
        if len(self.itemNames) != self.numItems:
            print("\tMore items given than expected. The model will be overwritten.")
            self.numItems = len(self.itemNames)
            self.createModel()
        if len(self.finalFeatureNames) != self.numBioFeatures:
            print("\tMore features given than expected. The model will be overwritten.")
            self.numBioFeatures = len(self.finalFeatureNames)
            self.createModel()
        self.createModel()
                
        # Keep track of error
        trainingError = []
        testingError = []
        # For each training epoch.
        for epoch in range(maxEpochs):
            initialLoss = self.scoreModel(Training_Data, Training_Labels)
            # Refresh past ratings.
            self.clearPastRatings()
            
            # For each user-item instance.
            for pointInd in range(len(Training_Labels)):
                # Organize variables for SGD.
                Ui = Training_Data[pointInd, :]
                timePoint, userInd, itemInd, userItemRating = Training_Labels[pointInd]
                userInd = int(userInd); itemInd = int(itemInd)
                
                # Shift the ratings in the holder to account for the new timePoint.
                self.updatePastRatingMatrix(timePoint, userInd) 
                # print(self.summarizeModelParams())
                if np.isnan(self.alpha): sys.exit()
            
                # Update the matrix factorization terms using gradient descent.
                self.updateBioMatrix(Ui, timePoint, userInd, itemInd, userItemRating)
                self.updateItemMatrix(Ui, timePoint, userInd, itemInd, userItemRating)
                # Update the bias terms using gradient descent.
                self.updateBioBaselineMatrix(Ui, timePoint, userInd, itemInd, userItemRating)
                self.updateUserBiasVector(Ui, timePoint, userInd, itemInd, userItemRating)
                self.updateItemBiasVector(Ui, timePoint, userInd, itemInd, userItemRating)
                self.updateMu(Ui, timePoint, userInd, itemInd, userItemRating)
                # Update content-based filtering parameters using gradient descent.
                self.updateDecayRate(Ui, timePoint, userInd, itemInd, userItemRating)
                self.updateAlpha(Ui, timePoint, userInd, itemInd, userItemRating)
                
                # Update past ratings holder.
                self.addNewRating(Ui, timePoint, userInd, itemInd, userItemRating) # Add the new timePoint to the holder.


            # Keep track of error
            trainingError.append(self.scoreModel(Training_Data, Training_Labels))
            testingError.append(self.scoreModel(Testing_Data, Testing_Labels))
        
            # Stopping condition.
            finalLoss = trainingError[-1]
            if initialLoss != 0 and abs((finalLoss - initialLoss)/initialLoss) < self.lossThreshold: # and 0.8 < finalLoss:
                print("\tStopped training early at epoch:", epoch)
                break
            
        import matplotlib.pyplot as plt
        plt.plot(trainingError, 'k', linewidth=2, label="Training Error")
        plt.plot(testingError, 'tab:red', linewidth=2, label="Testing Error")
        plt.show()
        
        # Return the final score
        print(testingError)
        return testingError[-1]

    # ---------------------------------------------------------------------- #
    # --------------------------- General Methods -------------------------- #
    
    def _loadModel(self):
        with open(self.modelPath, 'rb') as handle:
            self.model = joblib.load(handle, mmap_mode ='r')
        
    def _saveModel(self):
        # Save the model
        joblib.dump(self, self.modelPath)
    
# ---------------------------------------------------------------------------#

if __name__ == "__main__":
    # Set model paramters
    modelPath = "./CF.pkl"
    modelType = "CF"
    allFeatureNames = [""]
    overwriteModel = True
    # Set 
    numItems = 27
    numUsers = 20
    numBioFeatures = 100
    
    userNames = [str(i) for i in range(numUsers)]
    itemNames = [str(i) for i in range(numItems)]
    
    # Instantiate class.
    recommendationModel = recommendationModel(modelPath, modelType, allFeatureNames, overwriteModel, numUsers, numBioFeatures, numItems)
    
    Ui = np.random.uniform(0, 1, numBioFeatures)
    Ui = (Ui - np.mean(Ui))/np.std(Ui, ddof=1)
    timePoint = 0
    userInd = 2
    itemInd = 5
    userItemRating = 30
    
    Training_Data = np.array([Ui, Ui/2, Ui, Ui*2, Ui, Ui*2, Ui/2, Ui*4])
    Training_Labels = np.array([[timePoint, userInd+1, itemInd, userItemRating], 
                                [timePoint+10, userInd+1, itemInd+1, userItemRating/2],
                                [timePoint+20, userInd+1, itemInd, userItemRating],
                                [timePoint+30, userInd+1, itemInd+1, userItemRating*2],
                                [timePoint, userInd, itemInd+1, userItemRating],
                                [timePoint+10, userInd, itemInd+1, userItemRating/3],
                                [timePoint+20, userInd, itemInd, userItemRating],
                                [timePoint+30, userInd, itemInd, userItemRating*3]
                                ], dtype=int)
    Testing_Data = np.array([Ui, Ui*2, Ui, Ui*4])
    Testing_Labels = np.array([[timePoint, userInd+1, itemInd, userItemRating], 
                                [timePoint+30, userInd+1, itemInd+1, userItemRating*2],
                                [timePoint, userInd, itemInd+1, userItemRating],
                                [timePoint+30, userInd, itemInd, userItemRating*3]
                                ], dtype=int)
    featureNames = np.array([str(elem) for elem in Ui])
    
    R2 = recommendationModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels, featureNames, userNames, itemNames)
    print("R2 =", R2)
    
    # For each user-item instance.
    for pointInd in range(len(Training_Labels)):
        # Organize variables for prediction.
        Ui = Training_Data[pointInd, :]
        timePoint, userInd, itemInd, userItemRating = Training_Labels[pointInd]
        # Predit the user rating.
        predictedRating = recommendationModel.predictPoint(Ui, timePoint, userInd, itemInd)
        
        print("\nUser:", userInd, "; itemInd:", itemInd, "; True:", userItemRating, "; Predicted:", predictedRating)
        
    # For each user-item instance.
    for pointInd in range(len(Testing_Labels)):
        # Organize variables for prediction.
        Ui = Testing_Data[pointInd, :]
        timePoint, userInd, itemInd, userItemRating = Testing_Labels[pointInd]
        # Predit the user rating.
        predictedRating = recommendationModel.predictPoint(Ui, timePoint, userInd, itemInd)
        
        print("\nUser:", userInd, "; itemInd:", itemInd, "; True:", userItemRating, "; Predicted:", predictedRating)            

    
    
    