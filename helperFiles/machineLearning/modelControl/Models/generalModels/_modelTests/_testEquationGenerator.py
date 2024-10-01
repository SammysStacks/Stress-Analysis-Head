
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic modules
import numpy as np

# Import files
import recommendationModel # Global model class

# -------------------------------------------------------------------------- #
# -------------------------- Recommendation Model -------------------------- #

class testCollabFilt:
    
    def __init__(self):
        # Set general model paramters.
        self.modelPath = "./testModel.pkl"
        self.modelType = "testModel"
        self.allFeatureNames = ["testFeature"]
        self.overwriteModel = True
        
        # Set general model parameters.
        self.numItems = 3; self.numUsers = 2; self.numBioFeatures = 4
        self.learningRate = 0.01; self.regularization = 0.001
        self.numLatentFactors = 5; self.lossThreshold = 0.1
        # Set test parameters
        self.Ui = np.arange(0, self.numBioFeatures, 1)
        self.timePoint = 0; self.userInd = 0; self.itemInd = 0; self.userItemRating = 50
        
        #
        self.userNames = [str(i) for i in range(self.numUsers)]
        self.itemNames = [str(i) for i in range(self.numItems)]
        
        # Compile model inputs
        self.Training_Data = np.asarray([self.Ui, self.Ui/2, self.Ui, self.Ui*2, self.Ui, self.Ui*2, self.Ui/2, self.Ui*4])
        self.Training_Labels = np.asarray([[self.timePoint, self.userInd+1, self.itemInd, self.userItemRating], 
                                    [self.timePoint+10, self.userInd+1, self.itemInd+1, self.userItemRating/2],
                                    [self.timePoint+20, self.userInd+1, self.itemInd, self.userItemRating],
                                    [self.timePoint+30, self.userInd+1, self.itemInd+1, self.userItemRating*2],
                                    [self.timePoint, self.userInd, self.itemInd+1, self.userItemRating],
                                    [self.timePoint+10, self.userInd, self.itemInd+1, self.userItemRating/2],
                                    [self.timePoint+20, self.userInd, self.itemInd, self.userItemRating],
                                    [self.timePoint+30, self.userInd, self.itemInd, self.userItemRating*2]
                                    ], dtype=int)
        self.Testing_Data =  np.asarray([self.Ui, self.Ui*2])
        self.Testing_Labels = np.asarray([[self.timePoint, self.userInd, self.itemInd+1, self.userItemRating], [self.timePoint+10, self.userInd, self.itemInd, self.userItemRating*2]])
        self.featureNames = np.asarray([str(elem) for elem in self.Ui])
            
    def resetModel(self):
        # Set random seed
        np.random.seed(1234)
        # Instantiate model class.
        self.recommendationModel = recommendationModel.recommendationModel(self.modelPath, self.modelType, self.allFeatureNames, self.overwriteModel, self.numUsers, self.numBioFeatures, self.numItems)
        # Set hyperparameters
        self.recommendationModel.maxTimeRemember = 10
        self.recommendationModel.setOptimizedParameters(self.learningRate, self.regularization, self.numLatentFactors, self.lossThreshold)
        # Remove randomness from the model
        self.recommendationModel.B = np.random.uniform(-0.5, 0.5, (self.recommendationModel.numBioFeatures, self.recommendationModel.numLatentFactors)) 
        self.recommendationModel.V = np.random.uniform(-0.5, 0.5, (self.recommendationModel.numLatentFactors, self.recommendationModel.numItems))
        self.recommendationModel.W = np.random.uniform(-0.5, 0.5, (self.recommendationModel.numItems, self.recommendationModel.numItems)) 
        self.recommendationModel.W /= np.sum(self.recommendationModel.W, axis=0)
        # Add non-trivial variables to the model
        self.recommendationModel.mu = 1
        self.recommendationModel.alpha = 0.5
        self.recommendationModel.userBias += 2
        self.recommendationModel.itemBias += 3
        self.recommendationModel.bioBaseline += 4
        # Clear ratings
        self.recommendationModel.clearPastRatings()
    
    def testPredictions(self):  
        self.resetModel()
        
        # Assert collaborative filtering is working.
        ratingMF = self.recommendationModel.collabFiltRating(self.Ui, self.userInd, self.itemInd)
        assert ratingMF == 4.329176948656178, "ratingMF = " + str(ratingMF) 
        
        # Assert content based recommendation is working when past ratings is zero.
        ratingCB_noPast = self.recommendationModel.contentBasedRating(self.Ui, self.timePoint, self.userInd, self.itemInd)
        assert ratingCB_noPast == ratingMF, "ratingCB_noPast = " + str(ratingCB_noPast)
        
        # Check input/output parameters when adding a rating.
        self.assertPastRatings(self.userInd, 0, -1, -1)
        self.recommendationModel.addNewRating(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        self.assertPastRatings(self.userInd, 1, self.timePoint, self.userItemRating)

        # Add another raing to bring the total to 2 past ratings.
        newTime = self.timePoint + 4; newUi = self.Ui*2; newRating = self.userItemRating*2
        self.recommendationModel.addNewRating(newUi, newTime, self.userInd, self.itemInd, newRating)
        
        # Assert content based recommendation is working when 0 decay rate.
        self.recommendationModel.decayRates[self.userInd] = 0
        ratingCB_lambda0 = self.recommendationModel.contentBasedRating(newUi, newTime, self.userInd, self.itemInd)
        assert ratingCB_lambda0 == (newRating + self.userItemRating)/2, "ratingCB_lambda0 = " + str(ratingCB_lambda0)
        # Assert content based recommendation is working for positive decay rate.
        self.recommendationModel.decayRates[self.userInd] = .01
        ratingCB_lambda01 = self.recommendationModel.contentBasedRating(self.Ui, self.timePoint, self.userInd, self.itemInd)
        assert ratingCB_lambda01 == 75.49993334399828, "ratingCB_lambda01 = " + str(ratingCB_lambda01)
        # Assert content based recommendation is working for negative decay rate.
        self.recommendationModel.decayRates[self.userInd] = -.01
        ratingCB_negLambda01 = self.recommendationModel.contentBasedRating(self.Ui, self.timePoint, self.userInd, self.itemInd)
        assert ratingCB_negLambda01 == 74.50006665600174, "ratingCB_negLambda01 = " + str(ratingCB_negLambda01)
        # Assert content based recommendation is working when not all items rated for all times
        self.recommendationModel.decayRates[self.userInd] = 0
        self.recommendationModel.addNewRating(newUi, newTime + 4, self.userInd, self.itemInd+1, newRating)
        ratingCB_UnratedItems = self.recommendationModel.contentBasedRating(self.Ui, self.timePoint, self.userInd, self.itemInd)
        assert ratingCB_UnratedItems == 66.88383861883388, "ratingCB_UnratedItems = " + str(ratingCB_UnratedItems)

        # Assert prediction is good
        self.recommendationModel.decayRates[self.userInd] = 0
        predict1 = self.recommendationModel.predictPoint(self.Ui, self.timePoint, self.userInd, self.itemInd)
        assert predict1 == ratingMF*self.recommendationModel.alpha + (1-self.recommendationModel.alpha)* ratingCB_UnratedItems, str(predict1)
        # Assert prediction is good
        ratingMF2 = self.recommendationModel.collabFiltRating(newUi, self.userInd, self.itemInd)
        predict2 = self.recommendationModel.predictPoint(newUi, newTime, self.userInd, self.itemInd)
        assert predict2 == ratingMF2*self.recommendationModel.alpha + (1-self.recommendationModel.alpha)* ratingCB_UnratedItems

    def testPastRatings(self):
        self.resetModel()
        
        # Check input/output parameters when adding a rating.
        self.assertPastRatings(self.userInd, 0, -1, -1)
        self.recommendationModel.addNewRating(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        self.assertPastRatings(self.userInd, 1, self.timePoint, self.userItemRating)

        # Add a rating to a different user and reset the old user ratings.
        self.recommendationModel.addNewRating(self.Ui, self.timePoint, self.userInd+1, self.itemInd, self.userItemRating)
        self.recommendationModel.resetPastRatings(self.userInd)
        # Check input/output parameters for the new ratings.
        self.assertPastRatings(self.userInd, 0, -1, -1)
        self.assertPastRatings(self.userInd+1, 1, self.timePoint, self.userItemRating)
        # Have two similar ratings. Cannot have two ratings at the same timePoint.
        self.recommendationModel.addNewRating(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        self.recommendationModel.addNewRating(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        self.assertPastRatings(self.userInd, 1, self.timePoint, self.userItemRating)
        # Clear all ratings.
        self.recommendationModel.clearPastRatings()
        self.assertPastRatings(self.userInd, 0, -1, -1)
        self.assertPastRatings(self.userInd+1, 0, -1, -1)
        
        # Add a rating and try andupdate the rating matrix with an out of bounds rating time.
        self.recommendationModel.addNewRating(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        assert None == self.recommendationModel.updatePastRatingMatrix(self.timePoint - 1, self.userInd)
        self.assertPastRatings(self.userInd, 0, -1, -1)
        # Try and update the rating matrix when no rating has been added
        self.recommendationModel.addNewRating(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        assert None == self.recommendationModel.updatePastRatingMatrix(self.timePoint, self.userInd+1)
        
        # Try and update the rating matrix with a time the makes the previous rating out of bounds
        self.recommendationModel.updatePastRatingMatrix(self.timePoint + self.recommendationModel.maxTimeRemember + 1, self.userInd)
        self.assertPastRatings(self.userInd, 0, -1, -1)
        self.recommendationModel.clearPastRatings()
        
        # Try and update the rating matrix with a time the makes the previous rating out of bounds, but keep one rating
        self.recommendationModel.addNewRating(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        self.recommendationModel.addNewRating(self.Ui, self.timePoint + 1, self.userInd, self.itemInd, self.userItemRating)
        self.assertPastRatings(self.userInd, 2, self.timePoint + 1, self.userItemRating)
        # Try and update the rating matrix with a time the makes the previous rating out of bounds, but keep one rating
        self.recommendationModel.updatePastRatingMatrix(self.timePoint + self.recommendationModel.maxTimeRemember + 1, self.userInd)
        self.assertPastRatings(self.userInd, 1, self.timePoint+1, self.userItemRating)
        
        self.recommendationModel.clearPastRatings()
        
    def assertPastRatings(self, userInd, numPastRatings, timePoint, userItemRating):
        assert self.recommendationModel.numPastRatings[userInd] == numPastRatings, str(self.recommendationModel.numPastRatings[userInd])
        assert self.recommendationModel.pastRatingTimes[userInd][max(0, numPastRatings-1)] == timePoint, "pastTime = " + str(self.recommendationModel.pastRatingTimes[userInd][numPastRatings-1]) + "; Given: " + str(timePoint)
        assert self.recommendationModel.pastUserItemRatings[userInd][max(0, numPastRatings-1)][self.itemInd] == userItemRating
        
    def testMatrices(self):  
        self.resetModel()
                
        # Test updating B matrix.
        self.recommendationModel.addGradientBioMatrix(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        assert (self.recommendationModel.B == [[-0.05869506183413528, -0.3134395406930976, -0.07878307017511549, -0.04222360243888795, -0.2052304381272259], \
                                             [0.47085122888182773, -0.16465450742696924, -0.07016956767542293, -0.2797462338303399, 0.1322608077872402], \
                                             [-0.3689030413124663,  0.10810826935750589,  -0.5308398055062051,  -0.6525876771679328,  -0.6123200847837339], \
                                             [ -0.4086157455364502,  -0.07155037121852564,  -0.18300949310877979,  0.3940037522041861,  -0.4770890315397906]]).all(), "B = " + str(self.recommendationModel.B)
        
        self.resetModel()
        # Test updating V matrix.
        self.recommendationModel.addGradientItemMatrix(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        assert (self.recommendationModel.V[:, self.itemInd] == [0.09025033074152428, 0.1497156925671616, 0.18675239499596435, 0.17718249676599335, 0.2737470260315666]).all(), str(self.recommendationModel.V[:, self.itemInd])
        
        self.resetModel()
        # Test updating V matrix.
        self.recommendationModel.addGradientBioBaselineMatrix(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        assert (self.recommendationModel.bioBaseline[self.userInd] == [3.912223996292333, 3.9509576926190717, 4.073142895502328, 3.970005826328646]).all(), "Bb = " + str(self.recommendationModel.bioBaseline[self.userInd])
    
        self.resetModel()
        # Test updating user bias matrix.
        self.recommendationModel.addGradientUserBiasVector(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        assert self.recommendationModel.userBias[self.userInd] == 2.228334115256719, str(self.recommendationModel.userBias[self.userInd])
    
        self.resetModel()
        # Test updating item bias matrix.
        self.recommendationModel.addGradientItemBiasVector(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        assert self.recommendationModel.itemBias[self.itemInd] == 3.228324115256719, str(self.recommendationModel.itemBias[self.itemInd])
    
        self.resetModel()
        # Test updating average rating.
        self.recommendationModel.addGradientMu(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        assert self.recommendationModel.mu == 1.2283441152567192, str(self.recommendationModel.mu)
    
        self.resetModel()
        # Test updating alpha weight.
        initialAlpha = self.recommendationModel.alpha
        self.recommendationModel.addGradientAlpha(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        assert self.recommendationModel.alpha == initialAlpha, str(self.recommendationModel.alpha)
        # Test updating alpha weight when rating is added
        self.recommendationModel.addNewRating(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        self.recommendationModel.addGradientAlpha(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        assert self.recommendationModel.alpha == 0, str(self.recommendationModel.alpha)
        
        self.resetModel()
        # Test updating decay rates when no past ratings are recorded.
        self.recommendationModel.addGradientDecayRate(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        assert (self.recommendationModel.decayRates == 0).all(), str(self.recommendationModel.decayRates)
        # Test updating decay rates when one rating exists and decay rate starts as zero.
        self.recommendationModel.addNewRating(self.Ui, self.timePoint, self.userInd, self.itemInd, self.userItemRating)
        self.recommendationModel.addGradientDecayRate(self.Ui, self.timePoint + 1, self.userInd, self.itemInd, self.userItemRating)
        assert (self.recommendationModel.decayRates == 0).all(), str(self.recommendationModel.decayRates)
        # Test updating decay rates when two rating exist for the same user and decay rate starts at zero.
        self.recommendationModel.addNewRating(self.Ui, self.timePoint + 1, self.userInd, self.itemInd, self.userItemRating*2)
        self.recommendationModel.addGradientDecayRate(self.Ui, self.timePoint + 2, self.userInd, self.itemInd, self.userItemRating*2)
        assert (self.recommendationModel.decayRates == [-0.7541926440708988, 0]).all(), str(self.recommendationModel.decayRates)
        # Test updating decay rates when multiple rating exist for user and items.
        self.recommendationModel.addNewRating(self.Ui, self.timePoint + 2, self.userInd, self.itemInd+1, self.userItemRating)
        self.recommendationModel.addGradientDecayRate(self.Ui, self.timePoint + 2, self.userInd, self.itemInd, self.userItemRating)
        assert (self.recommendationModel.decayRates == [-0.7568988638922951, 0]).all(), str(self.recommendationModel.decayRates)

        
    def testTraining(self):
        self.resetModel()
        
        # Score the model with no training should return None
        modelError = self.recommendationModel.scoreModel(self.Training_Data, self.Training_Labels)
        assert modelError == None, modelError
        
        # Train the model on the data
        testingError = self.recommendationModel.trainModel(self.Training_Data, self.Training_Labels, self.Testing_Data, self.Testing_Labels, self.featureNames, self.userNames, self.itemNames)
        assert (self.recommendationModel.finalFeatureNames == self.featureNames).all() 
        
        # Score the trained model
        testingError_2 = self.recommendationModel.scoreModel(self.Testing_Data, self.Testing_Labels)
        trainingError = self.recommendationModel.scoreModel(self.Training_Data, self.Training_Labels)
        # Assert training and testing error are as it seems
        assert testingError == testingError_2, str(testingError) + " " + str(testingError_2)
        
# ---------------------------------------------------------------------------#

if __name__ == "__main__":
    # Instantiate unit test class.
    unitTesterCollabFilt = testCollabFilt()
    
    # Test machine learning models
    # unitTesterCollabFilt.testMatrices()
    # unitTesterCollabFilt.testPastRatings()
    # unitTesterCollabFilt.testPredictions()
    unitTesterCollabFilt.testTraining()
    
    
    
    
    