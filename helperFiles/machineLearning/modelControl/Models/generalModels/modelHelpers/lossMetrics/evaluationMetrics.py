
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic modules
import numpy as np

# -------------------------------------------------------------------------- #
# --------------------------- Evaluation Metrics --------------------------- #

class evaluationMetrics():        
    
    def __init__(self): 
        self.missingStateValue = np.nan
        
    def _setBoundaryValues(self, minStateValue, maxStateValue):
        self.minStateValue, self.maxStateValue = minStateValue, maxStateValue
        
    def _setMissingStateValue(self, missingStateValue):
        self.missingStateValue = missingStateValue
        
    # ---------------------------------------------------------------------- #
    # -------------------------- Metric Interface -------------------------- #
    
    def evaluateMetrics(self, metricTypes, trueVals, predictedVals, normalizeData = False, numParams = 0):
        # Setup variables as numpy arrays
        trueVals = np.asarray(trueVals.copy())
        predictedVals = np.asarray(predictedVals.copy())
        # Setup variables.
        if isinstance(metricTypes, str): metricTypes = [metricTypes]
        
        # If normalizing
        if normalizeData:
            # Calculate the length of the vectors.
            trueValueNorm = np.linalg.norm(trueVals)
            predictedValueNorm = np.linalg.norm(predictedVals)            
            
            # Normalize the vectors.
            if trueValueNorm != 0:
                trueVals /= trueValueNorm
            if predictedValueNorm != 0:
                predictedVals /= predictedValueNorm
                
        finalStateValues = np.zeros(len(metricTypes))
        # For each evaluation metric of the equation.
        for metricInd in range(len(metricTypes)):
            metricType = metricTypes[metricInd]
            
            # R2 calculation.
            if metricType == "R2":
                stateValue = self.calculateR2(trueVals, predictedVals, numParams)
            # Cos(theta) calculation.
            elif metricType == "cosSimilarity":
                stateValue = self.calculateCosineSimilarity(trueVals, predictedVals)
            # Pearson correlation calculation.
            elif metricType == "pearson":
                stateValue = self.calculatePearsonCorrelation(trueVals, predictedVals)
            # Spearman correlation calculation.
            elif metricType == "spearman":
                stateValue = self.calculateSpearmanCorrelation(trueVals, predictedVals)
            # Unknown request.
            else: assert False, "Unknown Metric: " + str(metricType)
            
            # Save the state value.
            finalStateValues[metricInd] = self.boundStateValue(stateValue)
        
        if len(finalStateValues) == 1:
            return finalStateValues[0]
        return finalStateValues
    
    def boundStateValue(self, stateValue):
        # Don't alter missing values.
        if stateValue == None or np.isnan(stateValue):
            return stateValue
        
        # Return a bounded value.
        return max(self.minStateValue, min(self.maxStateValue, stateValue))
    
    # ---------------------------------------------------------------------- #
    # --------------------------- Metric Methods --------------------------- #
    
    def calculateR2(self, trueVals, predictedVals, numParams = 0):
        # Calculate the variance of the dataset
        unexplainedVariance =  ((trueVals - predictedVals)**2).sum(axis = 0)
        totalVariance = ((trueVals - trueVals.mean(axis = 0))**2).sum(axis = 0)
        
        # Special case: the totalVariance is zero, the trueOutput are constant.
        if totalVariance == 0:
            # If the model worsened the prediction (the average), then R2 = 0.
            if unexplainedVariance != 0:
                return 0
            # Else, the model fully explains the data by keeping the average.
            return 1

        # If considering the degrees of freedom lost from extra features
        if numParams != 0:
            # Calculate the adjusted R2
            R2 = 1 - unexplainedVariance * (len(trueVals) - 1) / (totalVariance * (len(trueVals) - numParams - 1))
        else:
            # Calculate the R2 score.
            R2 = 1 - unexplainedVariance / totalVariance
            
        # Enforce a value between -1 and 1.
        return R2
        
    def calculateCosineSimilarity(self, trueVals, predictedVals):        
        # Calculate the length of the predicted values.
        predictedValueNorm = np.linalg.norm(predictedVals)
        trueValueNorm = np.linalg.norm(trueVals)
        
        # If there is no vector, there is no score.
        if predictedValueNorm == 0 or trueValueNorm == 0:
            return self.missingStateValue
        
        # Return the cosine similarity
        cosTheta = (trueVals*predictedVals).sum()/(predictedValueNorm * trueValueNorm)
        
        # Enforce a value between -1 and 1.
        return cosTheta
    
    def calculatePearsonCorrelation(self, trueVals, predictedVals):
        # Calculate the difference between the means of both points.
        predictedMeanDiff = predictedVals - predictedVals.mean()
        trueMeanDiff = trueVals - trueVals.mean()
                
        # Calculate the pearson correlation terms.
        covariance = (predictedMeanDiff * trueMeanDiff).sum()
        predictedSTD = np.linalg.norm(predictedMeanDiff)
        trueSTD = np.linalg.norm(trueMeanDiff)
        
        # If all true values are the same, there is no score.
        if trueSTD == 0: return self.missingStateValue
        # If all predicted values are the same, there is no score.
        if predictedSTD == 0: return self.missingStateValue
        
        # Calculate the pearson correlation.
        pearsonCorr =  covariance / (predictedSTD * trueSTD)
    
        # Enforce a value between -1 and 1.
        return 2*abs(pearsonCorr) - 1
    
    def calculateSpearmanCorrelation(self, trueVals, predictedVals):
        # If all true values are the same, there is no score.
        if (trueVals == trueVals[0]).all(): return self.missingStateValue
        # If all predicted values are the same, there is no score.
        if (predictedVals == predictedVals[0]).all(): return self.missingStateValue
        
        # Calculate the ranks of the true and predicted values
        trueRanks = self.rankdata(trueVals)
        predictedRanks = self.rankdata(predictedVals)
        
        # Calculate rank information.
        rankDiffs = trueRanks - predictedRanks # Calculate the differences between ranks
        sumSquaredDiffs = np.sum(rankDiffs ** 2) # Calculate the sum of squared rank differences
        
        # Calculate the spearman correlation
        n = len(trueVals) # Calculate the number of observations
        spearmanCorr = 1 - (6 * sumSquaredDiffs) / (n * (n ** 2 - 1))
        
        # Enforce a value between -1 and 1.
        return 2*abs(spearmanCorr) - 1
    
    def rankdata(self, arr):
        """ Assigns ranks to data, dealing with ties appropriately. """
        sorted_indices = np.argsort(arr)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(arr)) + 1
        return ranks

# ---------------------------------------------------------------------------#

if __name__ == "__main__":
    # Set model paramters
    modelPath = "./EGM.pkl"
    modelType = "EGM"
    allFeatureNames = [""]
    overwriteModel = True

        


