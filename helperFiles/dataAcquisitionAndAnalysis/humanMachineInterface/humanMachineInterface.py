
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General Modules
import os

# -------------------------------------------------------------------------- #
# ---------------------------- Global Function ----------------------------- #

class humanMachineInterface():
    
    def __init__(self, modelClasses, actionControl):
        # General parameters.
        self.modelClasses = modelClasses        # A list of machine learning models.
        self.actionControl = actionControl
                
        # Initialize mutable variables.
        self.resetVariables_HMI()
        
    def resetVariables_HMI(self):        
        # Aligned feature data structure
        self.alignedFeatures = []       # Linearly interpolated features to align all at the same timepoint.
        self.alignedFeatureTimes = []   # The interpolated timepoints of the ALIGNED feature.
        self.alignedFeatureLabels = [[] for _ in range(len(self.modelClasses))]  # The FINAL predicted labels at the current timepoint.
        
        # Subject information
        self.userName = None
        self.alignedUserNames = []
        self.alignedItemNames = []
        
    def setUserName(self, filePath):
        # Get user information
        fileName = os.path.basename(filePath).split(".")[0]
        self.userName = fileName.split(" ")[-1].lower()
    
    def predictLabels(self): 
        # Find the new final features, where no label has been predicted yet 
        allNewFeatures = self.alignedFeatures[len(self.alignedFeatureLabels[0]):].copy()
        allNewFeaturesTimes = self.alignedFeatureTimes[len(self.alignedFeatureLabels[0]):].copy()
        # Find new subject information
        newUserNames = self.alignedUserNames[len(self.alignedFeatureLabels[0]):].copy()
        newItemNames = self.alignedItemNames[len(self.alignedFeatureLabels[0]):].copy()
        
        # For each prediction model
        for modelInd in range(len(self.modelClasses)):
            # If the model was never trained, dont use it.
            if len(self.modelClasses[modelInd].finalFeatureNames) == 0:
                continue
            
            modelClass = self.modelClasses[modelInd]  
            # Standardize the incoming features (if the model requires)
            standardizedFeatures = modelClass.standardizeClass_Features.standardize(allNewFeatures) if modelClass.standardizeClass_Features else allNewFeatures
            # Select the model features
            newFinalFeatures = modelClass.getSpecificFeatures(modelClass.allFeatureNames, modelClass.finalFeatureNames, standardizedFeatures)
            
            # Predict the final labels
            if modelClass.modelType == "MF":
                standardizedPredictions = modelClass.model.predict(newFinalFeatures, allNewFeaturesTimes, newUserNames, newItemNames)
            else:
                standardizedPredictions = modelClass.predict(newFinalFeatures)
            # Rescale up the labels (if the model requires)
            predictedLabels = modelClass.standardizeClass_Labels.unStandardize(standardizedPredictions) if modelClass.standardizeClass_Labels else standardizedPredictions
            
            # Save the final results
            self.alignedFeatureLabels[modelInd].extend(predictedLabels[-len(predictedLabels):])
            
            
            