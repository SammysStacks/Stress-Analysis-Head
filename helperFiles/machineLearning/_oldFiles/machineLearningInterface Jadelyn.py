#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:47:49 2021

@author: samuelsolomon
"""

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import os
import sys
import time
import bisect
import itertools
import numpy as np
from scipy import stats
# Modules for Plotting
import matplotlib.pyplot as plt
# Neural Network Modules
from sklearn.model_selection import train_test_split

# Import interfaces for reading/writing data
sys.path.append(os.path.dirname(__file__) + "/../Data Aquisition and Analysis/Excel Processing/")
import saveDataProtocols     # Functions to Save/Read in Data from Excel

# Import Files for extracting models
sys.path.append(os.path.dirname(__file__) + "/Model Control/")
import _modelControl

# Import files.
from _dataPreparation import standardizeData

# -------------------------------------------------------------------------- #
# ------------------ Interface with Machine Learning Files ----------------- #

class machineLearningHead:
    
    def __init__(self, modelTypes, modelFile, allFeatureNames, trainingFolder):
        # Store constant parameters.
        self.modelFile = modelFile
        self.allFeatureNames = allFeatureNames
        # Store variable parameters.
        self.modelTypes = modelTypes
        
        self.saveDataFolder = None
        # Create data folder to save the analysis.
        if trainingFolder != None:
            self.saveDataFolder = trainingFolder + "dataAnalysis/machineLearning/"
            os.makedirs(self.saveDataFolder, exist_ok=True)
        
        # Initialize classes.
        self.saveDataInterface = saveDataProtocols.saveExcelData()
        self.modelControl = _modelControl.modelControl(self.modelFile, self.modelTypes, self.allFeatureNames, self.saveDataFolder) # Create a controller for all models.
        
    def createModels(self, modelTypes):
        self.modelTypes = modelTypes
        self.modelControl.createModels(modelTypes)
        
    def resetModels(self, modelTypes):
        self.modelControl.createModels(modelTypes)
        # self.modelControl.resetModels()
        
    def averageModelAccuracy(self, modelInd, featureData, featureLabels, featureNames, numEpochs = 200, stratifyBy = None, testSplitRatio = 0.2, categorical = [], imbalancedData = False):
        modelPerformances = []
        
        # If this is a categorical problem
        if len(categorical) != 0:
            # Keep track of each category
            labels = list(categorical.keys())
            # Assert that all the labels in the data (featureLabels) are in labels
            for elem in np.unique(featureLabels):
                assert elem in labels, f"Label {elem} is in not a category {labels}"
            modelPerformances_byLabel = [[] for _ in labels]
        
        
        # For each trial
        for _ in range(numEpochs):
            # Randomly split the data
            Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(featureData, featureLabels, test_size=testSplitRatio, shuffle= True)

            # Train the model and save the accuracy
            modelPerformance = self.modelControl.modelClasses[modelInd].trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels, featureNames, imbalancedData = imbalancedData)
            modelPerformances.append(modelPerformance)
            
            # If this is a categorical problem
            if len(categorical) != 0:
                # For each label
                for i in range(len(labels)):
                    # Save the model's performance on the data with this label only
                    modelPerformances_byLabel[i].append(self.modelControl.modelClasses[modelInd].scoreModel(featureData[featureLabels == labels[i]], featureLabels[featureLabels == labels[i]]))
            
        # Return the average classification accuracy
        averagePerformance = stats.trim_mean(modelPerformances, 0.2)
        
        # If categorical problem, get the average classification accuracy for each label
        if len(categorical) != 0:
            averagePerformance_byLabel = [stats.trim_mean(modelPerformances_byLabel[i], 0.2) for i in range(len(labels))]
        else: # Otherwise return an empty array
            averagePerformance_byLabel = []
        stdPerformance = np.std(modelPerformances)
        
        # This code plots the distribution of model performances across the epochs. We want to make sure that this distribution is normal, so the mean accurately represents the model's performance
        # plt.hist(modelPerformances, bins=50)
        # plt.axvline(averagePerformance, color='red', linestyle='dashed')
        # plt.show()
        # print("DONE", len(modelPerformances), "Epochs")
        return averagePerformance, stdPerformance, averagePerformance_byLabel

    def getMedianScore(self, scores, plotScores = False):
        # Plot the data distribution
        if plotScores:
            plt.hist(scores, 100, facecolor='blue', alpha=0.5)
            
        # Fit the skewed distribution and take the median
        ae, loce, scalee = stats.skewnorm.fit(scores)
        medianScore = np.round(loce*100, 2)
        
        return medianScore
            
    def findWorstSubject(self, modelInd, featureData, featureLabels, featureNames, subjectOrder, featureNamesCombination_String):
        subjectOrder = np.array(subjectOrder)
        featureLabels = np.array(featureLabels)
        
        # Get All Possible itertools.combinations
        if False:
            subjectCombinationInds = itertools.combinations(range(0, len(featureLabels)),  len(featureLabels) - 2)
            allSubjectLabels = set(range(0, len(featureLabels)))
        else:
            uniqueSubjects = np.unique(subjectOrder)
            subjectCombinationInds = itertools.combinations(range(0, len(uniqueSubjects)),  len(uniqueSubjects) - 3)
            allSubjectLabels = set(range(0, len(featureLabels)))
            
        removedInds = []
        removedSubjects = []
        finalPerformances = []
        for subjectInds in subjectCombinationInds:
            subjectInds = list(subjectInds)
            # Reset the Input Variables.
            self.resetModels(self.modelTypes) # Reset the ML Model
            
            newInds = []
            for pointInd in range(len(subjectOrder)):
                if subjectOrder[pointInd] in uniqueSubjects[subjectInds]:
                    newInds.append(pointInd)
            subjectInds = newInds                
            
            # Remove the subject features.
            culledSubjectData =  featureData[subjectInds, :]
            culledSubjectLabels = featureLabels[subjectInds]

            # Score the model with this data set.
            modelPerformance = self.averageModelAccuracy(modelInd, culledSubjectData, culledSubjectLabels, featureNames, numEpochs = 1, stratifyBy = None, testSplitRatio = 0.3)

            # Save which subjects were removed.
            discardedSubjectInds = np.array(list(allSubjectLabels.difference(set(subjectInds))))
            removedSubject = np.unique(subjectOrder[discardedSubjectInds])
        
            insertionPoint = bisect.bisect(finalPerformances, -modelPerformance, key=lambda x: -x)
            # Save the model score and standard deviation.
            removedInds.insert(insertionPoint, discardedSubjectInds)
            removedSubjects.insert(insertionPoint, removedSubject)
            finalPerformances.insert(insertionPoint, modelPerformance)
            
        print(featureNamesCombination_String)
        print("Worst Subject:", removedSubjects[0], finalPerformances[0])
        print("Best Subject:", removedSubjects[-1], finalPerformances[-1])
        print("")
                    
    def analyzeFeatureCombinations(self, modelInd, featureData, featureLabels, featureNames, numFeaturesCombine, subjectOrder, numEpochs = 10, saveData = True, 
                                   saveExcelName = "Feature Combination Accuracy.xlsx", printUpdateAfterTrial = 2000, scaleLabels = False, categorical = {}, imbalancedData = False):
        """
        NOTES:
            - I start with a minimum threshold of 0.7. This means that I cull out any feature that,
                individually, produces a performance score less than 0.7, and save the rest.
            - Feature combinations are considered based on how many features the combination consists of.
            - After each group of combinations, I basically stop considering any combinations that
                include a feature that was not in any group that made it past the minimum threshold.
                (I explain this more towards the end of the function)
                    - Ex. Say you have combinations (openingTime_Peak_EOG closingTime_Peak_EOG), (blinkDuration_EOG closingTime_Peak_EOG),
                        and (blinkDuration_EOG, tentDeviationX_EOG), but only (openingTime_Peak_EOG closingTime_Peak_EOG) and (blinkDuration_EOG closingTime_Peak_EOG)
                        perform higher than the threshold. We create the next group, combinations of three features, only using features
                        that made it past the threshold in at least one combination: Thus, we can only create (openingTime_Peak_EOG closingTime_Peak_EOG blinkDuration_EOG).
                    - The minimum threshold is re-set after these new combinations are made. I currently
                        set it to be the performance of the best feature combination, meaning that a new
                        combination with n features must be better than the best combination I had with
                        n-1 features. I technically take three standard deviations lower than the mean,
                        because there is variation in performance and that value corresponds to the lowest
                        the score would reasonably actually be. Not sure if this is being too generous.
        """
        # Format the incoming data.
        featureLabels = np.asarray(featureLabels.copy())
        featureData = np.asarray(featureData.copy())
        featureNames = np.asarray(featureNames)
        
        # Function variables
        numModelsTrack = 200   # The number of models to track
        
        # We initialize a bottom threshold for the performance a feature must have in order to consider it for the next combinations
        thresholdToKeep = 0.7
        
        subjectCombinationInds = list(itertools.combinations(range(0, len(featureLabels)),  len(featureLabels) - int(len(featureLabels)*0)))
        featureCombinationInds = list(itertools.combinations(range(0, len(featureNames)), 1))
        # Find total combinations
        numFeatureCombnatons = len(featureCombinationInds)
        # Standardize the features
        standardizeClass_Features = standardizeData(featureData)
        standardizedFeatures = standardizeClass_Features.standardize(featureData)
        
        if scaleLabels:
            standardizeClass_Labels = standardizeData(featureLabels)
            featureLabels = standardizeClass_Labels.standardize(featureLabels)
        
        # For combinations of n features, up to numFeaturesCombine number of features, we analyze the performance of the combinations
        for curr_numFeaturesCombine in numFeaturesCombine:
            print(f"\nAnalyze {len(list(featureCombinationInds))} combinations for {curr_numFeaturesCombine} number of features.")
            # Get All Possible itertools.combinations
            # Store performance of each combination
            self.finalPerformances = []; self.finalPerformancesSTDs = []; self.featureNamesCombinations = [];
            
            # Also store performance against each label
            self.finalPerformances_byLabel = []
            
            t1 = time.time(); combinationRoundInd = -1
            # For Each Combination of Features, we get the data of the specific features and compute the performance of the combination
            for combinationInds in featureCombinationInds:
                combinationInds = list(combinationInds)
                combinationRoundInd += 1
                
                # Collect the Signal Data for the Specific Features
                featureData_culledFeatures = standardizedFeatures[:, combinationInds]
                # Collect the Specific Feature Names
                featureNamesCombination_String = ''; featureNamesCull = []
                for ind in combinationInds:
                    name = featureNames[ind]
                    featureNamesCombination_String += name + ' '
                    featureNamesCull.append(name)
                                
                modelPerformances = []; modelSTDs = []; modelPerformances_byLabel = []
                # Get the model performance of this combination
                # For each subject/set of data. We typically have 1 subject
                for subjectInds in subjectCombinationInds:
                    subjectInds = list(subjectInds)
                    # Reset the Input Variables
                    self.resetModels(self.modelTypes) # Reset the ML Model
                                    
                    # Collect the Signal Data for the Specific Subjects
                    featureData_finalCull = featureData_culledFeatures[subjectInds, :]
                    featureLabels_finalCull = featureLabels[subjectInds]
    
                    # Score the model with this data set.
                    modelPerformance, modelSTD, modelPerformace_byLabel = self.averageModelAccuracy(modelInd, featureData_finalCull, featureLabels_finalCull, featureNamesCull, numEpochs = 100, stratifyBy = [], testSplitRatio = 0.2, categorical = categorical, imbalancedData = imbalancedData)
                    modelPerformances.append(modelPerformance)
                    modelSTDs.append(modelSTD)
                    
                    modelPerformances_byLabel.append(modelPerformace_byLabel)
                    
                modelPerformance = stats.trim_mean(modelPerformances, 0.3)
                
                # If the model's performance is one of the top scores
                if len(self.finalPerformances) < numModelsTrack or modelPerformance > self.finalPerformances[-1]:
                    modelSTD = 0 if len(modelSTDs) <= 0 else np.mean(modelSTDs)
                    
                    insertionPoint = bisect.bisect(self.finalPerformances, -modelPerformance, key=lambda x: -x)
                    # Save the model score and standard deviation
                    self.featureNamesCombinations.insert(insertionPoint, featureNamesCombination_String[0:-1])
                    self.finalPerformances.insert(insertionPoint, modelPerformance)
                    self.finalPerformancesSTDs.insert(insertionPoint, modelSTD)
                    
                    if len(categorical) != 0:
                        self.finalPerformances_byLabel.insert(insertionPoint, [stats.trim_mean(np.array(modelPerformances_byLabel)[:, i], 0.3) for i in range(len(np.unique(featureLabels)))])
                    
    
                    # Only track the best models
                    if len(self.finalPerformances) > numModelsTrack:
                        self.finalPerformances.pop()
                        self.finalPerformancesSTDs.pop()
                        self.featureNamesCombinations.pop()
                        if len(categorical) != 0:
                            self.finalPerformances_byLabel.pop()
                    
                # Report an Update Every Now and Then
                if (combinationRoundInd % printUpdateAfterTrial == 0 and combinationRoundInd != 0) or combinationRoundInd == 20:
                    t2 = time.time()
                    percentComplete = 100*combinationRoundInd/numFeatureCombnatons
                    setionPercent = 100*min(combinationRoundInd or 1, printUpdateAfterTrial)/numFeatureCombnatons
                    print(str(np.round(percentComplete, 2)) + "% Complete; Estimated Time Remaining: " + str(np.round((t2-t1)*(100-percentComplete)/(setionPercent*60), 2)) + " Minutes")
                    t1 = time.time()
                        
            print(self.finalPerformances[0], self.finalPerformancesSTDs[0], self.featureNamesCombinations[0])
            # Save the Data in Excel
            if saveData:
                headers = ["Mean Score", "STD", "Feature Combination"]
                data = np.dstack((self.finalPerformances, self.finalPerformancesSTDs, 
                                                self.featureNamesCombinations))
                # If this is a categorical problem
                if len(categorical) != 0:
                    # Add the category titles to the header row of the excel
                    for category in categorical.values():
                        headers.append(category)
                        
                    # Add the individual labels' performance scores to the excel
                    finalPerformances_byLabel_Stack = np.dstack([np.array(self.finalPerformances_byLabel)[:, i] for i in range(len(np.unique(featureLabels)))])
                    data = np.concatenate((data, finalPerformances_byLabel_Stack), axis=2)
                    
                # Save to excel. Formatted in order of best performance by the mean score.
                self.saveDataInterface.saveFeatureComparison(data[0], [], headers, self.saveDataFolder, saveExcelName, sheetName = str(numFeaturesCombine) + " Features in Combination", saveFirstSheet = True)
            
            # ----------- Setup next set of combinations for n + 1 ---------- #
            # --------- features, based on performance on n features -------- #
            
            # We find all of the features that were included in a combination that scored higher than the threshold
            # We make this the features we use to make the next set of combinations we search through
            
            # Find all combinations that scored higher than the threshold
            goodCombinations = np.array(self.featureNamesCombinations)[np.array(self.finalPerformances) >= thresholdToKeep]
            
            featureCombinationInds = []
            featureNamesTemp = set()
            # For each combination that scored higher than the threshold
            for currCombo in goodCombinations:
                # Add all of the features in the combination to the set of good features (we will continue to use these in future combinations)
                for feature in currCombo.split():
                    featureNamesTemp.add(feature)
            featureNames = list(featureNamesTemp)
            
            # Create new combinations with n + 1 features, using all of the features that were in the good combinations
            featureCombinationInds = list(itertools.combinations(range(0, len(featureNames)), curr_numFeaturesCombine + 1))
            
            numFeatureCombnatons = len(featureCombinationInds)
            # Set the new threshold to be the lowest probable performance of the best combination
            # Three standard deviations below the mean of the best combination
            thresholdToKeep = self.finalPerformances[0] - (3 * self.finalPerformancesSTDs[0])
            
        return np.array(self.finalPerformances), np.array(self.finalPerformancesSTDs), np.array(self.featureNamesCombinations)


# -------------------------------------------------------------------------- #


