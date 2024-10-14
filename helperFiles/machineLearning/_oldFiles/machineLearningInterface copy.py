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
        if trainingFolder is not None:
            self.saveDataFolder = trainingFolder + "dataAnalysis/Machine Learning/"
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
        
    def averageModelAccuracy(self, modelInd, featureData, featureLabels, featureNames, numEpochs = 200, stratifyBy = None, testSplitRatio = 0.2, blink = False):
        modelPerformances = []
        modelPerformances_Blink = []
        modelPerformances_Wire = []
        # For each trial
        for _ in range(numEpochs):
            # Randomly split the data
            Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(featureData, featureLabels, test_size=testSplitRatio, shuffle= True, stratify=stratifyBy)

            # Train the model and save the accuracy
            modelPerformance = self.modelClasses[modelInd].trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels, featureNames)
            modelPerformances.append(modelPerformance)
            
            if blink:
                modelPerformances_Blink.append(self.modelClasses[modelInd].scoreModel(featureData[featureLabels == 0], featureLabels[featureLabels == 0]))
                modelPerformances_Wire.append(self.modelClasses[modelInd].scoreModel(featureData[featureLabels == 2], featureLabels[featureLabels == 2]))
            
        
        # Return the average classification accuracy
        averagePerformance = stats.trim_mean(modelPerformances, 0.2)
        averagePerformance_Blink = stats.trim_mean(modelPerformances_Blink, 0.2)
        averagePerformance_Wire = stats.trim_mean(modelPerformances_Wire, 0.2)
        stdPerformance = np.std(modelPerformances)
        
        # plt.hist(modelPerformances, bins=50)
        # plt.axvline(averagePerformance, color='red', linestyle='dashed')
        # plt.show()
        # print("DONE", len(modelPerformances))
        return averagePerformance, stdPerformance, averagePerformance_Blink, averagePerformance_Wire

    def getMedianScore(self, scores, plotScores = False):
        # Plot the data distribution
        if plotScores:
            plt.hist(scores, 100, facecolor='blue', alpha=0.5)
            
        # Fit the skewed distribution and take the median
        ae, loce, scalee = stats.skewnorm.fit(scores)
        medianScore = np.round(loce*100, 2)
        
        return medianScore
            
    def findWorstSubject(self, modelInd, featureData, featureLabels, featureNames, subjectOrder, featureNamesCombination_String):
        subjectOrder = np.asarray(subjectOrder)
        featureLabels = np.asarray(featureLabels)
        
        # Get All Possible itertools.combinations
        subjectCombinationInds = itertools.combinations(range(0, len(featureLabels)),  len(featureLabels) - 3)
        # subjectCombinationInds = itertools.combinations(range(0, len(subjectOrder)),  len(subjectOrder) - 2)        
        allSubjectLabels = set(range(0, len(featureLabels)))
        
        removedInds = []
        removedSubjects = []
        finalPerformances = []
        for subjectInds in subjectCombinationInds:
            # subjectIndsReal = []
            # for subjectInd in subjectInds:
            #     subjectIndsReal.extend([subjectInd*6+j for j in range(6)])
            # subjectInds = subjectIndsReal
            
            # Reset the Input Variab;es
            self.resetModel() # Reset the ML Model
            
            culledSubjectData =  featureData[subjectInds, :]
            culledSubjectLabels = featureLabels[list(subjectInds)]

            # Score the model with this data set.
            modelPerformance = self.averageModelAccuracy(modelInd, culledSubjectData, culledSubjectLabels, featureNames, numEpochs = 50, stratifyBy = culledSubjectLabels, testSplitRatio = 0.3)
                        
            # Save which subjects were removed
            discardedSubjectInds = np.asarray(list(allSubjectLabels.difference(set(subjectInds))))
            removedSubject = subjectOrder[discardedSubjectInds]
        
            insertionPoint = bisect.bisect(finalPerformances, -modelPerformance, key=lambda x: -x)
            # Save the model score and standard deviation
            removedInds.insert(insertionPoint, discardedSubjectInds)
            removedSubjects.insert(insertionPoint, removedSubject)
            finalPerformances.insert(insertionPoint, modelPerformance)
            
        print(featureNamesCombination_String, finalPerformances[0], finalPerformances[-1], removedSubjects[0], removedInds[0])
        # print(finalPerformances[0], finalPerformances[-1], set(removedSubjects[0]), min(removedInds[0]), max(removedInds[0]))
                    
    def analyzeFeatureCombinations(self, modelInd, featureData, featureLabels, featureNames, numFeaturesCombine, subjectOrder, saveData = True, 
                                   saveExcelName = "Feature Combination Accuracy.xlsx", printUpdateAfterTrial = 750, scaleLabels = True):
        # Format the incoming data
        featureNames = np.asarray(featureNames)
        featureData = np.asarray(featureData.copy())
        featureLabels = np.asarray(featureLabels.copy())
        # Function variables
        numModelsTrack = 1000   # The number of models to track
        
        # Get All Possible itertools.combinations
        self.finalPerformances = []; self.finalPerformancesSTDs = []; self.featureNamesCombinations = [];
        
        featureCombinationInds = itertools.combinations(range(0, len(featureNames)), numFeaturesCombine)
        subjectCombinationInds = list(itertools.combinations(range(0, len(featureLabels)),  len(featureLabels) - int(len(featureLabels)*0)))
        # Find total combinations
        numFeatureCombnatons = math.comb(len(featureNames), numFeaturesCombine)
                
        # Standardize the features
        standardizeClass_Features = standardizeData(featureData)
        featureData = standardizeClass_Features.standardize(featureData)
        # Standardize the labels
        if scaleLabels:
            standardizeClass_Labels = standardizeData(featureLabels)
            featureLabels = standardizeClass_Labels.standardize(featureLabels)
        
        t1 = time.time(); combinationRoundInd = -1
        # For Each Combination of Features
        for combinationInds in featureCombinationInds:
            combinationInds = list(combinationInds)
            combinationRoundInd += 1
            
            # Collect the Signal Data for the Specific Features
            featureData_culledFeatures = featureData[:, combinationInds]
            # Collect the Specific Feature Names
            featureNamesCombination_String = ''; featureNamesCull = []
            for name in featureNames[combinationInds]:
                featureNamesCombination_String += name + ' '
                featureNamesCull.append(name)
                            
            modelPerformances = []; modelSTDs = []
            for subjectInds in subjectCombinationInds:
                subjectInds = list(subjectInds)
                # Reset the Input Variab;es
                self.resetModel() # Reset the ML Model
                                
                # Collect the Signal Data for the Specific Subjects
                featureData_finalCull = featureData_culledFeatures[subjectInds, :]
                featureLabels_finalCull = featureLabels[subjectInds]

                # Score the model with this data set.
                modelPerformance, modelSTD, modelPerformace_Blink, modelPerformace_Wire = self.averageModelAccuracy(modelInd, featureData_finalCull, featureLabels_finalCull, featureNamesCull, numEpochs = 100, stratifyBy = featureLabels_finalCull, testSplitRatio = 0.2, blink=True)
                modelPerformances.append(modelPerformance)
                modelSTDs.append(modelSTD)
                
            # if numFeaturesCombine != 1:
            # self.findWorstSubject(modelInds, featureData_culledFeatures, featureLabels, featureNames, subjectOrder, featureNamesCombination_String)
                    
            modelPerformance = stats.trim_mean(modelPerformances, 0.3)
            # If the model's performance is one of the top scores
            if len(self.finalPerformances) < numModelsTrack or modelPerformance > self.finalPerformances[-1]:
                modelSTD = 0 if len(modelSTDs) <= 0 else np.mean(modelSTDs)
                
                insertionPoint = bisect.bisect(self.finalPerformances, -modelPerformance, key=lambda x: -x)
                # Save the model score and standard deviation
                self.featureNamesCombinations.insert(insertionPoint, featureNamesCombination_String[0:-1])
                self.finalPerformances.insert(insertionPoint, modelPerformance)
                self.finalPerformancesSTDs.insert(insertionPoint, modelSTD)
                
                # Only track the best models
                if len(self.finalPerformances) > numModelsTrack:
                    self.finalPerformances.pop()
                    self.finalPerformancesSTDs.pop()
                    self.featureNamesCombinations.pop()

            # Report an Update Every Now and Then
            if (combinationRoundInd%printUpdateAfterTrial == 0 and combinationRoundInd != 0) or combinationRoundInd == 20:
                t2 = time.time()
                percentComplete = 100*combinationRoundInd/numFeatureCombnatons
                setionPercent = 100*min(combinationRoundInd or 1, printUpdateAfterTrial)/numFeatureCombnatons
                print(str(np.round(percentComplete, 2)) + "% Complete; Estimated Time Remaining: " + str(np.round((t2-t1)*(100-percentComplete)/(setionPercent*60), 2)) + " Minutes")
                t1 = time.time()
                    
        print(self.finalPerformances[0], self.finalPerformancesSTDs[0], self.featureNamesCombinations[0])
        # Save the Data in Excel
        if saveData:
            excelProcessing.processMLData().saveFeatureComparison(np.dstack((self.finalPerformances, self.finalPerformancesSTDs, 
                                        self.featureNamesCombinations))[0], [], ["Mean Score", "STD", "Feature Combination"], 
                                        self.saveDataFolder, saveExcelName, sheetName = str(numFeaturesCombine) + " Features in Combination", saveFirstSheet = True)
        return np.asarray(self.finalPerformances), np.asarray(self.finalPerformancesSTDs), np.asarray(self.featureNamesCombinations)

# -------------------------------------------------------------------------- #


