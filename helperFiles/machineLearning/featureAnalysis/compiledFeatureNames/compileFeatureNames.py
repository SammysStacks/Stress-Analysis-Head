# General
import os
import numpy as np

# Import interfaces for reading/writing data
from ....dataAcquisitionAndAnalysis.excelProcessing.extractDataProtocols import extractData


class compileFeatureNames:

    def __init__(self):
        # Store Extracted Features
        self.possibleFeatureNames = np.char.lower(["eog", "eeg", "ecg", "eda", "emg", "temp", "lowFreq", "highFreq", "acc", "bvp"])
        self.featureNamesFolder = os.path.dirname(__file__) + "/allFeatures/"
        self.extractDataInterface = extractData()

    def extractFeatureNames(self, extractFeaturesFrom):
        extractFeaturesFrom = [featureType.lower() for featureType in extractFeaturesFrom]  # Does NOT have to be unique.
        # Compile feature names
        featureNames = []
        biomarkerFeatureOrder = []
        biomarkerFeatureNames = []

        # For each feature we are processing.
        for featureName in extractFeaturesFrom:
            # Assert that we have a protocol for this feature.
            assert featureName in self.possibleFeatureNames, f"Unknown feature name: {featureName}"

            # Organize and store the feature name information.
            self.getFeatures(featureName, biomarkerFeatureNames, featureNames, biomarkerFeatureOrder)

        # Ensure the proper data structure.
        biomarkerFeatureOrder = np.asarray(biomarkerFeatureOrder, dtype=str)
        featureNames = np.asarray(featureNames, dtype=str)

        return featureNames, biomarkerFeatureNames, biomarkerFeatureOrder

    def getFeatures(self, featureName, biomarkerFeatureNames, featureNames, biomarkerFeatureOrder):
        # Specify the Paths to the EOG Feature Names
        featuresFile = self.featureNamesFolder + f"{featureName}FeatureNames.txt"
        # Extract the EOG Feature Names we are Using
        currentFeatureNames = self.extractDataInterface.extractFeatureNames(featuresFile, prependedString="finalFeatures.extend([", appendToName=f"_{featureName.upper()}")
        # Create Data Structure to Hold the Features.
        biomarkerFeatureNames.append(currentFeatureNames)  # biomarkerFeatureNames dimensions: [numBiomarkers, numFeatures_perBiomarker]
        featureNames.extend(currentFeatureNames)  # featureNames dimensions: [numAllFeatures]
        biomarkerFeatureOrder.append(featureName)  # biomarkerFeatureOrder dimensions: [numBiomarkers]
