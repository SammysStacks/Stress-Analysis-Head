

import sklearn
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50, ResNet152
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import VGG19

class timeSeries_CNN:
    
    def __init__(self, imageDimension):
        # Feature extraction parameters
        self.imageDimension = imageDimension
        self.ResNet50 = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.ResNet152 = ResNet152(weights='imagenet', include_top=False, pooling='avg')
        self.VGG19 = VGG19(weights='imagenet', include_top=False, input_shape=(imageDimension[0], imageDimension[1], 3))
        
    def get_image_features_DEPRECATED(self, images):
        features = []
        for image in images:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

            x = preprocess_input(image)
            x = np.expand_dims(x, axis=0)

            feature = self.ResNet152.predict(x).flatten()
            features.append(feature)
        # features = np.concatenate(features, axis=0)
        return np.asarray(features)
    
    
    def get_image_features(self, images):
        # Preprocess the images for the model
        modelInput = preprocess_input(images)
        modelInput = np.expand_dims(modelInput, axis=0)
        print(modelInput.shape)
        # Feature extraction from the images.
        feature = self.ResNet152.predict(modelInput)
        print(len(feature))
        
        return features

    def create_pixelated_curve(self, rawTimes, rawDataPoints, timeBoundaries, yBoundaries):
        """ Turn time series data into a 2D image array """
        # Extract the relevant information
        minTime, maxTime = timeBoundaries
        minVal, maxVal = yBoundaries
        # Initialize the array details.
        finalArray = np.zeros(self.imageDimension)
        dataPoints = np.linspace(minVal, maxVal, self.imageDimension[0])
        timepoints = np.linspace(minTime, maxTime, self.imageDimension[1])
            
        # For every raw data point
        for pointInd in range(len(rawDataPoints)):
            dataPoint = rawDataPoints[pointInd]
            timePoint = rawTimes[pointInd]
            
            # If the point is outside of the time window
            if maxTime < timePoint or timePoint < minVal:
                # Ignore the point
                continue
            
            xInd = np.searchsorted(timepoints, timePoint, side='left')
            yInd = np.searchsorted(dataPoints, dataPoint, side='left')
                    
            finalArray[yInd][xInd] = 1
    
        return timepoints, dataPoints, finalArray

if True:
    featureInd = 121
    matrixDim = (100, 1000)  # Size of the 2D array
    
    cnn = timeSeries_CNN(matrixDim)
    images = np.zeros((matrixDim[0], matrixDim[1], len(allRawFeatureIntervals)))
    for pointInd in range(len(allRawFeatureIntervals)):
        rawDataPoints = allRawFeatureIntervals[pointInd][featureInd]  # Raw data points
        rawTimes = allRawFeatureIntervalTimes[pointInd][featureInd]  # Timepoints
        
        timeBoundaries = (rawTimes[-1] - 3*60, rawTimes[-1])
        yBoundaries = (min(rawDataPoints), max(rawDataPoints))
        
        x,y,z = cnn.create_pixelated_curve(rawTimes, rawDataPoints, timeBoundaries, yBoundaries)
        images[:,:,pointInd] = z
        
    # allFeatures = cnn.get_image_features(images)
    allFeatures = cnn.get_image_features_DEPRECATED(images.reshape((len(allRawFeatureIntervals), matrixDim[0], matrixDim[1])))
    
    
    # # for each emotion of the survey
    for labelInd in range(len(surveyAnswersList[0])):
        labels = surveyAnswersList[:, labelInd]
        surveyQuestion = surveyQuestions[labelInd]
        
        
    # labelType = ["PA", "NA", "State Anxiety"]
    # for labelInd in range(len(allFinalLabels)):
    #     labels = allFinalLabels[labelInd]
    #     surveyQuestion = labelType[labelInd]
        
        for featureInd in range(len(allFeatures[0])):
            features = allFeatures[:, featureInd]
            
            R2 = sklearn.metrics.r2_score(labels, features)
            
            if 0 < R2:
                plt.plot(features, labels, 'ko')
                plt.title(R2)
                plt.ylabel(surveyQuestion)
                plt.xlabel(featureInd)
                plt.show()
        
    sys.exit()
        
        
else:
    rawTimes = np.arange(0, 100, 0.01)
    rawDataPoints = np.sin(rawTimes)
    
    matrixDim = (100, 1000)  # Size of the 2D array
    timeBoundaries = (0, 10)
    yBoundaries = (-1,1)


cnn = timeSeries_CNN(matrixDim)
x,y,z = cnn.create_pixelated_curve(rawTimes, rawDataPoints, timeBoundaries, yBoundaries)

features = cnn.get_image_features([z])

# Plot the heatmap
plt.imshow(z, cmap='hot', aspect='auto', origin='lower')
plt.colorbar()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Heatmap')

plt.show()

plt.plot(rawTimes, rawDataPoints)