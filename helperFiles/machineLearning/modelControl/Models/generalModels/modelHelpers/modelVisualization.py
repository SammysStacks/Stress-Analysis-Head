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
import numpy as np
# Modules for Plotting
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.colors import ListedColormap
# Machine Learning Modules
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------------------- #
# ---------------------------- Visualize Model ----------------------------- #

class modelVisualization:
    def __init__(self, saveDataFolder):
        # Store Parameters
        self.saveDataFolder = saveDataFolder
        
        # Plotting Styles
        self.stepSize = 0.01 # step size in the mesh
        self.cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue', 'red']) # Colormap
        self.cmap_bold = ['darkorange', 'c', 'darkblue', 'darkred'] # Colormap

        # Create Output File Directory to Save Data: If None
        os.makedirs(self.saveDataFolder, exist_ok=True)
        
        
    def accuracyDistributionPlot_Average(self, featureData, featureLabels, machineLearningClasses, analyzeType = "Full", name = "Accuracy Distribution", testSplitRatio = 0.4):
        numAverage = 200
        
        accMat = np.zeros((len(machineLearningClasses), len(machineLearningClasses)))
        # Taking the Average Score Each Time
        for roundInd in range(1,numAverage+1):
            # Train the Model with the Training Data
            Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(featureData, featureLabels, test_size=testSplitRatio, shuffle= True, stratify=featureLabels)
            self.predictionClass.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels)
            
            if analyzeType == "Full":
                inputData = featureData; inputLabels = featureLabels
            elif analyzeType == "Test":
                inputData = Testing_Data; inputLabels = Testing_Labels
            else:
                sys.exit("Unsure which data to use for the accuracy map");

            testingLabelsML = self.predictionClass.predict(inputData)
            # Calculate the Accuracy Matrix
            accMat_Temp = np.zeros((len(machineLearningClasses), len(machineLearningClasses)))
            for ind, channelFeatures in enumerate(inputData):
                # Sum(Row) = # of Gestures Made with that Label
                # Each Column in a Row = The Number of Times that Gesture Was Predicted as Column Label #
                accMat_Temp[inputLabels[ind]][testingLabelsML[ind]] += 1
        
            # Scale Each Row to 100
            for label in range(len(machineLearningClasses)):
                accMat_Temp[label] = 100*accMat_Temp[label]/(np.sum(accMat_Temp[label]))
            
                # Scale Each Row to 100
            for label in range(len(machineLearningClasses)):
                accMat[label] = (accMat[label]*(roundInd-1) + accMat_Temp[label])/roundInd

        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = 2
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Ubuntu'
        plt.rcParams['font.monospace'] = 'Ubuntu Mono'
                
        # Make plot
        fig, ax = plt.subplots()
        fig.set_size_inches(5,5)
        
        # Make heatmap on plot
        im = createMap.heatmap(accMat, machineLearningClasses, machineLearningClasses, ax=ax,
                           cmap="binary")
        createMap.annotate_heatmap(im, accMat, valfmt="{x:.2f}",)
        
        # Style the Fonts
        font = {'family' : 'serif',
                'serif': 'Ubuntu',
                'size'   : 20}
        matplotlib.rc('font', **font)

        
        # Format, save, and show
        fig.tight_layout()
        plt.savefig(self.saveDataFolder + name + " " + analyzeType + " " + self.modelType + ".png", dpi=130, bbox_inches='tight')
        plt.show()

    def mapTo2DPlot(self, featureData, featureLabels, name = "Channel Map"):
        # Plot and Save
        fig = plt.figure()
        fig.set_size_inches(15,12)
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(featureData, featureLabels)
        
        mds = MDS(n_components=2,random_state=0, n_init = 4)
        X_2d = mds.fit_transform(X_scaled)
        
        X_2d = self.rotatePoints(X_2d, -np.pi/2).T
        
        figMap = plt.scatter(X_2d[:,0], X_2d[:,1], c = featureLabels, cmap = plt.cm.get_cmap('cubehelix', self.numClasses), s = 130, marker='.', edgecolors='k')        
        
        # Figure Aesthetics
        fig.colorbar(figMap, ticks=range(self.numClasses), label='digit value')
        figMap.set_clim(-0.5, 5.5)
        plt.title('Channel Feature Map');
        fig.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=200, bbox_inches='tight')
        plt.show() # Must be the Last Line
        
        return X_2d
    
    def rotatePoints(self, rotatingMatrix, theta_rad = -np.pi/2):

        A = np.matrix([[np.cos(theta_rad), -np.sin(theta_rad)],
                       [np.sin(theta_rad), np.cos(theta_rad)]])
        
        m2 = np.zeros(rotatingMatrix.shape)
        
        for i,v in enumerate(rotatingMatrix):
          w = A @ v.T
          m2[i] = w
        m2 = m2.T
        
        return m2
    
    def plot3DLabels(self, featureData, featureLabels, name = "Channel Feature Distribution"):
        # Plot and Save
        fig = plt.figure()
        fig.set_size_inches(15,12)
        ax = plt.axes(projection='3d')
        
        # Scatter Plot
        ax.scatter3D(featureData[:, 3], featureData[:, 1], featureData[:, 2], c = featureLabels, cmap = plt.cm.get_cmap('cubehelix', self.numClasses), s = 100, edgecolors='k')
        
        ax.set_title('Channel Feature Distribution');
        ax.set_xlabel("Channel 4")
        ax.set_ylabel("Channel 2")
        ax.set_zlabel("Channel 3")
        #fig.tight_layout()
        fig.savefig(self.saveDataFolder + name + " " + self.modelType + ".png", dpi=200, bbox_inches='tight')
        plt.show() # Must be the Last Line
    
    def plot3DLabelsMovie(self, featureData, featureLabels, name = "Channel Feature Distribution Movie"):
        # Plot and Save
        fig = plt.figure()
        #fig.set_size_inches(15,15,10)
        ax = plt.axes(projection='3d')
        
        # Initialize Relevant Channel 4 Range
        errorPoint = 0.01; # Width of Channel 4's Values
        channel4Vals = np.arange(min(featureData[:, 3]), max(featureData[:, 3]), 2*errorPoint)
        
        # Initialize Movie Writer for Plots
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=name + " " + self.modelType, artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(fps=2, metadata=metadata)
        
        with writer.saving(fig, self.saveDataFolder + name + " " + self.modelType + ".mp4", 300):
            for channel4Val in channel4Vals:
                channelPoints1 = featureData[:, 0][abs(featureData[:, 3] - channel4Val) < errorPoint]
                channelPoints2 = featureData[:, 1][abs(featureData[:, 3] - channel4Val) < errorPoint]
                channelPoints3 = featureData[:, 2][abs(featureData[:, 3] - channel4Val) < errorPoint]
                currentLabels = featureLabels[abs(featureData[:, 3] - channel4Val) < errorPoint]
                
                if len(currentLabels) != 0:
                    # Scatter Plot
                    figMap = ax.scatter3D(channelPoints1, channelPoints2, channelPoints3, "o", c = currentLabels, cmap = plt.cm.get_cmap('cubehelix', self.numClasses), s = 50, edgecolors='k')
        
                    ax.set_title('Channel Feature Distribution; Channel 4 = ' + str(channel4Val) + " ± " + str(errorPoint));
                    ax.set_xlabel("Channel 1")
                    ax.set_ylabel("Channel 2")
                    ax.set_zlabel("Channel 3")
                    ax.yaxis._axinfo['label']['space_factor'] = 20
                    
                    ax.set_xlim3d(0, max(featureData[:, 0]))
                    ax.set_ylim3d(0, max(featureData[:, 1]))
                    ax.set_zlim3d(0, max(featureData[:, 2]))
                    
                    # Figure Aesthetics
                    cb = fig.colorbar(figMap, ticks=range(self.numClasses), label='digit value')
                    plt.rcParams['figure.dpi'] = 300
                    figMap.set_clim(-0.5, 5.5)
                    
                    # Write to Video
                    writer.grab_frame()
                    # Clear Previous Frame
                    plt.cla()
                    cb.remove()
                
        plt.show() # Must be the Last Line
        
        def plotModel(self, Training_Data, Testing_Data, Training_Labels, Testing_Labels, model):    
            # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = Training_Data[:, 0].min(), Training_Data[:, 0].max()
            y_min, y_max = Training_Data[:, 1].min(), Training_Data[:, 1].max()
            xx, yy = np.meshgrid(np.arange(x_min, x_max, self.stepSize),
                                 np.arange(y_min, y_max, self.stepSize))
            
            FFMpegWriter = manimation.writers['ffmpeg']
            metadata = dict(title="", artist='Matplotlib', comment='Movie support!')
            writer = FFMpegWriter(fps=3, metadata=metadata)
            
            setPointX4 = 0.002;
            errorPoint = 0.003;
            dataWithinChannel4 = Training_Data[abs(Training_Data[:,3] - setPointX4) <= errorPoint]
            
            channel3Vals = np.arange(0.0, dataWithinChannel4[:,2].max(), 0.01)
            fig = plt.figure()
            
            with writer.saving(fig, "./LogisticRegression.mp4", 300):
                for setPointX3 in channel3Vals:
                
                    #Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    # setPointX3 = 0.15; setPointX4 = 0.12;
                    x3 = np.ones(np.shape(xx.ravel())[0])*setPointX3
                    x4 = np.ones(np.shape(xx.ravel())[0])*setPointX4
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel(), x3, x4])# Put the result into a color plot
                            
                    # Put the result into a color plot                    
                    Z = Z.reshape(xx.shape)

                    plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap('cubehelix', 6), alpha=0.7, vmin=0, vmax=5)
                    
                    xPoints = []; yPoints = []; yLabelPoints = []
                    for j, point in enumerate(Training_Data):
                        if abs(point[2] - setPointX3) <= errorPoint and abs(point[3] - setPointX4) <= errorPoint:
                            xPoints.append(point[0])
                            yPoints.append(point[1])
                            yLabelPoints.append(Training_Labels[j])
                    
                    plt.scatter(xPoints, yPoints, c=yLabelPoints, cmap=plt.cm.get_cmap('cubehelix', 6), edgecolors='grey', s=50, vmin=0, vmax=5)
                    
                    plt.xlim(xx.min(), xx.max())
                    plt.ylim(yy.min(), yy.max())
                    #plt.title("Classification (k = %i, weights = '%s')"
                    #          % (self.numNeighbors, weight))
                    plt.title("Channel3 = " + str(round(setPointX3,3)) + "; Channel4 = " + str(setPointX4) + "; Error = " + str(errorPoint))
                    plt.xlabel('Channel 1')
                    plt.ylabel('Channel 2')
                    plt.rcParams['figure.dpi'] = 300
                    
                    cb = plt.colorbar(ticks=range(6), label='digit value')
                    plt.clim(-0.5, 5.5)
                
                    # Write to Video
                    writer.grab_frame()
                    plt.cla()
                    cb.remove()
                 