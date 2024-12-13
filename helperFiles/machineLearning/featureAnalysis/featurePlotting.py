# General
import os
import collections
import numpy as np
from scipy import stats
from copy import deepcopy

# Modules for Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Import Files for Machine Learning
from ..modelControl.modelSpecifications.compileModelInfo import compileModelInfo
from ...globalPlottingProtocols import globalPlottingProtocols


class featurePlotting(globalPlottingProtocols):

    def __init__(self, saveDataFolder, overwrite):
        super().__init__()
        # Save Information
        self.overwrite = overwrite
        self.saveDataFolder = saveDataFolder

        self.modelInfoClass = compileModelInfo()

        # Plotting
        plt.ioff()  # prevent memory leaks; plt.ion()
        self.colorList = ['k', 'tab:red', 'tab:blue', 'brown', 'purple', 'tab:green', 'k', 'tab:red']

    @staticmethod
    def calculateTimeUnit(timepoints, preAveragingSeconds, averageIntervalList, surveyCollectionTimes, experimentTimes):
        # Assume all time is in seconds
        timeUnit = "Second"
        scaleTime = 1
        # If the analysis is longer than 5 hours, use hours
        if 60 * 60 * 5 < timepoints[-1] - timepoints[0]:
            print("\t\tUsing Hours")
            timeUnit = "Hour"
            scaleTime = 60 * 60
        # Else if the analysis is longer than an hour, use minutes
        elif 60 * 60 < timepoints[-1] - timepoints[0]:
            print("\t\tUsing Minutes")
            timeUnit = "Minute"
            scaleTime = 60

        # Scale the time units.
        surveyCollectionTimes = np.asarray(surveyCollectionTimes.copy()) / scaleTime
        timepoints = np.asarray(timepoints.copy()) / scaleTime
        averageIntervalList = np.asarray(averageIntervalList) / scaleTime
        preAveragingSeconds = np.asarray(preAveragingSeconds) / scaleTime
        experimentTimes = np.asarray(experimentTimes.copy()) / scaleTime

        # Return the scaled times wit the new unit
        return timeUnit, timepoints, preAveragingSeconds, averageIntervalList, surveyCollectionTimes, experimentTimes

    @staticmethod
    def getAxisLimits(dataOnPlots=(), yLim=(None, None)):
        # For each plot on the graph
        for data in dataOnPlots:
            # Calculate the y-limits for the data
            yMin = 0.9 * min(data) if min(data) > 0 else min(data) * 1.1
            yMax = 1.1 * max(data) if max(data) > 0 else max(data) * 0.9

            # Update the axis bounds to account for all the data
            yLim[0], yLim[1] = min(yLim[0] or np.inf, yMin), max(yLim[1] or -np.inf, yMax)

        return yLim

    def addSurveyInfo(self, ax, surveyCollectionTimes, experimentTimes, experimentNames, predictionType, legendAxes, legendLabels, yLim, recordedScores=None):
        # Ensure the correct data type of variables
        experimentTimes = np.asarray(experimentTimes.copy())
        surveyCollectionTimes = np.asarray(surveyCollectionTimes.copy())

        # Add the feature collection times to the graph.
        if len(surveyCollectionTimes) != 0:
            legendAxes.append(ax.vlines(surveyCollectionTimes, yLim[0], yLim[1], 'tab:blue', linewidth=2, zorder=2))
            legendLabels.append(predictionType.replace(" ", "") + " Questionnaire Completed")
        # Add the experimental information to the graph.
        if len(experimentTimes) != 0:
            ax.vlines(experimentTimes[:, 0], yLim[0], yLim[1], 'tab:red', linestyles='--', linewidth=0.5, zorder=3)
            ax.vlines(experimentTimes[:, 1], yLim[0], yLim[1], 'tab:red', linestyles='--', linewidth=0.5, zorder=3)
            for experimentInd in range(len(experimentTimes)):
                # Add the labeled score
                if recordedScores is not None:
                    recordedScore = recordedScores[experimentInd]
                    addedScoreAxis = ax.hlines([recordedScore], experimentTimes[experimentInd][0], surveyCollectionTimes[experimentInd], 'tab:red', linewidth=2, zorder=4)
                    if experimentInd == 0:
                        legendAxes.append(addedScoreAxis)
                        legendLabels.append("Recorded " + predictionType)

                legendAxes.append(ax.fill_between(experimentTimes[experimentInd], yLim[0], yLim[1], color=self.colorList[experimentInd % len(self.colorList)], alpha=0.15))
                legendLabels.append(f"Experiment: {experimentNames[experimentInd]}")

        return ax, legendAxes, legendLabels

    @staticmethod
    def addBestFit(x, y, plotLabel=True, color='tab:red'):
        p = np.polyfit(x, y, 1)
        xNew = np.arange(min(x), max(x), (max(x) - min(x)) / 1000)
        plt.plot(xNew, np.polyval(p, xNew), '--', color=color, linewidth=2)

        if plotLabel:
            # Add the best fit equation as a label
            equation = f"Best Fit: {p[0]:.2f}x + {p[1]:.2f}"  # Example equation format: y = mx + c
            plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, ha='left', va='top')

    def plotRawData(self, readData, compiledRawData, surveyCollectionTimes, experimentTimes, experimentNames, streamingOrder, folderName):
        print("\tPlotting raw signals in folder:", folderName)
        # Create/verify a directory to save the figures
        saveDataFolder = self.saveDataFolder + folderName
        os.makedirs(saveDataFolder, exist_ok=True)

        # Scale the time to fit nicely into a plot
        timeUnit, timepoints, _, _, surveyCollectionTimes, experimentTimes = \
            self.calculateTimeUnit(compiledRawData[0], np.asarray([]), np.asarray([]), surveyCollectionTimes, experimentTimes)

        counter = collections.Counter()
        streamingChannelIndices = [counter.update({item: 1}) or counter[item] - 1 for item in streamingOrder]

        plotTitles = [" All", " Filtered"]
        # For each biomarker being streamed
        for streamingInd in range(len(compiledRawData[1])):
            biomarkerName = streamingOrder[streamingInd].upper()
            biomarkerData = np.asarray(compiledRawData[1][streamingInd].copy())
            channelIndex = streamingChannelIndices[streamingInd]

            # Filter the data
            analysisInd = readData.analysisOrder.index(biomarkerName.lower())
            filteredTimes, filteredData, _ = readData.analysisList[analysisInd].filterData(timepoints, biomarkerData)
            biomarkerName += f" Channel {channelIndex}"

            # For raw and filtered data plots
            for plotInd in range(len(plotTitles)):
                # Make the figure
                fig = plt.figure()
                ax = plt.gca()
                legendAxes = []
                legendLabels = []
                yLim = self.getAxisLimits([filteredData], yLim=[None, None])

                # Plot the raw signals
                if plotInd == 0:
                    yLim = self.getAxisLimits([biomarkerData], yLim)
                    legendLabels.append(biomarkerName + " Raw Signal")
                    legendAxes.append(ax.plot(timepoints, biomarkerData, 'ok', markersize=4)[0])
                # Plot the filtered signals
                legendLabels.append(biomarkerName + " Filtered Signal")
                legendAxes.append(ax.plot(filteredTimes, filteredData, 'o', c='tab:brown', markersize=2)[0])
                # Add experimental/survey information to the plot
                ax, legendAxes, legendLabels = self.addSurveyInfo(ax, surveyCollectionTimes, experimentTimes, experimentNames, "All", legendAxes, legendLabels, yLim, recordedScores=None)

                # Add figure information
                ax.set_ylim(yLim)
                ax.set_xlabel("Time (" + timeUnit + ")")
                ax.set_ylabel(biomarkerName + plotTitles[plotInd] + " Signal")
                ax.set_title(biomarkerName + plotTitles[plotInd] + " Signal")
                # Add Figure Legend
                legend = ax.legend(legendAxes[0:10], legendLabels[0:10], loc=9, bbox_to_anchor=(1.35, 1.02))
                # Save and clear the figure
                fig.savefig(saveDataFolder + plotTitles[plotInd] + "_" + biomarkerName + ".png", dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
                self.clearFigure(fig, legend=legend, showPlot=True); plt.show()

    def plotPredictedScores(self, timepoints, predictedLabels, recordedScores, surveyCollectionTimes, experimentTimes, experimentNames, predictionType="Stress", folderName="realTimePredictions/"):
        if len(predictedLabels) == 0:
            print("\tNo prediction for " + predictionType)
            return None

        print("\tPlotting predictions in folder:", folderName)
        # Assert data integrity
        assert len(timepoints) == len(predictedLabels), "Predictions dont match the timepoints"
        assert len(recordedScores) == len(experimentTimes)

        # Create/verify a directory to save the figures
        saveDataFolder = self.saveDataFolder + folderName
        os.makedirs(saveDataFolder, exist_ok=True)

        # Scale the time to fit nicely into a plot
        timeUnit, timepoints, _, _, surveyCollectionTimes, experimentTimes = \
            self.calculateTimeUnit(timepoints, np.asarray([]), np.asarray([]), surveyCollectionTimes, experimentTimes)

        # Make a figure
        fig = plt.figure()
        ax = plt.gca()
        legendAxes = []
        legendLabels = []
        yLim = self.getAxisLimits([predictedLabels, recordedScores], yLim=[None, None])

        legendLabels.append("Predicted " + predictionType)
        legendAxes.append(ax.plot(timepoints, predictedLabels, 'k', linewidth=2)[0])
        # Add the feature collection times to the graph.
        ax, legendAxes, legendLabels = self.addSurveyInfo(ax, surveyCollectionTimes, experimentTimes, experimentNames, predictionType, legendAxes, legendLabels, yLim, recordedScores)

        # Add figure information
        ax.set_xlabel("Time (" + timeUnit + ")")
        ax.set_ylabel(predictionType + " Score")
        ax.set_title("Real-Time " + predictionType + " Score Prediction")
        ax.set_ylim(yLim)
        # Add Figure Legend
        legend = ax.legend(legendAxes[0:10], legendLabels[0:10], loc=9, bbox_to_anchor=(1.35, 1.02))
        # Save and clear the figure
        fig.savefig(saveDataFolder + predictionType + ".png", dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
        self.clearFigure(fig, legend=legend, showPlot=True); plt.show()

    def singleFeatureAnalysis(self, readData, timepoints, featureList, featureNames, preAveragingSeconds=0, averageIntervalList=(0, 30),
                              surveyCollectionTimes=np.asarray([]), experimentTimes=(), experimentNames=(), folderName="singleFeatureAnalysis/"):
        print("\tPlotting features in folder:", folderName)
        # Assert data integrity
        assert len(featureNames) == len(featureList[0]), f"Mismatch between feature names and features given: {len(featureNames)} != {len(featureList[0])}"
        assert len(featureList[:, 0]) == len(timepoints), f"Mismatch between features and times: {len(featureList[:, 0])} != {len(timepoints)}"

        # Do not waste time if the analysis was performed already.
        saveDataFolder = self.saveDataFolder + folderName
        if not self.overwrite and os.path.exists(saveDataFolder):
            return None
        # Create/verify a directory to save the figures
        os.makedirs(saveDataFolder, exist_ok=True)

        # Scale the time to fit nicely into a plot
        timeUnit, timepoints, preAveragingSeconds, averageIntervalList, surveyCollectionTimes, experimentTimes = \
            self.calculateTimeUnit(timepoints, preAveragingSeconds, averageIntervalList, surveyCollectionTimes, experimentTimes)

        # Loop Through Each Feature
        for featureInd in range(len(featureNames)):
            fig = plt.figure()
            ax = plt.gca()
            legendAxes = []
            legendLabels = []
            yLim = [None, None]

            # Extract One Feature from the List
            allFeatures = featureList[:, featureInd]
            # Take Different Averaging Methods
            for colorInd, windowSize in enumerate(averageIntervalList):

                # Average the Feature Together at Each Point
                featureTimes, features = readData.averageFeatures_static(timepoints, allFeatures, windowSize, startTimeInd = 0)

                # Plot the features and track the legend
                legendAxes.append(ax.plot(featureTimes, features, 'o', c=self.colorList[colorInd], markersize=4)[0])
                legendLabels.append(str(windowSize + preAveragingSeconds) + " " + timeUnit + " Feature Average")

                # Keep track of all feature's bounds
                startFeatureIndex = (abs(timepoints - preAveragingSeconds - windowSize)).argmin()
                yLim = self.getAxisLimits([features[startFeatureIndex:]], yLim)

            # Add experimental/survey information to the plot
            ax, legendAxes, legendLabels = self.addSurveyInfo(ax, surveyCollectionTimes, experimentTimes, experimentNames, "All", legendAxes, legendLabels, yLim, recordedScores=None)

            # Add figure information
            ax.set_xlabel("Time (" + timeUnit + ")")
            ax.set_ylabel(featureNames[featureInd])
            ax.set_title(featureNames[featureInd] + " Analysis")
            ax.set_ylim(yLim)
            # Add Figure Legend
            legend = ax.legend(legendAxes[0:10], legendLabels[0:10], loc=9, bbox_to_anchor=(1.35, 1.02))
            # Save and clear the figure
            fig.savefig(saveDataFolder + featureNames[featureInd] + ".png", dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
            self.clearFigure(fig, legend=legend, showPlot=True); plt.show()

    def plotEmotionCorrelation(self, allFinalFeatures, currentSurveyAnswersList, surveyQuestions, featureNames, folderName, subjectOrder=()):
        print("\tPlotting emotion correlation in folder:", folderName)
        # Set up the variables
        allFinalFeatures = np.asarray(allFinalFeatures.copy())
        currentSurveyAnswersList = np.asarray(currentSurveyAnswersList.copy())

        # For each survey question (feature label)
        for surveyQuestionInd in range(len(surveyQuestions)):
            surveyQuestion = surveyQuestions[surveyQuestionInd]
            surveyLabels = currentSurveyAnswersList[:, surveyQuestionInd]

            # Do not waste time if the analysis was performed already.
            saveDataFolder = self.saveDataFolder + folderName + "/" + surveyQuestion + "/"
            if not self.overwrite and os.path.exists(saveDataFolder):
                continue
            # Create/verify a directory to save the figures
            os.makedirs(saveDataFolder, exist_ok=True)

            # For each feature collected
            for featureInd in range(len(allFinalFeatures[0])):
                fig = plt.figure()
                ax = plt.gca()
                # Plot each feature against its surveyLabels
                features = allFinalFeatures[:, featureInd]

                # Plot subjects as different colors
                if len(subjectOrder) != 0:
                    prevSubject = ""
                    colorInd = -1

                    for newSubjectInd in range(len(subjectOrder)):
                        newSubject = subjectOrder[newSubjectInd]
                        if newSubject != prevSubject:
                            colorInd = colorInd + 1 if colorInd + 1 < len(self.colorList) else 0
                            plt.plot(features[newSubjectInd], surveyLabels[newSubjectInd], 'o', c=self.colorList[colorInd], markersize=3, label=newSubject)
                        else:
                            plt.plot(features[newSubjectInd], surveyLabels[newSubjectInd], 'o', c=self.colorList[colorInd], markersize=3)
                        prevSubject = newSubject

                    # plt.legend()
                else:
                    plt.plot(features, surveyLabels, 'o', c=self.colorList[0], markersize=3)

                # Plot the linear best fit
                if len(surveyLabels) > 2:
                    self.addBestFit(features, surveyLabels)

                # Add figure information
                ax.set_ylabel(surveyQuestion)
                ax.set_xlabel(featureNames[featureInd])
                ax.set_title("Feature Correlation with Emotions")
                # Save and clear the figure
                fig.savefig(saveDataFolder + featureNames[featureInd] + " Feature-Emotion Correlation.png", dpi=300, bbox_inches='tight')
                self.clearFigure(fig, legend=None, showPlot=True)

    def plotPsychCorrelation(self, allFinalFeatures, currentSurveyAnswersList, featureNames, folderName, subjectOrder=()):
        print("\tPlotting psych correlation in folder:", folderName)
        # Set up the variables
        allFinalFeatures = np.asarray(allFinalFeatures.copy())
        currentSurveyAnswersList = np.asarray(currentSurveyAnswersList)

        # Specify the order of the positive and negative survey questions.
        anxietyTypes, relevantSurveyAnswers = self.modelInfoClass.extractFinalLabels(currentSurveyAnswersList, [])

        # For each stress type: positive and negative
        for anxietyTypeInd in range(len(anxietyTypes)):
            # if anxietyTypeInd < 2: continue
            anxietyType = anxietyTypes[anxietyTypeInd]

            # Extract the stress score (+/-) from the survey answers.
            stressScores = relevantSurveyAnswers[anxietyTypeInd]

            # Do not waste time if the analysis was performed already.
            saveDataFolder = self.saveDataFolder + folderName + anxietyType + "/"
            if not self.overwrite and os.path.exists(saveDataFolder):
                continue
            # Create/verify a directory to save the figures
            os.makedirs(saveDataFolder, exist_ok=True)

            # For each feature collected
            for featureInd in range(len(allFinalFeatures[0])):
                fig = plt.figure()
                ax = plt.gca()
                # Plot each feature against its stress score
                features = allFinalFeatures[:, featureInd]

                # Plot subjects as different colors
                if len(subjectOrder) != 0:
                    prevSubject = ""
                    colorInd = -1

                    x = []
                    y = []
                    for newSubjectInd in range(len(subjectOrder)):
                        newSubject = subjectOrder[newSubjectInd]
                        if newSubject != prevSubject:
                            if len(set(x)) > 3:
                                self.addBestFit(x, y, False, self.colorList[colorInd])
                            x = [features[newSubjectInd]]
                            y = [stressScores[newSubjectInd]]

                            colorInd = colorInd + 1 if colorInd + 1 < len(self.colorList) else 0
                            plt.plot(features[newSubjectInd], stressScores[newSubjectInd], 'o', c=self.colorList[colorInd], markersize=3, label=newSubject)
                        else:
                            x.append(features[newSubjectInd])
                            y.append(stressScores[newSubjectInd])

                            plt.plot(features[newSubjectInd], stressScores[newSubjectInd], 'o', c=self.colorList[colorInd], markersize=3)
                        prevSubject = newSubject

                else:
                    plt.plot(features, stressScores, 'o', c=self.colorList[0], markersize=3)

                # Plot the linear best fit
                if len(stressScores) > 2:
                    self.addBestFit(features, stressScores)

                # Add figure information
                ax.set_ylabel(anxietyType + " Scores")
                ax.set_xlabel(featureNames[featureInd])
                ax.set_title("Feature Correlation with " + anxietyType)
                # Save and clear the figure
                fig.savefig(saveDataFolder + featureNames[featureInd] + ".png", dpi=300, bbox_inches='tight')
                self.clearFigure(fig, legend=None, showPlot=True)

    def correlationMatrix(self, featureList, featureNames, folderName="correlationMatrix/"):
        print("Plotting the Correlation Matrix Amongst the Features")
        # Create/verify a directory to save the figures
        saveDataFolder = self.saveDataFolder + folderName
        os.makedirs(saveDataFolder, exist_ok=True)

        # Perform Deepcopy to Not Edit Features
        signalData = deepcopy(np.asarray(featureList))
        signalLabels = deepcopy(np.asarray(featureNames))

        # Standardize the Feature
        for i in range(len(signalData[0])):
            signalData[:, i] = (signalData[:, i] - np.mean(signalData[:, i])) / np.std(signalData[:, i], ddof=1)

        matrix = np.asarray(np.corrcoef(signalData.T))
        ax = sns.heatmap(matrix, cmap='icefire', xticklabels=signalLabels, yticklabels=signalLabels)
        # Save and clear the figure
        sns.set(rc={'figure.figsize': (18, 18)})
        fig = ax.get_figure()
        fig.savefig(saveDataFolder + "correlationMatrixFull.png", dpi=300)
        fig.show()

        # Cluster the Similar Features
        signalLabelsX = deepcopy(signalLabels)
        signalLabelsY = deepcopy(signalLabels)
        for i in range(1, len(matrix)):
            signalLabelsX = signalLabelsX[matrix[:, i].argsort()]
            matrix = matrix[matrix[:, i].argsort()]
        for i in range(1, len(matrix[0])):
            signalLabelsY = signalLabelsY[matrix[i].argsort()]
            matrix = matrix[:, matrix[i].argsort()]
        # Plot the New Cluster
        ax = sns.heatmap(matrix, cmap='icefire', xticklabels=signalLabelsX, yticklabels=signalLabelsY)
        # Save and clear the figure
        sns.set(rc={'figure.figsize': (25, 15)})
        fig = ax.get_figure()
        fig.savefig(saveDataFolder + "correlationMatrixSorted.png", dpi=300)
        self.clearFigure(fig, legend=None, showPlot=True)

        # Remove Small Correlations
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if abs(matrix[i][j]) < 0.96:
                    matrix[i][j] = 0
        # Plot the New Correlations
        ax = sns.heatmap(matrix, cmap='icefire', xticklabels=signalLabelsX, yticklabels=signalLabelsY)
        # Save and clear the figure
        sns.set(rc={'figure.figsize': (25, 15)})
        fig = ax.get_figure()
        fig.savefig(saveDataFolder + "correlationMatrixSortedCull.png", dpi=300)
        self.clearFigure(fig, legend=None, showPlot=True)

    def featureComparison(self, featureList1, featureList2, featureLabels, featureNames, xChemical, yChemical):
        # Create/verify a directory to save the figures
        saveDataFolder = self.saveDataFolder + "chemicalFeatureComparison/"
        os.makedirs(saveDataFolder, exist_ok=True)

        featureList1 = np.asarray(featureList1.copy())
        featureList2 = np.asarray(featureList2.copy())

        labelList = ['Cold', 'Exercise', 'VR']

        for featureInd1 in range(len(featureNames)):

            features1 = featureList1[:, featureInd1]

            for featureInd2 in range(len(featureNames)):
                features2 = featureList2[:, featureInd2]

                fig = plt.figure()
                ax = plt.gca()
                for ind in range(len(featureLabels)):
                    labelInd = featureLabels[ind]
                    ax.plot(features1[ind], features2[ind], 'o', c=self.colorList[labelInd], label=labelList[labelInd])

                ax.set_xlabel(xChemical + ": " + featureNames[featureInd1])
                ax.set_ylabel(yChemical + ": " + featureNames[featureInd2])
                ax.set_title("Feature Comparison")
                ax.legend()
                # Save and clear the figure
                fig.savefig(saveDataFolder + featureNames[featureInd1] + "_" + featureNames[featureInd2] + ".png", dpi=300, bbox_inches='tight')
                self.clearFigure(fig, legend=None, showPlot=True)

    def singleFeatureComparison(self, featureListFull, featureLabelFull, chemicalOrder, featureNames):
        # Create/verify a directory to save the figures
        saveDataFolder = self.saveDataFolder + "singleChemicalFeatureComparison/"
        os.makedirs(saveDataFolder, exist_ok=True)

        for chemicalInd in range(len(chemicalOrder)):
            chemicalName = chemicalOrder[chemicalInd]
            featureList = featureListFull[chemicalInd]
            featureLabels = featureLabelFull[chemicalInd]

            saveDataFolderChemical = saveDataFolder + chemicalName + "/"
            os.makedirs(saveDataFolderChemical, exist_ok=True)

            for featureInd in range(len(featureNames)):

                features = featureList[:, featureInd]

                fig = plt.figure()
                ax = plt.gca()
                for ind in range(len(featureLabels)):
                    labelInd = featureLabels[ind]
                    ax.plot(features[ind], [0], self.colorList[labelInd])

                ax.set_xlabel(chemicalName + ": " + featureNames[featureInd])
                ax.set_ylabel("Constant")
                ax.set_title("Feature Comparison")
                # plt.legend()
                # Save and clear the figure
                fig.savefig(saveDataFolderChemical + featureNames[featureInd] + ".png", dpi=300, bbox_inches='tight')
                self.clearFigure(fig, legend=None, showPlot=True)

    def featureDistribution(self, allFinalFeatures, finalLabels, featureNames, labelType="Stress Score", folderName="featureDistributions/"):
        print("Plotting Feature Distributions for " + labelType)
        classDistribution = collections.Counter(finalLabels)
        print("\tClass Distribution:", classDistribution)
        print("\tNumber of Unique Points = ", len(classDistribution))

        # Create/verify a directory to save the figures
        saveDataFolder = self.saveDataFolder + folderName + labelType + "/"
        os.makedirs(saveDataFolder, exist_ok=True)

        allFinalFeatures = np.asarray(allFinalFeatures.copy())
        finalLabels = np.asarray(finalLabels.copy())
        for featureInd in range(len(featureNames)):
            fig = plt.figure()

            for label in classDistribution.keys():
                features = allFinalFeatures[:, featureInd][finalLabels == label]

                plt.hist(features, bins=min(100, len(features)), alpha=0.5, label=label, align='mid', density=True)

            plt.legend()
            plt.ylabel(featureNames[featureInd])
            fig.savefig(saveDataFolder + featureNames[featureInd] + ".png", dpi=300, bbox_inches='tight')
            self.clearFigure(fig, legend=None, showPlot=True)

    @staticmethod
    def stressLabelPlotting(experimentalOrder, subjectOrder, featureLabelTypes, allFinalLabels):
        bounds = compileModelInfo().predictionBounds

        colors = []
        currentSubjectName = ""
        subjectExperimentInds = []
        for experimentInd in range(len(experimentalOrder)):
            subjectName = subjectOrder[experimentInd]

            if (currentSubjectName != subjectName and len(subjectExperimentInds)) != 0 or experimentInd == len(experimentalOrder) - 1:
                if experimentInd == len(experimentalOrder) - 1:
                    subjectExperimentInds.append(experimentInd)
                    colors.append('#333333')

                for finalLabelInd in range(len(featureLabelTypes)):
                    finalLabel = featureLabelTypes[finalLabelInd]
                    experimentNames = [experimentalOrder[i] for i in subjectExperimentInds]
                    plt.figure(figsize=(12, 6))  # Increase the figure size for better readability
                    bar_positions = np.arange(len(experimentNames))
                    bars = plt.bar(bar_positions, [allFinalLabels[finalLabelInd][i] for i in subjectExperimentInds], color=colors)

                    for bar in bars:
                        yval = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')

                    plt.xticks(ticks=bar_positions, labels=experimentNames, rotation=45, fontsize=12, ha='right')
                    plt.title(f'{currentSubjectName} - {finalLabel}', fontsize=16)
                    plt.xlabel("Experiment Number", fontsize=14)
                    plt.ylabel("Label Value", fontsize=14)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.ylim(bounds[finalLabelInd])
                    plt.tight_layout()
                    plt.show()

                colors = []
                subjectExperimentInds = []
            experimentName = experimentalOrder[experimentInd]
            subjectExperimentInds.append(experimentInd)
            currentSubjectName = subjectName

            colors.append('#333333')
            if 'cpt' in experimentName.lower():
                colors[-1] = 'skyblue'
            if 'heat' in experimentName.lower():
                colors[-1] = '#D62728'
