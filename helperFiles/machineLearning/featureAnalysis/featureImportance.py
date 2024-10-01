import os
import numpy as np
import pandas as pd

# Pytorch
import torch

# Plotting
import matplotlib.pyplot as plt

# Feature importance
import shap


class featureImportance:

    def __init__(self, saveAnalysisFolder):
        self.saveAnalysisFolder = saveAnalysisFolder

    # ---------------------------- SHAP Analysis --------------------------- #

    def shapAnalysis(self, model, featureData, featureLabels, featureNames=None, modelType="", shapSubfolder=""):
        """
        Randomly Permute a Feature's Column and Return the Average Deviation in the Score: |oldScore - newScore|
        NOTE: ONLY Compare Feature on the Same Scale: Time and Distance CANNOT be Compared
        """
        print("\tEntering SHAP Analysis")
        # Assert the correct input format.
        featureNames = np.asarray(featureNames)

        # Assert the correct input format.
        if isinstance(featureData, torch.Tensor):
            featureData = featureData.detach().numpy()
        else:
            featureData = np.asarray(featureData)

        # Make an output folder for SHAP analysis figures.
        saveShapAnalysisFolder = self.saveAnalysisFolder + "/SHAP Analysis/" + shapSubfolder
        os.makedirs(saveShapAnalysisFolder, exist_ok=True)

        # ---------------------- Calculate SHAP values --------------------- #

        # Perform SHAP analysis using a gneral explainer.
        masker = shap.maskers.Independent(data=featureData)  # Define a masker
        explainerGeneral = shap.Explainer(model=model.predict, masker=masker, feature_names=featureNames, algorithm="auto")
        shap_valuesGeneral = explainerGeneral(featureData, max_evals=30101)

        if modelType == "RF":
            # Perform SHAP analysis using a tree explainer.
            explainer = shap.TreeExplainer(model.predict)
            shap_values = explainer.shap_values(featureData)
        else:
            # Perform SHAP analysis using a kernel explainer.
            explainer = shap.KernelExplainer(model.predict, data=featureData, feature_names=featureNames)
            shap_values = explainer.shap_values(featureData, nsamples=featureData.shape[0])

        # ----------------------- Plot SHAP Analysis ----------------------- #
        # Specify Indivisual Sharp Parameters
        dataPoint = 0
        featurePoint = 5
        explainer.expected_value = 0

        # Summary Plot
        name = "Summary Plot"
        summaryPlot = plt.figure()
        shap.summary_plot(shap_values, features=featureData, feature_names=featureNames, max_display=None, title=None, alpha=1, show=True, sort=True, color_bar=True, plot_size='auto', class_names=None)
        summaryPlot.savefig(saveShapAnalysisFolder + "summary_plot.png", bbox_inches='tight', dpi=300)
        plt.show()

        # Dependance Plot
        name = "Dependance Plot"
        dependancePlot, dependanceAX = plt.subplots()
        shap.dependence_plot(featurePoint, shap_values[0], features=featureData, feature_names=featureNames, ax=dependanceAX)
        dependancePlot.savefig(saveShapAnalysisFolder + "dependance_plot.png", bbox_inches='tight', dpi=300)
        plt.show()

        # Indivisual Force Plot
        name = "Indivisual Force Plot"
        forcePlot = shap.force_plot(explainer.expected_value, shap_values[0][dataPoint, :], features=np.round(featureData[dataPoint, :], 5), feature_names=featureNames, matplotlib=True, show=False)
        forcePlot.savefig(saveShapAnalysisFolder + "indivisual_force_plot.png", bbox_inches='tight', dpi=300)
        plt.show()

        # Full Force Plot. NOTE: CANNOT USE matplotlib = True to See
        #name = "Full Force Plot"
        #fullForcePlot = plt.figure()
        #fullForcePlot = shap.force_plot(explainer.expected_value, shap_values[0], features = featureData, feature_names = featureNames, matplotlib = False, show = True)
        #fullForcePlot.savefig(saveShapAnalysisFolder + "full_force_plot.png", bbox_inches='tight', dpi=300)

        # WaterFall Plot
        name = "Waterfall Plot"
        waterfallPlot = plt.figure()
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0][dataPoint], feature_names=featureNames, show=True)
        #shap.plots.waterfall(shap_valuesGeneral[dataPoint], show = True)
        waterfallPlot.savefig(saveShapAnalysisFolder + "waterfall_plot.png", bbox_inches='tight', dpi=300)
        plt.show()

        # Decision Plot
        name = "Decision Plot"
        decisionPlotOne = plt.figure()
        shap.decision_plot(explainer.expected_value, shap_values[0], features=featureData, feature_names=None, feature_order="importance")
        decisionPlotOne.savefig(saveShapAnalysisFolder + "Decision Plot.png", bbox_inches='tight', dpi=300)
        plt.show()

        # Bar Plot
        name = "Bar Plot"
        barPlot = plt.figure()
        shap.plots.bar(shap_valuesGeneral, show=True)
        barPlot.savefig(saveShapAnalysisFolder + "Bar Plot.png", bbox_inches='tight', dpi=300)
        plt.show()

        # HeatMap Plot
        name = "Heatmap Plot"
        heatmapPlot = plt.figure()
        shap.plots.heatmap(shap_valuesGeneral, show=True, instance_order=shap_valuesGeneral.sum(1))
        heatmapPlot.savefig(saveShapAnalysisFolder + "Heatmap Plot.png", bbox_inches='tight', dpi=300)
        plt.show()

        # Monitoring Plot (The Function is a Beta Test As of 11-2021)
        if featureData.shape[1] > 150:  # They Skip Every 50 Points I Believe
            name = "Monitor Plot"
            monitorPlot = plt.figure()
            shap.monitoring_plot(featurePoint, shap_values[0], features=featureData, feature_names=None)
            monitorPlot.savefig(saveShapAnalysisFolder + "Monitor Plot.png", bbox_inches='tight', dpi=300)
            plt.show()

        # ## NOT WORKIMNG

        # # Indivisual Decision Plot
        # misclassified = abs(shapFeatureLabels - model.predict(featureData)) < 10
        # decisionFolder = saveShapAnalysisFolder + "Decision Plots/"
        # os.makedirs(decisionFolder, exist_ok=True) 
        # #for dataPoint1 in range(len(featureData)):
        # #    name = "Indivisual Decision Plot DataPoint Num " + str(dataPoint1)
        # #    decisionPlot = plt.figure()
        # #    shap.decision_plot(explainer.expected_value, shap_values[0][dataPoint1,:], features = featureData[dataPoint1,:], feature_order = "importance")
        # #    decisionPlot.savefig(saveShapAnalysisFolder + f"waterfall_plot_num{dataPoint1}.png", bbox_inches='tight', dpi=300)

        # # Scatter Plot
        # scatterFolder = saveShapAnalysisFolder + "Scatter Plots/"
        # os.makedirs(scatterFolder, exist_ok=True)
        # for featurePoint1 in range(shap_valuesGeneral.shape[1]):
        #     for featurePoint2 in range(shap_valuesGeneral.shape[1]):
        #         scatterPlot, scatterAX = plt.subplots()
        #         shap.plots.scatter(shap_valuesGeneral[:, featurePoint1], color = shap_valuesGeneral[:, featurePoint2], ax = scatterAX)
        #         scatterPlot.savefig(saveShapAnalysisFolder + f"Scatter Plot F{featurePoint1} F{featurePoint2}.png", bbox_inches='tight', dpi=300)

    # ---------------------------------------------------------------------- #
    # ------------------------- Deprecated Analyses ------------------------ #

    # DEPRECATED
    def plotImportance(self, perm_importance_result, featureLabels, name="Relative Feature Importance"):
        """ bar plot the feature importance """

        fig, ax = plt.subplots()

        indices = perm_importance_result['importances_mean'].argsort()
        plt.barh(range(len(indices)),
                 perm_importance_result['importances_mean'][indices],
                 xerr=perm_importance_result['importances_std'][indices])

        ax.set_yticks(range(len(indices)))
        if len(featureLabels) != 0:
            _ = ax.set_yticklabels(np.asarray(featureLabels)[indices])
        #      headers = np.asarray(featureLabels)[indices]
        #      for i in headers:
        #          print('%s Weight: %.5g' % (str(i),v))
        plt.savefig(self.saveDataFolder + name + " " + modelType + ".png", dpi=150, bbox_inches='tight')

    # DEPRECATED
    def randomForestImportance(self, featureNames=[]):
        # get importance
        importance = self.predictionModel.model.feature_importances_
        # summarize feature importance
        for i, v in enumerate(importance):
            if len(featureNames) != 0:
                i = featureNames[i]
                print('%s Weight: %.5g' % (str(i), v))
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        freq_series = pd.Series(importance)
        ax = freq_series.plot(kind="bar")

        # Specify Figure Aesthetics
        ax.set_title("Feature Importance in Model")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Feature Importance")

        # Set X-Labels
        if len(featureNames) != 0:
            ax.set_xticklabels(featureNames)
            self.add_value_labels(ax)
        # Show Plot
        name = "Feature Importance"
        plt.savefig(self.saveDataFolder + name + " " + modelType + ".png", dpi=150, bbox_inches='tight')
        plt.show()

    def add_value_labels(self, ax, spacing=5):
        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.3f}".format(y_value)

            # Create annotation
            ax.annotate(
                label,  # Use `label` as label
                (x_value, y_value),  # Place label at end of the bar
                xytext=(0, space),  # Vertically shift label by `space`
                textcoords="offset points",  # Interpret `xytext` as offset in points
                ha='center',  # Horizontally center label
                va=va)  # Vertically align label differently for
            # positive and negative values.
