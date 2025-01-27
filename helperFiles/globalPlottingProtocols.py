import shutil
import time

import matplotlib.pyplot as plt
import seaborn as sns
import os


class globalPlottingProtocols:

    def __init__(self):
        # Setup matplotlib
        self.baseFolderName = "_basePlots/"
        plt.rcdefaults()
        plt.ion()

        # Specify the color order.
        self.lightColors = ["#F17FB1", "#5DCBF2", "#B497C9", "#90D6AD", "#FFC162", "#231F20"]  # Red, Blue, Purple, Green, Orange, Grey
        self.darkColors = ["#F3757A", "#489AD4", "#7E71B4", "#50BC84", "#F9A770", "#4A4546"]  # Red, Blue, Purple, Green, Orange, Grey
        self.blackColor = "#231F20"

        # Set the saving folder
        self.baseSavingDataFolder = None
        self.saveDataFolder = None
        self.datasetName = None

    def setSavingFolder(self, baseSavingDataFolder, stringID, datasetName):
        self.baseSavingDataFolder = baseSavingDataFolder + self.baseFolderName
        self.saveDataFolder = baseSavingDataFolder + stringID
        self.datasetName = datasetName

        if baseSavingDataFolder:
            self._createFolder(self.baseSavingDataFolder)
            if stringID: self._createFolder(self.saveDataFolder)

    @staticmethod
    def _createFolder(filePath):
        if filePath: os.makedirs(os.path.dirname(filePath), exist_ok=True)

    @staticmethod
    def clearFigure(fig=None, legend=None, showPlot=True):
        if showPlot: plt.show()  # Ensure the plot is displayed

        # Clear and close the figure/legend if provided
        if legend is not None: legend.remove()
        if fig: fig.clear();  plt.close(fig)
        else: plt.cla(); plt.clf()
        plt.close('all')

    def displayFigure(self, saveFigureLocation, saveFigureName, baseSaveFigureName=None, fig=None, showPlot=True, clearFigure=True):
        self._createFolder(self.saveDataFolder + saveFigureLocation)
        if fig is None: fig = plt.gcf()

        # Save to base location if specified
        if baseSaveFigureName is not None:
            base_path = os.path.join(self.baseSavingDataFolder, f"{self.datasetName} {baseSaveFigureName}")
            fig.savefig(base_path, transparent=True, dpi=300)

            # Copy the saved figure to the second location
            shutil.copy(base_path, os.path.join(self.saveDataFolder, f"{saveFigureLocation}{saveFigureName}"))
        else: fig.savefig(os.path.join(self.saveDataFolder, f"{saveFigureLocation}{saveFigureName}"), transparent=True, dpi=300)

        if clearFigure: self.clearFigure(fig=fig, legend=None, showPlot=showPlot)  # Clear the figure after saving
        elif showPlot: plt.show()

    def heatmap(self, data, saveDataPath=None, title=None, xlabel=None, ylabel=None):
        # Plot the heatmap
        ax = sns.heatmap(data, robust=True, cmap='icefire')
        # Save the Figure
        sns.set(rc={'figure.figsize': (7, 9)})
        if title: ax.set_title(title)
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)
        fig = ax.get_figure()
        if self.saveDataFolder and saveDataPath:
            fig.savefig(f"{saveDataPath}.pdf", dpi=500, bbox_inches='tight')
            fig.savefig(f"{saveDataPath}.png", dpi=500, bbox_inches='tight')
        self.clearFigure(fig, legend=None, showPlot=True)
        plt.rcdefaults()
