
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------- #
# ------------------------- Real-Time Plotting Head ------------------------ #

class globalPlottingProtocols:
    
    def __init__(self):
        # Setup matplotlib
        plt.rcdefaults()
        plt.ioff() # prevent memory leaks; Reverse: plt.ion()
    
    def heatmap(self, data, saveDataPath = None, title = None, xlabel = None, ylabel = None):
        # Plot the heatmap
        ax = sns.heatmap(data, robust = True, cmap='icefire')
        # Save the Figure
        sns.set(rc={'figure.figsize':(7, 9)})
        if title: ax.set_title(title)
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel) 
        fig = ax.get_figure(); 
        if self.saveDataFolder and saveDataPath:
            fig.savefig(f"{saveDataPath}.pdf", dpi=500, bbox_inches='tight')
            fig.savefig(f"{saveDataPath}.png", dpi=500, bbox_inches='tight')
        self.clearFigure(fig, None)
        plt.rcdefaults()
        
    def clearFigure(self, fig = None, legend = None):
        if legend != None: legend.remove()
        if fig: fig.clear(); plt.close(fig);
        # Clear plots
        plt.show(); plt.close('all')
        plt.cla(); plt.clf()
        plt.rcdefaults()
        
    def _createFolder(self, filePath):
        # Create the folders if they do not exist.
        os.makedirs(os.path.dirname(filePath), exist_ok=True) 
        
    def saveFigure(self, saveFigurePath):
        # Create a folder for saving the file.
        self._createFolder(saveFigurePath)
        
        # Save the figure
        plt.savefig(saveFigurePath)
        self.clearFigure()