# ---------------------------- Imported Modules ---------------------------- #

# Import machine learning files
from .signalEncoderModules import signalEncoderModules


class changeVariance(signalEncoderModules):

    def __init__(self):
        super(changeVariance, self).__init__()
        # Map the initial signals into a common subspace.
        self.adjustSignals = self.varianceTransformation(inChannel=1)
        self.removeSignalAdjustment = self.varianceTransformation(inChannel=1)

    def adjustSignalVariance(self, inputData):
        return self.encodingInterface(inputData, self.adjustSignals)

    def unAdjustSignalVariance(self, inputData):
        return self.encodingInterface(inputData, self.removeSignalAdjustment)
