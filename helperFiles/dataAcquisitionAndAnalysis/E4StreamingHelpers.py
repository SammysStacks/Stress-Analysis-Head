import collections
import numpy as np
import os

# Import Bioelectric Analysis Files
from .biolectricProtocols.eogAnalysis import eogProtocol
from .biolectricProtocols.eegAnalysis import eegProtocol
from .biolectricProtocols.ecgAnalysis import ecgProtocol
from .biolectricProtocols.edaAnalysis import edaProtocol
from .biolectricProtocols.emgAnalysis import emgProtocol
from .biolectricProtocols.temperatureAnalysis import tempProtocol
from .biolectricProtocols.generalAnalysis_lowFreq import generalProtocol_lowFreq
from .biolectricProtocols.generalAnalysis_highFreq import generalProtocol_highFreq


class E4StreamingHelpers:
    def __init__(self, server_address='127.0.0.1', server_port=28000, device_id='B516C6',
                 buffer_size=4096, output_file="E4_data.xlsx", plotStreamedData=True):
        self.server_address = server_address
        self.server_port = server_port
        self.device_id = device_id
        self.buffer_size = buffer_size
        self.output_file = os.path.join(os.getcwd(), output_file)
        self.plotStreamedData = plotStreamedData
