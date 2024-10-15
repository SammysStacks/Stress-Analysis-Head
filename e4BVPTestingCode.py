import pandas as pd
import os
import numpy as np
from helperFiles.dataAcquisitionAndAnalysis.biolectricProtocols.bvpAnalysis import bvpProtocol

if __name__ == "__main__":

    # user-specific details
    date = "20240924"
    user = "Ruixiao"
    experiment = "Baseline"
    biomarker = "bBvp"

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experimental_data_folder = os.path.join(script_dir, "_experimentalData")
    e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")
    excel_file = os.path.join(e4_watch_data_folder, f"{date}_{user}_E4_{experiment}.xlsx")

    # Read the Excel file
    excel_data = pd.read_excel(excel_file, sheet_name='BVP')
    bvp_data = excel_data['BVP'].values
    timepoints = excel_data['Timestamp'].tolist()



    # Create the necessary input data for the bvpProtocol
    numPointsPerBatch = 3000
    moveDataFinger = 10
    channelIndices = [0]  # Assuming a single channel for BVP data
    plottingClass = None  # Replace with an appropriate plotting class if available
    readData = None  # Replace with appropriate data reading class if available

    # Initialize the bvpProtocol class
    bvp_protocol = bvpProtocol(numPointsPerBatch=numPointsPerBatch, moveDataFinger=moveDataFinger, channelIndices=channelIndices, plottingClass=plottingClass, readData=readData)

    # Set up dummy data to simulate streaming BVP data
    bvp_protocol.numChannels = 1
    bvp_protocol.channelData = np.expand_dims(bvp_data, axis=0)
    bvp_protocol.timepoints = timepoints
    bvp_protocol.samplingFreq = None  # Assuming regular intervals
    bvp_protocol.collectFeatures = True  # Enable feature collection
    bvp_protocol.plottingIndicator = True  # Enable plotting

    # Run data analysis on the entire dataset
    dataFinger = 0  # Start at the beginning of the dataset
    print('here')
    bvp_protocol.analyzeData(dataFinger)

    print(f"Detected features: {bvp_protocol.rawFeatures}")

