
from helperFiles.dataAcquisitionAndAnalysis.excelProcessing import E4ExcelProcessing
from helperFiles.dataAcquisitionAndAnalysis.biolectricProtocols import accelerationAnalysis_old
from helperFiles.dataAcquisitionAndAnalysis.biolectricProtocols import bvpAnalysis_old
import os
import pandas as pd
import numpy

if __name__ == '__main__':
    # User specific details
    date = "20240924"
    user = "Ruixiao"
    experiment = "Baseline"
    biomarker = "Acceleration"

    # Define the folder path
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize the E4ExcelProcessing class
    e4_excel = E4ExcelProcessing.e4ExcelProcessing(date, user, experiment, biomarker, script_dir)

    if biomarker == "Acceleration":
        print('3 axis acceleration analysis started')
        samplingFrequency = 32
        windowSizes = [5, 10, 30]
        acc_x, acc_y, acc_z = e4_excel.getExcelData()
        acc_x = numpy.array(acc_x)
        acc_y = numpy.array(acc_y)
        acc_z = numpy.array(acc_z)
        acc_analysis = accelerationAnalysis.accelerationProtocol(acc_x, acc_y, acc_z, samplingFrequency, windowSizes, overlap=0.25)
        time_vector, normalizedFeatures = acc_analysis.analyzeData()
        # Plot and save the plots
        feature_analysis_folder = os.path.join(e4_excel.e4_watch_data_folder, "featureAnalysis")
        accelerations_folder = os.path.join(feature_analysis_folder, f"{date}_{user}_{experiment}_{biomarker}_Features")
        plot_output_folder = os.path.join(accelerations_folder, f"{e4_excel.date}_{e4_excel.user}_{e4_excel.experiment}_{e4_excel.biomarker}_Plots")
        acc_analysis.plotting_and_saving_features(time_vector, normalizedFeatures, acc_analysis.featureNames, windowSizes, plot_output_folder)
        # save the features in excel
        e4_excel.save_features_to_excel(time_vector, normalizedFeatures, acc_analysis.featureNames, windowSizes)

    elif biomarker == "Bvp":
        print('BVP analysis started')
        samplingFrequency = 64
        windowSizes = [10, 30, 60]
        timePointer, bvp = e4_excel.getExcelData()
        timePointer = numpy.array(timePointer)
        bvp = numpy.array(bvp)
        bvp_analysis = bvpAnalysis.bvpProtocol(timePointer, bvp, samplingFrequency, windowSizes, overlap=0.25)
        time_vector, normalizedFeatures = bvp_analysis.analyzeData()
        # Plot and save the features
        feature_analysis_folder = os.path.join(e4_excel.e4_watch_data_folder, "featureAnalysis")
        bvp_folder = os.path.join(feature_analysis_folder, f"{date}_{user}_{experiment}_{biomarker}_Features")
        plot_output_folder = os.path.join(bvp_folder, f"{e4_excel.date}_{e4_excel.user}_{e4_excel.experiment}_{e4_excel.biomarker}_Plots")
        bvp_analysis.plotting_and_saving_features(time_vector, normalizedFeatures, bvp_analysis.featureNames, windowSizes, plot_output_folder)
        # save the features in excel
        e4_excel.save_features_to_excel(time_vector, normalizedFeatures, bvp_analysis.featureNames, windowSizes)
