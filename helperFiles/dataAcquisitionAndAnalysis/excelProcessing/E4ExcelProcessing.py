import os
import pandas as pd

class e4ExcelProcessing:
    def __init__(self, date, user, experiment, biomarker, script_dir):
        self.date = date
        self.user = user
        self.experiment = experiment
        self.biomarker = biomarker
        self.script_dir = script_dir
        self.experimental_data_folder = os.path.join(self.script_dir, "_experimentalData")
        self.e4_watch_data_folder = os.path.join(self.experimental_data_folder, "e4WatchData")
        self.excel_file = os.path.join(self.e4_watch_data_folder, f"{self.date}_{self.user}_E4_{self.experiment}.xlsx")
        self.excel_data = pd.read_excel(self.excel_file, sheet_name=None)

    def getExcelData(self):
        if self.biomarker == "Acceleration":
            excel_data = self.excel_data['ACC']
            acc_x = excel_data['ACC_X'].values
            acc_y = excel_data['ACC_Y'].values
            acc_z = excel_data['ACC_Z'].values
            return acc_x, acc_y, acc_z
        elif self.biomarker == "bvp":
            excel_data = self.excel_data['BVP']
            timePointer = excel_data['Timestamp'].values
            bvp = excel_data['BVP'].values
            return timePointer, bvp
        elif self.biomarker == "GSR":
            excel_data = self.excel_data['GSR']
            gsr = excel_data['GSR'].values
            return gsr
        elif self.biomarker == "Temp":
            excel_data = self.excel_data['Temp']
            temp = excel_data['Temp'].values
            return temp
        else:
            return None

    def save_features_to_excel(self, time_vector, features_all_windows_normalized, feature_names, window_sizes):
        # Create directory structure for saving Excel file
        feature_analysis_excel_folder = os.path.join(self.e4_watch_data_folder, "featureAnalysisExcel")
        user_feature_excel_data_folder = os.path.join(feature_analysis_excel_folder, f'{self.date}_{self.user}_FeatureExcelData')

        if not os.path.exists(feature_analysis_excel_folder):
            os.makedirs(feature_analysis_excel_folder)
        if not os.path.exists(user_feature_excel_data_folder):
            os.makedirs(user_feature_excel_data_folder)

        # Save each window size features to separate sheets in Excel
        excel_save_path = os.path.join(user_feature_excel_data_folder, f'{self.biomarker}_{self.experiment}_normalized_windowed_features.xlsx')
        with pd.ExcelWriter(excel_save_path) as writer:
            for i, window_size in enumerate(window_sizes):
                time_col_name = f'Time_{window_size}s'
                features_df = pd.DataFrame(features_all_windows_normalized[i], columns=feature_names)
                features_df.insert(0, time_col_name, time_vector[i])
                features_df.to_excel(writer, sheet_name=f'{window_size}s_Window', index=False)

        print(f"Woohoo, Windowed and normalized features saved to {excel_save_path}")


