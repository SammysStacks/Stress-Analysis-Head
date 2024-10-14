# adapted from https://github.com/HectorCarral/Empatica-E4-LSL
import socket
import time
import matplotlib

if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
import pandas as pd
from collections import deque
import os

import matplotlib.pyplot as plt
from .empaticaInterface import empaticaInterface


class E4Streaming(empaticaInterface):
    def __init__(self, server_address='127.0.0.1', server_port=28000, device_id='B516C6',
                 buffer_size=4096, output_file="E4_data.xlsx", plotStreamedData=True):
        super().__init__(server_address=server_address, server_port=server_port, device_id=device_id, buffer_size=buffer_size, output_file=output_file, plotStreamedData=plotStreamedData)

        # 100 points for real-time plotting
        self.s = None
        self.acc_data = deque(maxlen=100)
        self.bvp_data = deque(maxlen=100)
        self.gsr_data = deque(maxlen=100)
        self.tmp_data = deque(maxlen=100)
        self.time_stamps_acc = deque(maxlen=100)
        self.time_stamps_bvp = deque(maxlen=100)
        self.time_stamps_gsr = deque(maxlen=100)
        self.time_stamps_tmp = deque(maxlen=100)

        # Initialize start_time as None
        self.start_time_acc = None
        self.start_time_bvp = None
        self.start_time_gsr = None
        self.start_time_tmp = None
        self.stream_experiment_time = None

        # DataFrames for saving to Excel sheets， need changes later to interface with questionnaires
        self.acc_df = pd.DataFrame(columns=['Timestamp', 'ACC_X', 'ACC_Y', 'ACC_Z'])
        self.bvp_df = pd.DataFrame(columns=['Timestamp', 'BVP'])
        self.gsr_df = pd.DataFrame(columns=['Timestamp', 'GSR'])
        self.tmp_df = pd.DataFrame(columns=['Timestamp', 'Temp'])

        # Initialize plots only if plotting is enabled
        if self.plotStreamedData:
            print("Plotting enabled. Initializing plots...")
            plt.ion()  # Enable interactive mode
            self.fig, self.axs = plt.subplots(4, 1, figsize=(12, 10))
            self.acc_lines = [self.axs[0].plot([], [], label="ACC_X")[0],
                              self.axs[0].plot([], [], label="ACC_Y")[0],
                              self.axs[0].plot([], [], label="ACC_Z")[0]]
            self.bvp_line = self.axs[1].plot([], [], label="BVP", color='purple')[0]
            self.gsr_line = self.axs[2].plot([], [], label="GSR", color='orange')[0]
            self.tmp_line = self.axs[3].plot([], [], label="Temp", color='cyan')[0]

            # Setup axis labels and titles
            self.setup_plots()

    def setup_plots(self):
        # Only set up plots if plotting is enabled
        if self.plotStreamedData:
            self.axs[0].set_title("3-axis Acceleration")
            self.axs[0].set_ylabel("Acceleration (g)")
            self.axs[0].legend()

            self.axs[1].set_title("Blood Volume Pulse (BVP)")
            self.axs[1].set_ylabel("BVP (AU)")

            self.axs[2].set_title("Galvanic Skin Response (GSR)")
            self.axs[2].set_ylabel("GSR (µS)")

            self.axs[3].set_title("Temperature (Temp)")
            self.axs[3].set_ylabel("Temp (°C)")
            self.axs[3].set_xlabel("Time (s)")



    def update_plots(self):
        # Only update plots if plotting is enabled
        if not self.plotStreamedData:
            return  # Skip plotting if disabled

        if len(self.time_stamps_acc) == len(self.acc_data):
            for i in range(3):
                self.acc_lines[i].set_data(self.time_stamps_acc, [d[i] for d in self.acc_data])  # ACC X, Y, Z axes
            self.axs[0].relim()
            self.axs[0].autoscale_view()

        if len(self.time_stamps_bvp) == len(self.bvp_data):
            self.bvp_line.set_data(self.time_stamps_bvp, self.bvp_data)
            self.axs[1].relim()
            self.axs[1].autoscale_view()

        if len(self.time_stamps_gsr) == len(self.gsr_data):
            self.gsr_line.set_data(self.time_stamps_gsr, self.gsr_data)
            self.axs[2].relim()
            self.axs[2].autoscale_view()

        if len(self.time_stamps_tmp) == len(self.tmp_data):
            self.tmp_line.set_data(self.time_stamps_tmp, self.tmp_data)
            self.axs[3].relim()
            self.axs[3].autoscale_view()

        # avoid overlapping of labels and titles and crashing
        try:
            plt.tight_layout()  # Prevent overlapping of labels and titles
            plt.draw()
            plt.pause(0.001)  # Allow the plot to update
        except Exception as e:
            print(f"Error during plotting: {e}")

    def save_to_excel(self):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the folder structure based on the script directory
        experimental_data_folder = os.path.join(script_dir, "_experimentalData")
        e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")

        # Check and create directories if they don't exist
        os.makedirs(e4_watch_data_folder, exist_ok=True)

        # Save the file in the e4WatchData folder
        output_path = os.path.join(e4_watch_data_folder, self.output_file)
        print(f"Saving data to Excel at {output_path}...")

        with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
            self.acc_df.to_excel(writer, sheet_name='ACC', index=False)
            self.bvp_df.to_excel(writer, sheet_name='BVP', index=False)
            self.gsr_df.to_excel(writer, sheet_name='GSR', index=False)
            self.tmp_df.to_excel(writer, sheet_name='Temp', index=False)

        with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
            self.acc_df.to_excel(writer, sheet_name='ACC', index=False)
            self.bvp_df.to_excel(writer, sheet_name='BVP', index=False)
            self.gsr_df.to_excel(writer, sheet_name='GSR', index=False)
            self.tmp_df.to_excel(writer, sheet_name='Temp', index=False)

    def stream(self):
        try:
            print("Streaming...")
            while True:
                response = self.s.recv(self.buffer_size).decode("utf-8")
                if "connection lost to device" in response:
                    print(response)
                    break

                # Plot only if enabled
                if self.plotStreamedData:
                    self.update_plots()

        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
        finally:
            self.save_to_excel()
            self.s.send("device_disconnect\r\n".encode())
            self.s.close()

    def update_data_frames(self, data_row, stream_type):
        data_df = pd.DataFrame([data_row])  # Store normalized timestamp directly

        if stream_type == "E4_Acc":
            self.acc_df = pd.concat([self.acc_df, data_df], ignore_index=True)
        elif stream_type == "E4_Bvp":
            self.bvp_df = pd.concat([self.bvp_df, data_df], ignore_index=True)
        elif stream_type == "E4_Gsr":
            self.gsr_df = pd.concat([self.gsr_df, data_df], ignore_index=True)
        elif stream_type == "E4_Temperature":
            self.tmp_df = pd.concat([self.tmp_df, data_df], ignore_index=True)

    def getCurrentTime(self):
        if self.stream_experiment_time is None:
            print("E4 streaming has not started yet.")
            return None
        # Calculate the elapsed time since the start of streaming
        elapsed_time = time.perf_counter() - self.stream_experiment_time
        return elapsed_time
