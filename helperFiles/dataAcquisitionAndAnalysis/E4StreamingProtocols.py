import socket
import time
import matplotlib
matplotlib.use('TkAgg')  # set backend plotting to Tkinter
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import os


class E4Streaming:
    def __init__(self, server_address='127.0.0.1', server_port=28000, device_id='B516C6',
                 buffer_size=4096, output_file="E4_data.xlsx", plotStreamedData=True):
        self.server_address = server_address
        self.server_port = server_port
        self.device_id = device_id
        self.buffer_size = buffer_size
        self.output_file = os.path.join(os.getcwd(), output_file)
        self.plotStreamedData = plotStreamedData  # New argument to control plotting

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
        self.start_time = None

        # DataFrames for saving to Excel sheets
        self.acc_df = pd.DataFrame(columns=['Timestamp', 'ACC_X', 'ACC_Y', 'ACC_Z'])
        self.bvp_df = pd.DataFrame(columns=['Timestamp', 'BVP'])
        self.gsr_df = pd.DataFrame(columns=['Timestamp', 'GSR'])
        self.tmp_df = pd.DataFrame(columns=['Timestamp', 'Temp'])

        # Initialize plots only if plotting is enabled
        if self.plotStreamedData:
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
        # This method sets up the plot labels and titles
        # Only called if plotting is enabled
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

    def connect(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(3)

        print("Connecting to server")
        self.s.connect((self.server_address, self.server_port))
        print("Connected to server\n")

        print("Devices available:")
        self.s.send("device_list\r\n".encode())
        response = self.s.recv(self.buffer_size)
        print(response.decode("utf-8"))

        print("Connecting to device")
        self.s.send(("device_connect " + self.device_id + "\r\n").encode())
        response = self.s.recv(self.buffer_size)
        print(response.decode("utf-8"))

        print("Pausing data receiving")
        self.s.send("pause ON\r\n".encode())
        response = self.s.recv(self.buffer_size)
        print(response.decode("utf-8"))

        time.sleep(1)  # Stabilize connection

    def subscribe_to_data(self, acc=True, bvp=True, gsr=True, tmp=True):
        if acc:
            print("Subscribing to ACC")
            self.s.send(("device_subscribe acc ON\r\n").encode())
            response = self.s.recv(self.buffer_size)
            print(response.decode("utf-8"))

        if bvp:
            print("Subscribing to BVP")
            self.s.send(("device_subscribe bvp ON\r\n").encode())
            response = self.s.recv(self.buffer_size)
            print(response.decode("utf-8"))

        if gsr:
            print("Subscribing to GSR")
            self.s.send(("device_subscribe gsr ON\r\n").encode())
            response = self.s.recv(self.buffer_size)
            print(response.decode("utf-8"))

        if tmp:
            print("Subscribing to Temp")
            self.s.send(("device_subscribe tmp ON\r\n").encode())
            response = self.s.recv(self.buffer_size)
            print(response.decode("utf-8"))

        print("Resuming data receiving")
        self.s.send("pause OFF\r\n".encode())
        response = self.s.recv(self.buffer_size)
        print(response.decode("utf-8"))

    def update_plots(self):
        # Only update plots if plotting is enabled
        if not self.plotStreamedData:
            return  # Do nothing if plotting is disabled

        # Normalize time before plotting
        if self.start_time is not None:
            self.time_stamps_acc = [t - self.start_time for t in self.time_stamps_acc]
            self.time_stamps_bvp = [t - self.start_time for t in self.time_stamps_bvp]
            self.time_stamps_gsr = [t - self.start_time for t in self.time_stamps_gsr]
            self.time_stamps_tmp = [t - self.start_time for t in self.time_stamps_tmp]

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

        plt.tight_layout()  # Prevent overlapping of labels and titles
        plt.draw()
        plt.pause(0.001)  # Allow the plot to update

    def save_to_excel(self):
        print("Saving data to Excel...")
        with pd.ExcelWriter(self.output_file, engine='openpyxl', mode='w') as writer:
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

                samples = response.split("\n")
                for i in range(len(samples) - 1):
                    sample_data = samples[i].split()
                    if len(sample_data) < 3:
                        continue

                    stream_type = sample_data[0]
                    timestamp = float(sample_data[1].replace(',', '.'))

                    # Initialize start time on first sample
                    if self.start_time is None:
                        self.start_time = timestamp
                        print(f"Start time set to {self.start_time}")

                    # Append the raw time (normalization will happen in update methods)
                    if stream_type == "E4_Acc":
                        if len(sample_data) >= 5:
                            data = [int(sample_data[2].replace(',', '.')),
                                    int(sample_data[3].replace(',', '.')),
                                    int(sample_data[4].replace(',', '.'))]
                            self.acc_data.append(data)
                            self.time_stamps_acc.append(timestamp)  # Raw time, normalization happens later
                            data_row = {'ACC_X': data[0], 'ACC_Y': data[1], 'ACC_Z': data[2]}
                            self.update_data_frames(data_row, "E4_Acc")

                    elif stream_type == "E4_Bvp":
                        data = float(sample_data[2].replace(',', '.'))
                        self.bvp_data.append(data)
                        self.time_stamps_bvp.append(timestamp)  # Raw time, normalization happens later
                        data_row = {'BVP': data}
                        self.update_data_frames(data_row, "E4_Bvp")

                    elif stream_type == "E4_Gsr":
                        data = float(sample_data[2].replace(',', '.'))
                        self.gsr_data.append(data)
                        self.time_stamps_gsr.append(timestamp)  # Raw time, normalization happens later
                        data_row = {'GSR': data}
                        self.update_data_frames(data_row, "E4_Gsr")

                    elif stream_type == "E4_Temperature":
                        data = float(sample_data[2].replace(',', '.'))
                        self.tmp_data.append(data)
                        self.time_stamps_tmp.append(timestamp)  # Raw time, normalization happens later
                        data_row = {'Temp': data}
                        self.update_data_frames(data_row, "E4_Temperature")

            if self.plotStreamedData:
                self.update_plots()

        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
        finally:
            self.save_to_excel()
            self.s.send("device_disconnect\r\n".encode())
            self.s.close()

    def update_data_frames(self, data_row, stream_type):
        """Update the data frames with normalized time."""
        # Normalize time when saving data to DataFrame
        if self.start_time is not None:
            if stream_type == "E4_Acc":
                self.acc_df['Timestamp'] = [t - self.start_time for t in self.time_stamps_acc]
            elif stream_type == "E4_Bvp":
                self.bvp_df['Timestamp'] = [t - self.start_time for t in self.time_stamps_bvp]
            elif stream_type == "E4_Gsr":
                self.gsr_df['Timestamp'] = [t - self.start_time for t in self.time_stamps_gsr]
            elif stream_type == "E4_Temperature":
                self.tmp_df['Timestamp'] = [t - self.start_time for t in self.time_stamps_tmp]

        data_df = pd.DataFrame([data_row])  # Ensure we use the normalized timestamp

        if stream_type == "E4_Acc":
            self.acc_df = pd.concat([self.acc_df, data_df], ignore_index=True)
        elif stream_type == "E4_Bvp":
            self.bvp_df = pd.concat([self.bvp_df, data_df], ignore_index=True)
        elif stream_type == "E4_Gsr":
            self.gsr_df = pd.concat([self.gsr_df, data_df], ignore_index=True)
        elif stream_type == "E4_Temperature":
            self.tmp_df = pd.concat([self.tmp_df, data_df], ignore_index=True)
