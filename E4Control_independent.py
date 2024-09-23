# adapted from https://github.com/HectorCarral/Empatica-E4-LSL
import socket
import time
import matplotlib
matplotlib.use('TkAgg') # set backend plotting to Tkinter
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import os

# SELECT DATA TO STREAM
acc = True  # 3-axis acceleration
bvp = True  # Blood Volume Pulse
gsr = True  # Galvanic Skin Response (Electrodermal Activity)
tmp = True  # Temperature

serverAddress = '127.0.0.1'
serverPort = 28000
bufferSize = 4096

deviceID = 'B516C6'

# Matplotlib setup
plt.ion()  # Enable interactive mode
fig, axs = plt.subplots(4, 1, figsize=(12, 10))  # 4 subplots for ACC, BVP, GSR, and Temp

# Data buffers for real-time plotting only plot the most recent 100 datapoints
acc_data = deque(maxlen=100)
bvp_data = deque(maxlen=100)
gsr_data = deque(maxlen=100)
tmp_data = deque(maxlen=100)

# Time stamps for each type of data
time_stamps_acc = deque(maxlen=100)
time_stamps_bvp = deque(maxlen=100)
time_stamps_gsr = deque(maxlen=100)
time_stamps_tmp = deque(maxlen=100)

# Initialize plot lines
acc_lines = [axs[0].plot([], [], label="ACC_X")[0],
             axs[0].plot([], [], label="ACC_Y")[0],
             axs[0].plot([], [], label="ACC_Z")[0]]
bvp_line = axs[1].plot([], [], label="BVP", color='purple')[0]
gsr_line = axs[2].plot([], [], label="GSR", color='orange')[0]
tmp_line = axs[3].plot([], [], label="Temp", color='cyan')[0]

# Setup the axis labels and titles
axs[0].set_title("3-axis Acceleration")
axs[0].set_ylabel("Acceleration (g)")
axs[0].legend()

axs[1].set_title("Blood Volume Pulse (BVP)")
axs[1].set_ylabel("BVP (AU)")

axs[2].set_title("Galvanic Skin Response (GSR)")
axs[2].set_ylabel("GSR (µS)")

axs[3].set_title("Temperature (Temp)")
axs[3].set_ylabel("Temp (°C)")
axs[3].set_xlabel("Time (s)")

# Setup for saving data to Excel
output_file = os.path.join(os.getcwd(), "E4_data.xlsx")

# DataFrames for saving to Excel sheets
acc_df = pd.DataFrame(columns=['Timestamp', 'ACC_X', 'ACC_Y', 'ACC_Z'])
bvp_df = pd.DataFrame(columns=['Timestamp', 'BVP'])
gsr_df = pd.DataFrame(columns=['Timestamp', 'GSR'])
tmp_df = pd.DataFrame(columns=['Timestamp', 'Temp'])


def save_to_excel():
    print("Saving data to Excel...")
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
        acc_df.to_excel(writer, sheet_name='ACC', index=False)
        bvp_df.to_excel(writer, sheet_name='BVP', index=False)
        gsr_df.to_excel(writer, sheet_name='GSR', index=False)
        tmp_df.to_excel(writer, sheet_name='Temp', index=False)


# Helper functions
def calculate_sampling_frequency(timestamps):
    """Calculate the sampling frequency based on timestamps."""
    if len(timestamps) > 1:
        return 1 / (timestamps[-1] - timestamps[-2])
    else:
        return None


def update_data_frames(data_row, stream_type):
    """Append data to respective DataFrames."""
    global acc_df, bvp_df, gsr_df, tmp_df

    # Convert data_row to DataFrame for concatenation
    data_df = pd.DataFrame([data_row])

    if stream_type == "E4_Acc":
        acc_df = pd.concat([acc_df, data_df], ignore_index=True)
    elif stream_type == "E4_Bvp":
        bvp_df = pd.concat([bvp_df, data_df], ignore_index=True)
    elif stream_type == "E4_Gsr":
        gsr_df = pd.concat([gsr_df, data_df], ignore_index=True)
    elif stream_type == "E4_Temperature":
        tmp_df = pd.concat([tmp_df, data_df], ignore_index=True)


def connect():
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)

    print("Connecting to server")
    s.connect((serverAddress, serverPort))
    print("Connected to server\n")

    print("Devices available:")
    s.send("device_list\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))

    print("Connecting to device")
    s.send(("device_connect " + deviceID + "\r\n").encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))

    print("Pausing data receiving")
    s.send("pause ON\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))


connect()
time.sleep(1)


def subscribe_to_data():
    if acc:
        print("Subscribing to ACC")
        s.send(("device_subscribe " + 'acc' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if bvp:
        print("Subscribing to BVP")
        s.send(("device_subscribe " + 'bvp' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if gsr:
        print("Subscribing to GSR")
        s.send(("device_subscribe " + 'gsr' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if tmp:
        print("Subscribing to Temp")
        s.send(("device_subscribe " + 'tmp' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

    print("Resuming data receiving")
    s.send("pause OFF\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))


subscribe_to_data()


def update_plots():
    # Update acceleration plot
    if len(time_stamps_acc) == len(acc_data):
        for i in range(3):
            acc_lines[i].set_data(time_stamps_acc, [d[i] for d in acc_data])  # ACC X, Y, Z axes
        axs[0].relim()
        axs[0].autoscale_view()

    # Update BVP plot
    if len(time_stamps_bvp) == len(bvp_data):
        bvp_line.set_data(time_stamps_bvp, bvp_data)
        axs[1].relim()
        axs[1].autoscale_view()

    # Update GSR plot
    if len(time_stamps_gsr) == len(gsr_data):
        gsr_line.set_data(time_stamps_gsr, gsr_data)
        axs[2].relim()
        axs[2].autoscale_view()

    # Update Temperature plot
    if len(time_stamps_tmp) == len(tmp_data):
        tmp_line.set_data(time_stamps_tmp, tmp_data)
        axs[3].relim()
        axs[3].autoscale_view()

    # Draw the updated plots
    plt.tight_layout()  # Prevent overlapping of labels and titles
    plt.draw()
    plt.pause(0.001)  # Allow the plot to update


def stream():
    try:
        print("Streaming...")
        while True:
            try:
                response = s.recv(bufferSize).decode("utf-8")
                if "connection lost to device" in response:
                    print(response)
                    break
                samples = response.split("\n")
                for i in range(len(samples) - 1):
                    # Ensure the sample has enough components to avoid 'IndexError'
                    sample_data = samples[i].split()
                    if len(sample_data) < 3:  # Check if there are at least 3 components (type, timestamp, and data)
                        continue  # Skip malformed or incomplete samples

                    stream_type = sample_data[0]

                    # Check if the sample is valid before processing
                    try:
                        timestamp = float(sample_data[1].replace(',', '.'))

                        data_row = {'Timestamp': timestamp}

                        if stream_type == "E4_Acc":
                            if len(sample_data) >= 5:  # Ensure enough data for ACC (X, Y, Z)
                                data = [int(sample_data[2].replace(',', '.')),
                                        int(sample_data[3].replace(',', '.')),
                                        int(sample_data[4].replace(',', '.'))]
                                acc_data.append(data)
                                time_stamps_acc.append(timestamp)

                                acc_sampling_freq = calculate_sampling_frequency(time_stamps_acc)
                                data_row.update({'ACC_X': data[0], 'ACC_Y': data[1], 'ACC_Z': data[2]})
                                update_data_frames(data_row, "E4_Acc")

                        if stream_type == "E4_Bvp" and len(sample_data) >= 3:
                            data = float(sample_data[2].replace(',', '.'))
                            bvp_data.append(data)
                            time_stamps_bvp.append(timestamp)

                            bvp_sampling_freq = calculate_sampling_frequency(time_stamps_bvp)
                            data_row.update({'BVP': data})
                            update_data_frames(data_row, "E4_Bvp")

                        if stream_type == "E4_Gsr" and len(sample_data) >= 3:
                            data = float(sample_data[2].replace(',', '.'))
                            gsr_data.append(data)
                            time_stamps_gsr.append(timestamp)

                            gsr_sampling_freq = calculate_sampling_frequency(time_stamps_gsr)
                            data_row.update({'GSR': data})
                            update_data_frames(data_row, "E4_Gsr")

                        if stream_type == "E4_Temperature" and len(sample_data) >= 3:
                            data = float(sample_data[2].replace(',', '.'))
                            tmp_data.append(data)
                            time_stamps_tmp.append(timestamp)

                            tmp_sampling_freq = calculate_sampling_frequency(time_stamps_tmp)
                            data_row.update({'Temp': data})
                            update_data_frames(data_row, "E4_Temperature")

                    except ValueError:
                        # Ignore invalid data (like 'pause' messages)
                        continue

                update_plots()  # Call plot update function

            except socket.timeout:
                print("Socket timeout")
                break
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        save_to_excel()  # Automatically save data without prompt
        s.send("device_disconnect\r\n".encode())
        s.close()


# Start the data streaming
stream()