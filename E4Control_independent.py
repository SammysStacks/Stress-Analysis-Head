import os
import sys

from helperFiles.dataAcquisitionAndAnalysis import E4StreamingProtocols  # Import interfaces for reading/writing data from E4 wristband

# Add the directory of the current file to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # specify if using the E4 watch for streaming
    E4StreamingIndicator = True
    # specify if user wants to view the data in real time
    plotStreamedData = False

    if E4StreamingIndicator:
        server_address = '127.0.0.1'
        server_port = 28000
        device_id = 'B516C6'
        buffer_size = 4096

        # Define the path to save the file in the e4WatchData folder within _experimentalData
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
        experimental_data_folder = os.path.join(script_dir, "_experimentalData")  # Path to _experimentalData folder
        e4_watch_data_folder = os.path.join(experimental_data_folder, "e4WatchData")  # Path to e4WatchData folder

        # Ensure the e4WatchData folder exists
        os.makedirs(e4_watch_data_folder, exist_ok=True)

        # Set the output file path to save in the e4WatchData folder
        output_file = os.path.join(e4_watch_data_folder, "20240926_E4_data_Testing.xlsx")

        e4_streamer = E4StreamingProtocols.E4Streaming(server_address, server_port, device_id, buffer_size, output_file, plotStreamedData)
        e4_streamer.connect()
        e4_streamer.subscribe_to_data()
        e4_streamer.stream()
