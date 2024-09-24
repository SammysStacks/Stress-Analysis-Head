import os
import sys

from helperFiles.dataAcquisitionAndAnalysis import E4StreamingProtocols  # Import interfaces for reading/writing data from E4 wristband


if __name__ == '__main__':
    # specify if using the E4 watch for streaming
    E4StreamingIndicator = True
    # specify if user wants to view the data in real time
    plotStreamedData = True

    if E4StreamingIndicator:
        server_address = '127.0.0.1'
        server_port = 28000
        device_id = 'B516C6'
        buffer_size = 4096
        output_file = "E4_data.xlsx"  # Specify the output file name
        e4_streamer = E4StreamingProtocols.E4Streaming(server_address, server_port, device_id, buffer_size, output_file, plotStreamedData)
        e4_streamer.connect()
        e4_streamer.subscribe_to_data()
        e4_streamer.stream()
