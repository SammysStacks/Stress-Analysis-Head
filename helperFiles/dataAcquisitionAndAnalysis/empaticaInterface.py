import time


# Import Bioelectric Analysis Files


class empaticaInterface:

    def __init__(self, device_id='B516C6', streamingOrder=(), analysisProtocols=()):
        # General parameters.
        self.analysisProtocols = analysisProtocols
        self.streamingOrder = streamingOrder
        self.device_id = device_id
        self.firstTimePoint = None
        self.endServer = False

        # Hard-coded universal empatica parameters.
        self.server_address = '127.0.0.1'
        self.communication_port = 28000  
        self.buffer_size = 4096

    def deviceSpecificConnection(self, serverSocket):
        # Connect to the device.
        serverSocket.send(("device_connect " + self.device_id + "\r\n").encode())
        response = serverSocket.recv(self.buffer_size)
        print("\n\tConnecting to device", response.decode("utf-8").replace("\n", ""))

        serverSocket.send("pause ON\r\n".encode())
        response = serverSocket.recv(self.buffer_size)
        print(f"\t{response.decode("utf-8").replace("\n", "")}")
        time.sleep(1)  # Stabilize connection

        if "acc" in self.streamingOrder:
            serverSocket.send("device_subscribe acc ON\r\n".encode())
            response = serverSocket.recv(self.buffer_size)
            print(f"\t{response.decode("utf-8").replace("\n", "")}")

        if "bvp" in self.streamingOrder:
            serverSocket.send("device_subscribe bvp ON\r\n".encode())
            response = serverSocket.recv(self.buffer_size)
            print(f"\t{response.decode("utf-8").replace("\n", "")}")

        if "eda" in self.streamingOrder:
            serverSocket.send("device_subscribe gsr ON\r\n".encode())
            response = serverSocket.recv(self.buffer_size)
            print(f"\t{response.decode("utf-8").replace("\n", "")}")

        if "temp" in self.streamingOrder:
            serverSocket.send("device_subscribe tmp ON\r\n".encode())
            response = serverSocket.recv(self.buffer_size)
            print(f"\t{response.decode("utf-8").replace("\n", "")}")

        serverSocket.send("pause OFF\r\n".encode())
        response = serverSocket.recv(self.buffer_size)
        print(f"\t{response.decode("utf-8").replace("\n", "")}")

    def process_message(self, receivedMessage):
        # Check if the connection is still valid.
        if "connection lost to device" in receivedMessage: return True

        # Separate out each sensor reading.
        for sensorReading in receivedMessage.split("\n"):
            sensorReading = sensorReading.replace(',', '.')
            sample_data = sensorReading.split()
            if len(sample_data) < 3: continue
            stream_type = sample_data[0]
            data = sample_data[2:]

            # Skip non-numeric values.
            try: timestamp = float(sample_data[1])
            except ValueError: print("\t", f"Invalid time: {sample_data[1]}"); continue

            # Initialize start time on first sample
            if self.firstTimePoint is None: self.firstTimePoint = timestamp
            normalized_timestamp = timestamp - self.firstTimePoint

            match stream_type:
                case "E4_Acc": analysis = self.analysisProtocols['acc']
                case "E4_Bvp": analysis = self.analysisProtocols['bvp']
                case "E4_Gsr":  analysis = self.analysisProtocols['eda']
                case "E4_Temperature": analysis = self.analysisProtocols['temp']
                case _: raise ValueError(f"Unknown stream type: {stream_type}")

            # Organize the data.
            self.organizeData(analysis=analysis, timepoint=normalized_timestamp, datapoint=data)
        return False

    @staticmethod
    def organizeData(analysis, timepoint, datapoint):
        # Update the timepoints.
        analysis.timepoints.append(timepoint)

        # For each channel, update the voltage data.
        for channelIndex in range(analysis.numChannels):
            # Compile the datapoints for each of the sensor's channels.
            streamingDataIndex = analysis.streamingChannelInds[channelIndex]
            newData = datapoint[streamingDataIndex]

            # Add the Data to the Correct Channel
            analysis.channelData[channelIndex].append(newData)

    def close(self):
        self.endServer = True
        