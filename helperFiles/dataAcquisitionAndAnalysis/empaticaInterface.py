import time


class empaticaInterface:

    def __init__(self, streamingOrder=(), analysisProtocols=()):
        # General parameters.
        self.analysisProtocols = analysisProtocols
        self.streamingOrder = streamingOrder
        self.firstTimePoint = None
        self.closeServer = True
        self.device_id = None

        # Hard-coded universal empatica parameters.
        self.possibleSensors = {'acc': 'acc', 'bat': 'bat', 'bvp': 'bvp', 'eda': 'gsr', 'ibi': 'idi', 'tag': 'tag', 'temp': 'tmp'}
        self.server_address = '127.0.0.1'
        self.communication_port = 28000
        self.buffer_size = 4096

    def deviceSpecificConnection(self, serverSocket):
        # Get the device.
        deviceList = self.sendMessage(serverSocket, message=f"device_list\r\n")
        device_id = deviceList.split('Empatica')[0]
        self.device_id = device_id.split(" | ")[-1]

        # Connect to the device.
        self.sendMessage(serverSocket, message=f"device_connect {self.device_id}\r\n")
        self.sendMessage(serverSocket, message=f"pause ON\r\n")
        time.sleep(1)  # Stabilize connection

        # Subscribe to the sensors.
        for sensor in self.streamingOrder:
            if sensor in self.possibleSensors.keys():
                self.sendMessage(serverSocket, message=f"device_subscribe {self.possibleSensors[sensor]} ON\r\n")

    def startStreamingData(self, serverSocket):
        self.sendMessage(serverSocket, message=f"pause OFF\r\n")

    def sendMessage(self, serverSocket, message):
        # Send the message.
        serverSocket.sendall(message.encode())

        # Receive the response.
        response = serverSocket.recv(self.buffer_size)
        response = response.decode("utf-8").replace("\n", "")
        print(f"\t{response}")

        # Check if the connection is still valid.
        if "You are not connected to any device" in response: self.closeServer = True

        return response

    def process_message(self, receivedMessage):
        # Check if the connection is still valid.
        if "connection lost to device" in receivedMessage: return True

        # Separate out each sensor reading.
        for sensorReading in receivedMessage.split("\n"):
            sensorData = sensorReading.split(" ")
            if 'pause' in sensorData: continue
            if len(sensorData) < 3: continue

            # Separate out the data.
            rawTimeSeconds = float(sensorData[1])
            dataChannels = sensorData[2:]
            sensorType = sensorData[0]

            # Initialize start time on first sample
            if self.firstTimePoint is None: self.firstTimePoint = rawTimeSeconds
            normalized_timestamp = rawTimeSeconds - self.firstTimePoint

            # match sensorType:
            #     case "E4_Acc": analysis = self.analysisProtocols['acc']
            #     case "E4_Bvp": analysis = self.analysisProtocols['bvp']
            #     case "E4_Gsr": analysis = self.analysisProtocols['eda']
            #     case "E4_Temperature": analysis = self.analysisProtocols['temp']
            #     case _: raise ValueError(f"Unknown stream type: {sensorType} {sensorReading}")
            #
            # # Organize the data.
            # self.organizeData(analysis=analysis, timepoint=normalized_timestamp, datapoint=dataChannels)

        return False

    @staticmethod
    def organizeData(analysis, timepoint, datapoint):
        # Update the timepoints.
        analysis.timepoints.append(float(timepoint))

        # For each channel, update the voltage data.
        for channelIndex in range(analysis.numChannels):
            # Extract the data and correct any units.
            newData = float(datapoint[channelIndex])
            if analysis.analysisType == 'eda': newData = newData * 1E-6

            # Add the data to the correct channel
            analysis.channelData[channelIndex].append(newData)

    def close(self):
        self.closeServer = True
