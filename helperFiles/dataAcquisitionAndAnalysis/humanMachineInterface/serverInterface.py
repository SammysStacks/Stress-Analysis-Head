import socket
import threading
import time

from pexpect import TIMEOUT

from helperFiles.dataAcquisitionAndAnalysis.empaticaInterface import empaticaInterface
from helperFiles.machineLearning.modelControl.modelSpecifications.compileModelInfo import compileModelInfo


class serverInterface:

    def __init__(self, streamingOrder, analysisProtocols, deviceType="empatica"):
        # General parameters.
        self.communicationFailedSTR = "Communication failed."
        self.messageDelimiter = '\r\n'
        self.deviceType = deviceType

        if self.deviceType == "empatica":
            self.mainDevice = empaticaInterface(streamingOrder=streamingOrder, analysisProtocols=analysisProtocols)
        else:
            raise f"Invalid device type: {self.deviceType}."

    def startServer(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as serverSocket:
            while True:
                # Try to bind to the communication port.
                try:
                    serverSocket.connect(('127.0.0.1', self.mainDevice.communication_port))
                    serverSocket.setblocking(False)
                    serverSocket.settimeout(3)
                    break
                except OSError:
                    continue

            # Listen for incoming connections with a backlog of 100
            print(f"\nServer listening on port {self.mainDevice.communication_port}...")
            self.mainDevice.deviceSpecificConnection(serverSocket)
            while self.mainDevice.closeServer: time.sleep(1)
            self.mainDevice.startStreamingData(serverSocket)

            while True:
                try:
                    # Receive the message
                    responseMessage = serverSocket.recv(self.mainDevice.buffer_size).decode("utf-8")
                    if not responseMessage: continue  # Keep the connection alive if no message is received

                    # Process the message (modify as needed for your use case)
                    connectionLost = self.mainDevice.process_message(responseMessage)
                    if connectionLost or self.mainDevice.closeServer: return None
                except TIMEOUT as e:
                    print("Timeout occurred:", e)

                # Should never occur.
                except Exception as e:
                    print(f'Error processing request: {e}')


if __name__ == "__main__":
    # Specify the device and the streaming order.
    streamingOrderTEMP = compileModelInfo().streamingOrder_e4
    serverClass = serverInterface(streamingOrderTEMP, analysisProtocols=(), deviceType='empatica')
    serverClass.mainDevice.closeServer = False
    serverClass.startServer()
