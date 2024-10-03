import socket
import threading

from helperFiles.dataAcquisitionAndAnalysis.empaticaInterface import empaticaInterface


class serverInterface:

    def __init__(self, streamingOrder, analysisProtocols, deviceType="empatica", device_id='B516C6'):
        # General parameters.
        self.communicationFailedSTR = "Communication failed."
        self.messageDelimiter = '\r\n'
        self.deviceType = deviceType

        if self.deviceType == "empatica":
            self.mainDevice = empaticaInterface(device_id=device_id, streamingOrder=streamingOrder, analysisProtocols=analysisProtocols)
        else: raise f"Invalid device type: {self.deviceType}."

    def startServer(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as serverSocket:
            while True:
                # Try to bind to the communication port.
                try: serverSocket.connect(('127.0.0.1', self.mainDevice.communication_port)); break
                except OSError: continue

            # Listen for incoming connections with a backlog of 100
            print(f"\nServer listening on port {self.mainDevice.communication_port}...")
            self.mainDevice.deviceSpecificConnection(serverSocket)

            while True:
                try:
                    # Accept the connection.
                    client_socket, addr = serverSocket.accept()

                    # Handle the client in a separate thread.
                    communicationThread = threading.Thread(target=self.processRequest, args=(client_socket,))
                    communicationThread.daemon = True
                    communicationThread.start()

                    # If the server is set to end, break the loop.
                    if self.mainDevice.endServer: break
                except Exception as e:
                    print(f"\tError accepting connections: {e}")

    def processRequest(self, client_socket):
        try:
            while True:
                try:
                    # Receive the message
                    message = client_socket.recv(self.mainDevice.buffer_size).decode('utf-8')
                    if not message: continue  # Keep the connection alive if no message is received

                    # Process the message (modify as needed for your use case)
                    connectionLost = self.mainDevice.process_message(message)
                    if connectionLost or self.mainDevice.endServer: break
                except ConnectionResetError: break

                # Should never occur.
                except Exception as e:
                    print(f'Error processing request: {e}')
                    client_socket.send("15".encode('utf-8'))
        finally:
            client_socket.close()
