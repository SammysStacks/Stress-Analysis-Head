import os
import socket
import time


# Import Bioelectric Analysis Files


class E4StreamingHelpers:
    def __init__(self, device_id='B516C6'):
        # General parameters.
        self.device_id = device_id
        self.serverSocket = None

        # Hard-coded universal empatica parameters.
        self.server_address = '127.0.0.1'
        self.communication_port = 28000  
        self.buffer_size = 4096

    def connect(self):
        # Connect to the device.
        self.serverSocket.send(("device_connect " + self.device_id + "\r\n").encode())
        response = self.serverSocket.recv(self.buffer_size)
        print("Connecting to device", response.decode("utf-8"))

        print("Pausing data receiving")
        self.serverSocket.send("pause ON\r\n".encode())
        response = self.serverSocket.recv(self.buffer_size)
        print(response.decode("utf-8"))

        time.sleep(1)  # Stabilize connection

    def subscribe_to_data(self, acc=True, bvp=True, gsr=True, tmp=True):
        if acc:
            print("Subscribing to ACC")
            self.serverSocket.send("device_subscribe acc ON\r\n".encode())
            response = self.serverSocket.recv(self.buffer_size)
            print(response.decode("utf-8"))

        if bvp:
            print("Subscribing to BVP")
            self.serverSocket.send("device_subscribe bvp ON\r\n".encode())
            response = self.serverSocket.recv(self.buffer_size)
            print(response.decode("utf-8"))

        if gsr:
            print("Subscribing to GSR")
            self.serverSocket.send("device_subscribe gsr ON\r\n".encode())
            response = self.serverSocket.recv(self.buffer_size)
            print(response.decode("utf-8"))

        if tmp:
            print("Subscribing to Temp")
            self.serverSocket.send("device_subscribe tmp ON\r\n".encode())
            response = self.serverSocket.recv(self.buffer_size)
            print(response.decode("utf-8"))

        print("Resuming data receiving")
        self.serverSocket.send("pause OFF\r\n".encode())
        response = self.serverSocket.recv(self.buffer_size)
        print(response.decode("utf-8"))