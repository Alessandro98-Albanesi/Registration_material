import itertools
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import MerweScaledSigmaPoints as SigmaPoints
from scipy.spatial.distance import cdist
import socket
import pickle


while True:
    HOST = "192.168.227.213"
    PORT = 1000

    # Define a 4x4 matrix
    matrix_to_send = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    matrixString = '\n'.join([','.join(map(str, row)) for row in matrix_to_send])

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()

        print(f"Server listening on {HOST}:{PORT}")

        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Accepted connection from {client_address}")

            with client_socket:
                client_socket.sendall(matrixString.encode("UTF-8"))
                print("Matrix sent to client")
                 # Close the client socket after sending the matrix
            client_socket.close()
        