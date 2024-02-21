import os
from pynput import keyboard
import numpy as np
import cv2
import pykinect_azure as pykinect
import open3d as o3d
from open3d import *
import math
import socket
import pickle
import json
import time
import imutils
import itertools
import matplotlib.pyplot as plt
import pyvista as pv
import keyboard
import multiprocessing as mp
import copy
import struct


#Function to perform PCA initial registration
def PCA_registration(points_U,points_Y):
    
    N_U = points_U.shape[1]
    N_Y = points_Y.shape[1]
    
    # calculate centroids
    U_centroid = np.reshape(1/N_U * (np.sum(points_U, axis=1)), (3,1))
    Y_centroid = np.reshape(1/N_Y * (np.sum(points_Y, axis=1)), (3,1))

    cov_U = np.cov(points_U)
    cov_Y = np.cov(points_Y)

    U_pca_points_U,S_pca_points_U,V_T_pca_points_U = np.linalg.svd(cov_U)
    W_0 = U_pca_points_U

    W_1 = np.zeros((3,3))
    W_1[:,0] = - W_0[:,0]
    W_1[:,1] = - W_0[:,1]
    W_1[:,2] =   W_0[:,2]

    W_2 = np.zeros((3,3))
    W_2[:,0] =   W_0[:,0]
    W_2[:,1] = - W_0[:,1]
    W_2[:,2] = - W_0[:,2]

    W_3 = np.zeros((3,3))
    W_3[:,0] = - W_0[:,0]
    W_3[:,1] =   W_0[:,1]
    W_3[:,2] = - W_0[:,2]


    U_pca_points_Y,S_pca_points_Y,V_T_pca_points_Y = np.linalg.svd(cov_Y)
    V = U_pca_points_Y

    R_0 = V @ W_0.T
    print(np.linalg.det(R_0))
    R_1 = V @ W_1.T
    print(np.linalg.det(R_1))
    R_2 = V @ W_2.T
    print(np.linalg.det(R_2))
    R_3 = V @ W_3.T
    print(np.linalg.det(R_3))

    t_0 = Y_centroid - R_0 @ U_centroid
    t_1 = Y_centroid - R_1 @ U_centroid
    t_2 = Y_centroid - R_2 @ U_centroid
    t_3 = Y_centroid - R_3 @ U_centroid

    R_k = [R_0,R_1,R_2,R_3]
    t_k = [t_0,t_1,t_2,t_3]
    
    minimun_error = 100

    for k in range(4):
        
        U_k = R_k[k] @ points_U + t_k[k]
        
        C_k = []

        for i in range(U_k.shape[1]):

            distances = np.linalg.norm(points_Y - U_k[:,i].reshape(-1,1), axis=0)
            closest_index = np.argmin(distances)
            closest_point = points_Y[:,closest_index]
            C_k.append(closest_point)
        
        C_k = np.array(C_k).T

        squared_diff = np.square(C_k - U_k)
        mean_squared_error = np.mean(squared_diff)
        rmse_error = np.sqrt(mean_squared_error)
        print(rmse_error)
        

        if(rmse_error < minimun_error):
            minimun_error = rmse_error
            result_index = k

    print("RMSE_pca:", minimun_error)
    T_k = np.identity(4)
    T_k[0:3,0:3] = R_k[result_index]
    
    T_k[0][-1] = t_k[result_index][0]
    T_k[1][-1] = t_k[result_index][1]
    T_k[2][-1] = t_k[result_index][2]

    return R_k[result_index], t_k[result_index], T_k




phantom = pv.read(r"C:\Users\Alessandro\Desktop\DAVIDTrials\data\3DModels\UNITY\skin_UNITY_two_layers.obj")

list_points = []
list_actors = []
surf = pv.PolyData()

def put_control_points(pos):
    
        sphere = pv.Sphere(radius=1, center=pos)
        control_point=p.add_mesh(sphere, color="red", opacity=0.8)
        list_points.append(pos)
        list_actors.append(control_point)



def remove_control_points():
    p.remove_actor(list_actors[-1])
    list_actors.pop()
    list_points.pop()
    p.render()

def reconstruct_surface():
    cloud = pv.PolyData(np.asarray(list_points))
    surf = cloud.delaunay_2d()
    
    print(np.asarray(surf))
    p.add_mesh(surf,show_edges=True)
    p.render()

    return surf

# Tracks right clicks

axes = pv.Axes(show_actor=True, actor_scale=40.0, line_width=5)
reference_frame = np.asarray([0,0,0])
axes.origin = (0,0,0)
p = pv.Plotter(notebook=0)
p.add_mesh(phantom,opacity=0.3)
p.add_actor(axes.actor)
p.enable_surface_point_picking(put_control_points)
p.add_key_event('c',remove_control_points)
p.show()

surf = reconstruct_surface()

p = pv.Plotter(notebook=0)
p.add_mesh(phantom,opacity=0.3)
p.add_mesh(surf,opacity=0.3,color='red')
p.show()

print(surf.faces)
print(surf.points)


triangles = [value for index, value in enumerate(surf.faces.tolist()) if index % 4 != 0]


points = [
    x
    for row in surf.points.tolist()
    for x in row
]




while True:
        HOST = "192.168.0.103"
        PORT = 2000

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((HOST, PORT))
                server_socket.listen()

                print(f"Server listening on {HOST}:{PORT}")

                client_socket, client_address = server_socket.accept()
                print(f"Accepted connection from {client_address}")
          

        while True:
            
            data = {'faces': triangles,'points': points}
            print(data)
            
            with client_socket:
                client_socket.sendall(json.dumps(data).encode("UTF-8"))
                
                
            client_socket.close()
            break
