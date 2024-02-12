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



pointcloud_torso = open3d.io.read_point_cloud('pcd_10.ply')

mesh = o3d.io.read_triangle_mesh(r"C:\Users\Alessandro\Desktop\Registration_material\Acquisitions\EP_3DModels\EDITED_remeshed.obj")
vertices = np.array(mesh.vertices)  # Transpose for a 3xN matrix
vertices = vertices / 1000
reduction_factor = 0.5 # Adjust as needed #TODO: set reduction factor
downsampled_points = vertices[np.random.choice(vertices.shape[0], int(reduction_factor * vertices.shape[0]), replace=False)] #downsample the pointcloud if it is too heavy
points_phantom = downsampled_points
print(points_phantom.shape)


min_bound = np.array([-math.inf, -math.inf,  -math.inf])
max_bound = np.array([math.inf, math.inf, math.inf])
inlier_indices = np.all((min_bound <= pointcloud_torso.points) & (pointcloud_torso.points <= max_bound), axis=1)
cropped_point_cloud = pointcloud_torso.select_by_index(np.where(inlier_indices)[0].tolist())
cropped_point_cloud = np.asarray(cropped_point_cloud.points)
print(cropped_point_cloud.shape)


R_pca,t_pca,T_pca = PCA_registration(cropped_point_cloud.T,points_phantom.T)


 #Create a o3d pointcloud of the "source" pointcloud (patient's point CT)
source_cloud = o3d.geometry.PointCloud()
source_cloud.points = o3d.utility.Vector3dVector(cropped_point_cloud)

target_cloud = o3d.geometry.PointCloud()
target_cloud.points = o3d.utility.Vector3dVector(points_phantom)
target_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

result = o3d.pipelines.registration.registration_icp(
    source_cloud, target_cloud,  # Source and target point clouds
    1,  # Maximum correspondence distance (increase if points are far apart)
    T_pca,  # Initial transformation guess
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # Point-to-point ICP
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100000)  # Max iterations
)
refined_transform = result.transformation #perform icp refined registration using the output from PCA registration as initial guess

icp_regist = source_cloud.transform(refined_transform)#Apply the computed trasnfrom to the "source" pointcloud

# Compute the RMSE
evaluation = o3d.pipelines.registration.evaluate_registration(
    icp_regist, target_cloud,
    max_correspondence_distance=1  # Adjust as needed
)

rmse_icp = evaluation.inlier_rmse
print("RMSE_icp:", rmse_icp)
       

#pc_torso = pv.PolyData(np.asarray(pointcloud_torso.points))
pc_cropped_torso = pv.PolyData(np.asarray(cropped_point_cloud))
pc_phantom = pv.PolyData(points_phantom)
phantom_registered = pv.PolyData(np.asarray(icp_regist.points))

plotter = pv.Plotter()
plotter.add_mesh(phantom_registered,color='green')
plotter.add_mesh(pc_cropped_torso,color='red')
plotter.add_mesh(pc_phantom,color='blue')
plotter.add_axes_at_origin()
plotter.show()