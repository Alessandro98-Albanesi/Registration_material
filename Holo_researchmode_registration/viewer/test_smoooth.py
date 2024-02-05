from pynput import keyboard
import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_rus
import hl2ss_3dcv
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
from tracker import Tracker
import pyvista as pv
import keyboard



def hole_filling_mls(point_cloud, search_radius):
    # Compute normals for the input point cloud
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Create a KD-tree from the input point cloud
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)

    # Iterate through each point in the point cloud
    for i in range(len(point_cloud.points)):
        # Get the coordinates of the current point
        current_point = np.asarray(point_cloud.points[i])

        # Check if the current point has NaN coordinates (indicating a hole)
        if np.any(np.isnan(current_point)):
            # Use the KD-tree to find nearest neighbors for the current point
            [k, idx, _] = kdtree.search_knn_vector_3d(current_point, 100)

            # Extract coordinates and normals of the neighbors
            neighbors_points = np.asarray(point_cloud.points)[idx[1:], :]
            neighbors_normals = np.asarray(point_cloud.normals)[idx[1:], :]

            # Perform Moving Least Squares (MLS) to estimate the filled point
            mls_result = o3d.geometry.PointCloud()
            mls_result.points = o3d.utility.Vector3dVector(neighbors_points)
            mls_result.normals = o3d.utility.Vector3dVector(neighbors_normals)
            mls_result = mls_result.voxel_down_sample(voxel_size=0.01)
            o3d.geometry.estimate_normals(mls_result, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
            # Extract the first point from the MLS result as the filled point
            filled_point = np.asarray(mls_result.points)[0, :]

            # Replace the NaN coordinates in the original point cloud with the filled point
            point_cloud.points[i] = filled_point

    return point_cloud

# Load a point cloud with holes (replace with your actual point cloud)
point_cloud = o3d.io.read_point_cloud("C:/Users/Alessandro/Desktop/Neuro/pcl_holo.ply")

# Set the search radius for MLS
search_radius = 0.001

# Apply the hole-filling algorithm
filled_point_cloud = hole_filling_mls(point_cloud, search_radius)

# Visualize the original and filled point clouds
o3d.visualization.draw_geometries([point_cloud, filled_point_cloud])