from pynput import keyboard
import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_rus
import hl2ss_3dcv
import hl2ss_mp
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
import multiprocessing as mp
import copy


# HoloLens address
host = "192.168.0.101"

# Create a socket server

# Operating mode
# 0: video
# 1: video + rig pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_0

# Framerate denominator (must be > 0)
# Effective framerate is framerate / divisor
divisor = 1 

# Depth encoding profile
profile_z = hl2ss.DepthProfile.SAME

# Video encoding profile
profile_ab = hl2ss.VideoProfile.H265_MAIN

# Buffer length in seconds
buffer_length = 5

calibration_path = 'C:/Users/Alessandro/Desktop/Neuro'

#------------------------------------------------------------------------------
if __name__ == '__main__':

    #calibration = hl2ss_lnm.download_calibration_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    calibration = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)
    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration.scale)

    
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS * buffer_length)
    producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, ...)
    sink_depth.get_attach_response()

    '''
    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    '''



    def Compute_3D_coord(keys,depth):
        key_3D = []
        
        for blob in keys:
            
            u = int(blob.pt[0])
            v = int(blob.pt[1])
            depth = depth/1
            z = depth[v][u]
            xy = uv2xy[u][v]
            XYZ = [xy[0]*z,xy[1]*z,z]
            #x = (u - CX_DEPTH) * z  / FX_DEPTH
            #y = (v - CY_DEPTH) * z / FY_DEPTH
            #XYZ_sphere = (XYZ *(1 + 0.0054/z)).tolist()
            #print("point from function", [x,y,z])
            key_3D.append([XYZ])
        print(key_3D)
        
        return key_3D

    def clusterObjManich(pcd, epsValue):
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as mm:
            labels = np.array(
                pcd.cluster_dbscan(eps=epsValue, min_points=20, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcd])
        points = np.asarray(pcd.points)
        listCluster=[]
        nPt_cluster_i=np.zeros(max_label+1)
        for i in range (0, max_label+1):
            cluster_i=points[labels == i]
            listCluster.append(cluster_i)
            nPt_cluster_i[i]=cluster_i.shape[0]

        clusterMaxind= np.argmax(nPt_cluster_i)

        return listCluster[clusterMaxind]
 

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

    def hiddenPointRemoval(pcd):
        # Convert mesh to a point cloud and estimate dimensions.
        
        pcd = pcd.sample_points_poisson_disk(50000)
        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        #min_bound = np.array([-math.inf,-math.inf ,-math.inf])
        #max_bound = np.array([math.inf, 0.0, -0.06])
        #inlier_indices = np.all((min_bound <= pcd.points) & (pcd.points <= max_bound), axis=1)
        #cropped_point_cloud = pcd.select_by_index(np.where(inlier_indices)[0].tolist())
        print("Displaying input point cloud ...")
        # o3d.visualization.draw([pcd], point_size=5)

        # Define parameters used for hidden_point_removal.
        originCamera = np.asarray([0, 0, 0])
        camera = [0, -diameter, 0]  # l'asse y è quello che punta verso la faccia, verificare che sia sempre così
        radius = diameter * 100

        # Get all points that are visible from given view point.
        _, pt_map = pcd.hidden_point_removal(camera, radius)

        print("Displaying point cloud after hidden point removal ...")
        pcd_withoutHidden = copy.deepcopy(pcd)
        pcd_withoutHidden = pcd_withoutHidden.select_by_index(pt_map)
        pcd_withoutHidden.paint_uniform_color([0, 0.706, 0])
        return pcd_withoutHidden


    while True:
        HOST = "192.168.0.100"
        PORT = 1000

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((HOST, PORT))
                server_socket.listen()

                print(f"Server listening on {HOST}:{PORT}")

                while True:

                    client_socket, client_address = server_socket.accept()
                    print(f"Accepted connection from {client_address}")
                    
                    '''
                    data = client.get_next_packet()
                    data_depth = hl2ss_3dcv.rm_depth_undistort(data.payload.depth,calibration.undistort_map)
                    data_IR = hl2ss_3dcv.rm_depth_undistort(data.payload.ab,calibration.undistort_map)
                    
                    #cv2.imshow("AB" , data.payload.depth/ np.max(data.payload.depth)) # Scaled for visibility
                    #depth_normal = hl2ss_3dcv.rm_depth_normalize(data.payload.ab, calibration.scale)
                    #depth_undistort = hl2ss_3dcv.rm_depth_undistort(depth_normal,calibration.undistort_map)
                    #points = Compute_3D_coord(data)
                    cv2.imshow('AB', data.payload.ab / np.max(data.payload.ab))
                    
                    Blob_detector(data_IR,data_depth,detector)
                    
                    cv2.waitKey(1)
                    '''
                    sink_depth.acquire()
                    _, data = sink_depth.get_most_recent_frame()

                    depth = hl2ss_3dcv.rm_depth_undistort(data.payload.depth, calibration.undistort_map)
                    depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)
                    
                    
                    #data = hl2ss_3dcv.rm_depth_undistort(data,calibration.undistort_map)
                    o3d_depth_image = o3d.geometry.Image(depth)
                    
                    intrinsic = o3d.camera.PinholeCameraIntrinsic()
                    intrinsic.set_intrinsics(hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT, calibration.intrinsics[0, 0], calibration.intrinsics[1, 1], calibration.intrinsics[2, 0], calibration.intrinsics[2, 1])
                    
                    # Create a point cloud from the depth image
                    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                        o3d_depth_image,
                        intrinsic,
                        depth_scale=1.0,  # Adjust based on your specific depth values
                        #depth_trunc=0.5,  # Adjust based on your specific depth values
                    )

                    min_bound = np.array([-math.inf,-math.inf,0.1])
                    max_bound = np.array([math.inf,-0.1, 0.7])
                    
                    inlier_indices = np.all((min_bound <= point_cloud.points) & (point_cloud.points <= max_bound), axis=1)
                    cropped_point_cloud = point_cloud.select_by_index(np.where(inlier_indices)[0].tolist())
                    
                    # Remove statistical outliers
                    cl, ind = cropped_point_cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.1) 
                    filtered_pc = cropped_point_cloud.select_by_index(ind)
                    #filtered_pc = filtered_pc.voxel_down_sample(voxel_size=0.001)
                    clust = clusterObjManich(filtered_pc, 0.03)
                    #cropped_point_cloud.estimate_normals()
                    #mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cropped_point_cloud)

                    #o3d.io.write_triangle_mesh("C:/Users/Alessandro/Desktop/Neuro/pcl_holo.obj", mesh)
                    
                    clustered_cloud = o3d.geometry.PointCloud()
                    clustered_cloud.points = o3d.utility.Vector3dVector(clust)
                    #o3d.io.write_point_cloud("C:/Users/Alessandro/Desktop/Neuro/pcl_holo.ply",clustered_cloud)
                    
                    pyvista_cloud = pv.PolyData(np.asarray(clustered_cloud.points))
                    pyvista_cloud.plot(point_size=1, color="red")
                    

                    T_depth_to_world = hl2ss_3dcv.camera_to_rignode(calibration.extrinsics) @ hl2ss_3dcv.reference_to_world(data.pose)
                    points_patient_depth = np.asarray(clustered_cloud.points)
                    points_patient_world = hl2ss_3dcv.transform(points_patient_depth, T_depth_to_world)
                    pyvista_transformed = pv.PolyData(points_patient_world)
                    pyvista_transformed.plot(point_size=1, color="red")
                    
                    points_patient_world = points_patient_world.T
                    print(points_patient_world.shape)
                    

                    mesh = o3d.io.read_triangle_mesh("C:/Users/Alessandro/Desktop/Neuro/face_3t_mWtextr.obj")
                    filtered_pca = hiddenPointRemoval(mesh)
                    vertices = np.array(filtered_pca.points)  # Transpose for a 3xN matrix
                    reduction_factor = 0.5 # Adjust as needed
                    downsampled_points = vertices[np.random.choice(vertices.shape[0], int(reduction_factor * vertices.shape[0]), replace=False)]
                    points_patient_CT = downsampled_points.T
                    print(points_patient_CT.shape)

                    R_pca,t_pca,T_pca = PCA_registration(points_patient_world,points_patient_CT)
                    registered_pca = R_pca @ points_patient_world + t_pca

                    plotter = pv.Plotter()

                    source_cloud = o3d.geometry.PointCloud()
                    source_cloud.points = o3d.utility.Vector3dVector(points_patient_world.T)  # Transpose for correct shape

                    target_cloud = o3d.geometry.PointCloud()
                    target_cloud.points = o3d.utility.Vector3dVector(points_patient_CT.T)
                    target_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                    result = o3d.pipelines.registration.registration_icp(
                        source_cloud, target_cloud,  # Source and target point clouds
                        1,  # Maximum correspondence distance (increase if points are far apart)
                        T_pca,  # Initial transformation guess
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # Point-to-point ICP
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100000)  # Max iterations
                    )

                    refined_transform = result.transformation

                    icp_regist = source_cloud.transform(refined_transform)

                    # Compute the RMSE
                    evaluation = o3d.pipelines.registration.evaluate_registration(
                        icp_regist, target_cloud,
                        max_correspondence_distance=0.01  # Adjust as needed
                    )

                    rmse_icp = evaluation.inlier_rmse
                    print("RMSE_icp:", rmse_icp)


                    icp_regist = pv.PolyData(np.asarray(icp_regist.points))
                    cloud_target = pv.PolyData(points_patient_CT.T)

                    plotter.add_mesh(icp_regist, color="blue", point_size=5)
                    #plotter.add_mesh(cloud_registered, color="red", point_size=5)
                    plotter.add_mesh(cloud_target, color="green", point_size=5)
                    plotter.show()
                    
                    
                    T_CT_to_world = np.linalg.inv(refined_transform)

                    matrixString = '\n'.join([','.join(map(str, row)) for row in T_CT_to_world ])
                    print(matrixString)

                    
                    with client_socket:
                        client_socket.sendall(matrixString.encode("UTF-8"))
                        print("Matrix sent to client")
                        
                    client_socket.close()
                    break
                    
                    

                
                
                

               