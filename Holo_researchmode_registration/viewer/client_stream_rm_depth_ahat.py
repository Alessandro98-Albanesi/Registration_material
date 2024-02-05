#------------------------------------------------------------------------------
# This script receives video from the HoloLens depth camera in ahat mode and 
# plays it. The resolution is 512x512 @ 45 FPS. The stream supports three 
# operating modes: 0) video, 1) video + rig pose, 2) query calibration (single 
# transfer). Depth and AB data are scaled for visibility. The ahat and long 
# throw streams cannot be used simultaneously.
# Press esc to stop.
# See https://github.com/jdibenes/hl2ss/tree/main/extensions before setting
# profile_z to hl2ss.DepthProfile.ZDEPTH (lossless* compression).
#------------------------------------------------------------------------------

from pynput import keyboard
import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_rus
import hl2ss_3dcv
import pykinect_azure as pykinect
import open3d as o3d
import math
import socket
import pickle
import json
import time
import imutils
import itertools
import matplotlib.pyplot as plt
import multiprocessing as mp
from tracker import Tracker
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import pyvista as pv
from scipy.spatial.transform import Rotation as R


# Settings --------------------------------------------------------------------
FX_DEPTH = 113.92193
FY_DEPTH = 114.5772
CX_DEPTH = 258.27924
CY_DEPTH = 257.61118

Obj_frame_tool = [[-0.02527,0.012049,-0.096151],[0.024793,0.012049,-0.061299],[0.000976,0.012049,-0.010063],[0.000976,0.012049,0.039937]]
T_world_object = np.identity(4)
temporal_array = []

# HoloLens address
host = "192.168.0.102"

# Operating mode
# 0: video
# 1: video + rig pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# Framerate denominator (must be > 0)
# Effective framerate is framerate / divisor
divisor = 1 

# Depth encoding profile
profile_z = hl2ss.DepthProfile.SAME

# Video encoding profile
profile_ab = hl2ss.VideoProfile.H265_MAIN

buffer_length = 5

calibration_path = 'C:/Users/Alessandro/Desktop/Neuro'

#------------------------------------------------------------------------------
if __name__ == '__main__':

    #calibration = hl2ss_lnm.download_calibration_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    calibration = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_AHAT, calibration_path)
    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration.intrinsics, hl2ss.Parameters_RM_DEPTH_AHAT.WIDTH, hl2ss.Parameters_RM_DEPTH_AHAT.HEIGHT)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration.scale)

    
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT, profile_z=profile_z, profile_ab=profile_ab))
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_AHAT, buffer_length * hl2ss.Parameters_RM_DEPTH_AHAT.FPS)
    producer.start(hl2ss.StreamPort.RM_DEPTH_AHAT)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_AHAT, manager, ...)

    sink_depth.get_attach_response()

    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    



    def KF_marker(coordinate,image,KF):

        # Predict
        (x, y) = KF.predict()
        # Draw a rectangle as the predicted object position
        cv2.rectangle(image, (int(x- 15), int(y- 15)), (int(x+ 15), int(y+ 15)), (0,0,255), 2)

        # Update
        (x1, y1) = KF.update(coordinate[0])
        print(x1, y1)
        # Draw a rectangle as the estimated object position
        cv2.rectangle(image, (int(x1- 15), int(y1- 15)), (int(x1+ 15), int(y1+ 15)), (0,255,0), 2)

        cv2.putText(image, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0,255,0), 2)
        cv2.putText(image, "Predicted Position", (int(x + 15), int(y + 10 )), 0, 0.5, (0,0,255), 2)
        cv2.putText(image, "Measured Position", (int(coordinate[0][0] + 15), int(coordinate[1][0] - 15)), 0, 0.5, (0,0,255), 2)

        return x1, y1

    def _fitzpatricks_X(svd):
        """This is from Fitzpatrick, chapter 8, page 470.
        it's used in preference to Arun's equation 13,
        X = np.matmul(svd[2].transpose(), svd[0].transpose())
        to avoid reflections.
        """
        VU = np.matmul(svd[2].transpose(), svd[0])
        detVU = np.linalg.det(VU)

        diag = np.eye(3, 3)
        diag[2][2] = detVU

        X = np.matmul(svd[2].transpose(), np.matmul(diag, svd[0].transpose()))
        return X

   
    def Compute_3D_blob_coord(keys,depth):
        
        key_3D = []
        depth = depth/1000
        for blob in keys:
            u = int(blob.pt[0])
            v = int(blob.pt[1])
            z = depth[v][u]  + 0.0054
            
            #x = (u - CX_DEPTH) * z  / FX_DEPTH
            #y = (v - CY_DEPTH) * z / FY_DEPTH
            xy = uv2xy[v][u]
            XYZ = [xy[0]*z,xy[1]*z,z]/np.linalg.norm([xy[0],xy[1],1])
            #XYZ_sphere = (XYZ *(1 + 0.0054/z)).tolist()
            #print("point from function", [x,y,z])
            key_3D.append(XYZ)
        #print(key_3D)
        
        return key_3D

    def Kabsch_Algorithm (A,B):

        
        N = A.shape[1]
        
        T = np.array([[0.0,0.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0],
                    [0.0,0.0,0.0,1]])
    
        # calculate centroids
        A_centroid = np.reshape(1/N * (np.sum(A, axis=1)), (3,1))
        B_centroid = np.reshape(1/N * (np.sum(B, axis=1)), (3,1))
        
        # calculate the vectors from centroids
        A_prime = A - A_centroid
        B_prime = B - B_centroid
        
        # rotation estimation
        H = np.matmul(A_prime, B_prime.transpose())
        svd = np.linalg.svd(H)

        # Replace Arun Equation 13 with Fitzpatrick, chapter 8, page 470,
        # to avoid reflections, see issue #19
        X = _fitzpatricks_X(svd)

        # Arun step 5, after equation 13.
        det_X = np.linalg.det(X)

        if det_X < 0 and np.all(np.flip(np.isclose(svd[1], np.zeros((3, 1))))):

            # Don't yet know how to generate test data.
            # If you hit this line, please report it, and save your data.
            raise ValueError("Registration fails as determinant < 0"
                            " and no singular values are close enough to zero")

        if det_X < 0 and np.any(np.isclose(svd[1], np.zeros((3, 1)))):
            # Implement 2a in section VI in Arun paper.
            v_prime = svd[2].transpose()
            v_prime[0][2] *= -1
            v_prime[1][2] *= -1
            v_prime[2][2] *= -1
            X = np.matmul(v_prime, svd[0].transpose())

        # Compute output
        R = X
        t = B_centroid - R @ A_centroid
        T[0:3,0:3] = R
        T[0][-1] = t[0]
        T[1][-1] = t[1]
        T[2][-1] = t[2]
        return R, t, T


    def Brute_force_matching(Measured_points):
        
        num_points_to_permute = len(Measured_points)
        permuted_list = list(itertools.permutations(Obj_frame_tool,num_points_to_permute))
        Y = np.transpose(np.array(Measured_points))
        min_err = 1000
        for iter in range(len(permuted_list)):
        
            P = np.transpose(np.array(permuted_list[iter]))
            #print(P)
            Rot, Transl, T = Kabsch_Algorithm (P,Y)
            error = np.linalg.norm(Rot @ P + Transl - Y, 'fro')
            squared_diff = np.square(Rot @ P + Transl - Y)
            mean_squared_error = np.mean(squared_diff)
            rmse_error = np.sqrt(mean_squared_error)
            #print(rmse_error)
            if rmse_error < min_err:
                min_err = rmse_error
                match = P
                match_R = Rot
                match_t = Transl
                T_final = T
        
        print(min_err)
        return match_R,match_t,T_final
            


    def Blob_detector(im,depth,detector):
        
        # Detect blobs.
        
        im_conv = hl2ss_3dcv.rm_depth_undistort(im,calibration.undistort_map)
        im_conv = hl2ss_3dcv.rm_depth_to_uint8(im_conv)
        ret,im_conv_treshold = cv2.threshold(im_conv, 4, 255, cv2.THRESH_BINARY)
        #cv2.imshow('image_trsh', im_conv_treshold)
        #kernel = np.ones((3,3),np.uint8)
        #im_conv = cv2.dilate(im_conv_treshold, kernel, iterations = 1)
        

        depth_undistort = hl2ss_3dcv.rm_depth_undistort(depth,calibration.undistort_map)
        depth_conv = hl2ss_3dcv.rm_depth_to_uint8(depth_undistort)   
        
    
        
        keypoints = detector.detect(im_conv_treshold)

        # Draw the detected circle
        im_with_keypoints = cv2.drawKeypoints(im_conv_treshold, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        '''
        centers = []
        corrected_keypoint = []
        
        for i, keypoint in enumerate(keypoints):
            centers.append(np.array([[keypoint.pt[0]], [keypoint.pt[1]]]))
            

        if (len(centers) > 0):
            # Track object using Kalman Filter
            tracker.Update(centers)
            
            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace)-1):

                        #print("measured",(centers[i]))
                        # Draw trace line
                        #print("updated" + str(j),(tracker.tracks[i].trace[0][0][0],tracker.tracks[i].trace[0][1][0]))
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j+1][0][0]
                        y2 = tracker.tracks[i].trace[j+1][1][0]
                        clr = tracker.tracks[i].track_id % 9
                        cv2.line(im_with_keypoints, (int(x1), int(y1)), (int(x2), int(y2)),5)
                        cv2.rectangle(im_with_keypoints, (int(x1 - 15), int(y1 - 15)), (int(x2 + 15), int(y2 + 15)), (255, 0, 0), 2)
                    corrected_keypoint.append((tracker.tracks[i].trace[0][0][0],tracker.tracks[i].trace[0][1][0]))
                    print(corrected_keypoint)
        
        
        cv2.imshow('image', im_with_keypoints)
        '''
            #im_with_keypoints = cv2.drawKeypoints(im_conv_treshold, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        depth_with_keypoints = cv2.drawKeypoints(depth_conv, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
            #cv2.imshow("Keypoints", im_with_keypoints)
        cv2.imshow("Keypoints_depth", depth_with_keypoints/np.max(depth_conv))
        
        Camera_coord = Compute_3D_blob_coord(keypoints,depth_undistort)

        return Camera_coord
        

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
        R_1 = V @ W_1.T
        R_2 = V @ W_2.T
        R_3 = V @ W_3.T
        

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
            #print(rmse_error)
            

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
    
    
    
    dt = 0.02
    f = KalmanFilter (dim_x=13, dim_z=7)
    f.x = np.array([0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,0.])
    f.F = np.identity(13)
    f.F[0,3]=dt
    f.F[1,4]=dt
    f.F[2,5]=dt
    f.F[6,9]=dt
    f.F[7,10]=dt
    f.F[8,11]=dt
    f.F[9,12]=dt
    f.H = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.]])
    f.P *= 0.002
    f.R = 0.002 * np.identity(7)
    f.R[3,3] = 0.004
    f.R[4,4] = 0.004
    f.R[5,5] = 0.004
    f.R[6,6] = 0.004
    f.Q = 0.002 * np.identity(13)
    
    #2.52473693e-03 5.50279411e-04 8.01799854e-05 variance values
    print(f.F)



    # Set up the SimpleBlobDetector with default parameters
    params = cv2.SimpleBlobDetector_Params()

    # Set the threshold
    #params.minThreshold = 100
    #params.maxThreshold = 500

    # Set the area filter
    params.filterByArea = True
    params.minArea = 0.01
    #params.maxArea = 100
    # Set the circularity filter
    params.filterByCircularity = True
    params.minCircularity = 0.1
    #params.maxCircularity = 1

    # Set the convexity filter
    params.filterByConvexity = True
    params.minConvexity = 0.1
    #params.maxConvexity = 1

    # Set the inertia filter
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    #params.maxInertiaRatio = 1

    # Set the color filter
    params.filterByColor = False
    params.blobColor = 255


    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    tracker = Tracker(60, 5, 3, 4)

    coordinates_obj = []
    coordinates_KF = []
    angles_obj = []
    angles_KF = []
    
    
    HOST = "192.168.0.101"
    PORT = 1000

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((HOST, PORT))
            server_socket.listen()

            print(f"Server listening on {HOST}:{PORT}")

            client_socket, client_address = server_socket.accept()
            print(f"Accepted connection from {client_address}")
    
    
    z = np.array([0.,0.,0.,0.,0.,0.,0.])
    send_info = np.array([0.,0.,0.,0.,0.,0.,0.])
    T_rounded = np.zeros((4,4))
    first_meas=True
    tracked = False
    
  
    while(enable):
        
        #input("press ENTER..")
        sink_depth.acquire()
        _, data = sink_depth.get_most_recent_frame()
        #print(data.pose)
        #T_world_rig = data.pose
        #T_world_rig[:3,-1] = T_world_rig[-1,:3]
        #T_world_rig[-1,:3] = 0
        #\print("T_world_rig", T_world_rig)
        #cv2.imshow('Depth', data.payload.depth / np.max(data.payload.depth)) # Scaled for visibility
        cv2.imshow('AB', data.payload.ab / np.max(data.payload.ab)) # Scaled for visibility


        Camera_frame_tool = Blob_detector(data.payload.ab,data.payload.depth,detector)
        points_fiducials = np.asarray(Camera_frame_tool)
        #print(points_fiducials)
        
        
        '''
        depth_undistort = hl2ss_3dcv.rm_depth_undistort(data.payload.depth,calibration.undistort_map)
        o3d_depth_image = o3d.geometry.Image(depth_undistort)
                
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(hl2ss.Parameters_RM_DEPTH_AHAT.WIDTH, hl2ss.Parameters_RM_DEPTH_AHAT.HEIGHT, calibration.intrinsics[0, 0], calibration.intrinsics[1, 1], calibration.intrinsics[2, 0], calibration.intrinsics[2, 1])

        # Create a point cloud from the depth image
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth_image,
            intrinsic,
            depth_scale=1000.0,  # Adjust based on your specific depth values
            #depth_trunc=0.5,  # Adjust based on your specific depth values
        )
        
        
        pl = pv.Plotter()
        pl.add_points(np.asarray(point_cloud.points), color='blue', point_size=1)
        pl.add_points(points_fiducials, color='red', render_points_as_spheres=True,point_size=10)

        pl.show()
        '''



        if(3<=points_fiducials.shape[0]<=4):

            #tracked=True
            T_depth_to_world = hl2ss_3dcv.camera_to_rignode(calibration.extrinsics) @ hl2ss_3dcv.reference_to_world(data.pose)
            points_patient_world = hl2ss_3dcv.transform(points_fiducials, T_depth_to_world)
            #print(points_patient_world)




            X = np.transpose(np.array(Obj_frame_tool))
            Y = np.transpose(np.array(points_patient_world))

            
            R_brute,t_brute,T_brute = Brute_force_matching(points_patient_world.tolist())
            #R_pca,t_pca,T_pca = PCA_registration(X,Y)

            #point_pca = R_pca @ X + t_pca
            point_brute = R_brute @ X + t_brute
            #print(point_pca)
            #print(point_brute)
            
            
            #pl = pv.Plotter()
            #pl.add_points(np.asarray(point_pca.T), color='blue',  render_points_as_spheres=True,point_size=10)
            #pl.add_points(np.asarray(point_brute.T), color='red', render_points_as_spheres=True,point_size=10)
            #pl.add_points(Y.T, color='green', render_points_as_spheres=True,point_size=10)

            #pl.show()

            '''
        

            coordinates_obj.append(T_brute[0:3,3])
            r = R.from_matrix(T_brute[0:3,0:3])
            quaternions = r.as_quat()
            angles_obj.append(quaternions)
            z[0:3] = T_brute[0:3,3]
            z[3:7] = quaternions
            #print(z)

            if(first_meas == True):
           
                f.x = np.array([z[0], z[1], z[2], 0., 0., 0.,0.,0.,0.,0., 0., 0.,0.])
                first_meas = False
            
            print("position",z[0:3])
            print("angle",z[3:7])  
            
            
            f.predict()

           
            f.update(z)

           
            
            coordinates_KF.append(f.x[0:3])
            angles_KF.append(f.x[6:10])

            send_info[0:3] = f.x[0:3] 
            send_info[3:7] = f.x[6:10] 
            print("position_KF",f.x[0:3]) 
            print("angle_KF",f.x[6:10])

                
            send_info = np.round(send_info, decimals=3)
            #print("send_info",send_info)
    
            matrixString = ','.join(map(str, send_info))
            print(matrixString)
            client_socket.sendall(matrixString.encode("UTF-8"))
            #print("Matrix sent to client")
        '''
            #T_rounded = np.round(T_brute, decimals=3)
            matrixString = '\n'.join([','.join(map(str, row)) for row in  T_brute])
            print(matrixString)
            client_socket.sendall(matrixString.encode("UTF-8"))
            print("Matrix sent to client")
        
        #time.sleep(0.05)

        cv2.waitKey(1) 

    # Assuming quaternion is a 4D vector [qw, qx, qy, qz]
    def quaternion_to_euler(q):
        r = R.from_quat(q)
        euler = r.as_euler('xyz')  # You can choose other sequences like 'zyx'
        return euler

    # Example usage:
    angles_obj = [quaternion_to_euler(q) for q in angles_obj]
    angles_KF = [quaternion_to_euler(q) for q in angles_KF]

    # Extract coordinates for plotting
    x1, y1, z1 = zip(*coordinates_obj)
    x2, y2, z2 = zip(*coordinates_KF)
    x3, y3, z3,  = zip(*angles_obj)
    x4, y4, z4 = zip(*angles_KF)

    # Create a 3D scatter plot for the two paths
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first path
    ax.plot(x1, y1, z1, label='pos_obj', marker='o')

    # Plot the second path
    ax.plot(x2, y2, z2, label='pos_KF', marker='x')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


    # Create a 3D scatter plot for the two paths
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first path
    ax.plot(x3, y3, z3, label='angles_obj', marker='o')

    # Plot the second path
    ax.plot(x4, y4, z4, label='angles_KF', marker='x')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()
            

    #print(temporal_array)
    #client_socket.close()
    #listener.join()
    