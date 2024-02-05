import json
import cv2
import numpy as np
import glob
import os
import pyvista as pv
import cv2
import numpy as np
import pykinect_azure as pykinect
import open3d as o3d
import math
import matplotlib.pyplot as plt
import copy
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t
import socket
import itertools

Obj_frame_tool = [[45.00, -10.00, 90.00],[85.00, 8.00, 40.00],[70.00, -85.00, 65.00],[15.00, -50.00, 120.00],[-30.00, 35.00, 57.00],[-15.00, -105.00, 75.00]]



def color_detector(image, lower, upper ):   ### FIND OPT HSV with interactive_color_detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower, upper)
    detected_color = cv2.bitwise_and(image, image, mask = mask)
    
    return detected_color

def Compute_3D_blob_coord(keys,depth):
    key_3D = []
    depth = depth/1
    # Get the calibration information
  
    for blob in keys:
       u = int(blob.pt[0])
       v = int(blob.pt[1])
       print(u,v)
       z = depth[v][u] 
       print(z)
       point_3D = device.calibration.convert_2d_to_3d(k4a_float2_t((blob.pt[0],blob.pt[1])), z,  K4A_CALIBRATION_TYPE_COLOR,K4A_CALIBRATION_TYPE_DEPTH)
       key_3D.append([point_3D.xyz.x/1000, point_3D.xyz.y/1000, point_3D.xyz.z/1000])

    return key_3D

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

def blob_detector(im,depth):
    """
    given an image this function starts the detection as it is defined by the detector object.
    KEYPOINTS: opencv obj
    CENTROIDS: list of list with centroids pixel coordinates in im frame
    """
    #im = (im/256).astype('uint8')
    depth_conv= (depth/256).astype('uint8')

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 100
    params.maxThreshold = 300
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 1000
    params.filterByCircularity = True
    params.minCircularity = 0.3
    #params.maxCircularity = 1
    params.filterByConvexity = False
    params.minConvexity = 0.1
    #params.maxConvexity = 1
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 1
    params.filterByColor = False
    params.blobColor = 255
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    # ret, im_treshold = cv2.threshold(im, 250, 255, cv2.THRESH_BINARY)
    # cv2.imshow("TRESH", im_treshold)
    keypoints = detector.detect(im)

    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    depth_with_keypoints = cv2.drawKeypoints(depth_conv, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #for blob in keypoints:
        #cv2.circle(im_with_keypoints, (int(blob.pt[0]),int(blob.pt[1])), 1, (0,255,0), -1)

    #cv2.imshow("Keypoints", im_with_keypoints)
    cv2.imshow("Keypoints_depth", depth_with_keypoints)
    
    return(Compute_3D_blob_coord(keypoints,depth))

def Brute_force_matching(Measured_points):
            
        permuted_list = list(itertools.permutations(Measured_points))
        Y = np.transpose(np.array(Obj_frame_tool)/1000)
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


## color detection
N_MARKERS = 6 # set how many markers are we detecting
lower = np.array([0,160,158])
upper = np.array([179,255,255]) # set them up using color picking tool or interactive color detection from opencv
points_3D = []



if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    device = pykinect.start_device(config=device_config)
    print(device_config)
    
    while True:
            HOST = "192.168.0.101"
            PORT = 1000

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                    server_socket.bind((HOST, PORT))
                    server_socket.listen()

                    print(f"Server listening on {HOST}:{PORT}")

                    client_socket, client_address = server_socket.accept()
                    print(f"Accepted connection from {client_address}")

                    counter_send = 0
                    mean_matrix = 0
            while True:

                
               
                # Get capture
                capture = device.update()

                # Get the infrared image
                
                ret, points = capture.get_pointcloud()
                ret2, color_image = capture.get_color_image()
                _, depth_image = capture.get_transformed_depth_image()

                

                

                if not ret:
                    continue
                if not ret2:
                    continue

                print(color_image.shape)
                print(depth_image.shape)




                # color and blob detection
                detected = color_detector(color_image, lower, upper)
               
                detected = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(detected, 1, 255, cv2.THRESH_BINARY)
                

                # returning centroids as lists
                points_3d = blob_detector(binary,depth_image)
                print(points_3d)


                
                '''
                pl = pv.Plotter()
                pl.add_points(np.asarray(points_3d), color="red",render_points_as_spheres=True, point_size=5)
                pl.add_points(np.asarray(points), color="blue", point_size=1)

                pl.show_grid()
                pl.show()
                '''
                
                if(len(points_3d)==6):
                    X = np.transpose(np.asarray(points_3d))
                    Y = np.transpose(np.array(Obj_frame_tool)/1000)


                                
                    R_brute,t_brute,T_brute = Brute_force_matching(points_3d)
                    point_brute = R_brute @ X + t_brute
                    #R_pca,t_pca,T_pca = PCA_registration(X,Y)
                    #point_pca = R_pca @ X + t_pca
                    
                    '''
                    pl = pv.Plotter()
                    pl.add_points(Y.T, color="red",render_points_as_spheres=True, point_size=5)
                    pl.add_points(np.asarray(point_pca.T), color="blue", render_points_as_spheres=True, point_size=5)

                    pl.show_grid()
                    pl.show()
                    '''
                    
                    
                    T_rounded = np.round(np.linalg.inv(T_brute), decimals=3)
                   
                    counter_send+=1
                    mean_matrix += T_rounded
                    
                    if(counter_send == 1):
                       T_mean = mean_matrix/1
                       matrixString = '\n'.join([','.join(map(str, row)) for row in T_mean])
                       print(matrixString)
                       client_socket.sendall(matrixString.encode("UTF-8"))
                       print("Matrix sent to client")
                       counter_send = 0
                       mean_matrix = 0
                    

                    # Press q key to stop
                    if cv2.waitKey(1) == ord('q'):  
                        break

