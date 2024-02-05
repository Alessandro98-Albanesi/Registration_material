import itertools
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import MerweScaledSigmaPoints as SigmaPoints
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import pyvista as pv
import copy
import math
import socket

host2 = "127.0.0.1"  # Server IP address
port = 3816  # Port number


#sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#sock.connect((host2, port))
#print(f"Connection from {sock}")


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

def compute_fre(fixed, moving, rotation, translation):
    """
    Computes the Fiducial Registration Error, equal
    to the root mean squared error between corresponding fiducials.

    :param fixed: point set, N x 3 ndarray
    :param moving: point set, N x 3 ndarray of corresponding points
    :param rotation: 3 x 3 ndarray
    :param translation: 3 x 1 ndarray
    :returns: Fiducial Registration Error (FRE)
    """
    # pylint: disable=assignment-from-no-return

    transformed_moving = np.matmul(rotation, moving) + translation
    squared_error_elementwise = np.square(fixed
                                          - transformed_moving)
    square_distance_error = np.sum(squared_error_elementwise, 1)
    sum_squared_error = np.sum(square_distance_error, 0)
    mean_squared_error = sum_squared_error / fixed.shape[0]
    fre = np.sqrt(mean_squared_error)
    return fre

  
    
def arun(A, B):
    """Solve 3D registration using Arun's method: B = RA + t
    """
    N = A.shape[1]
    assert B.shape[1] == N

    # calculate centroids
    A_centroid = np.reshape(1/N * (np.sum(A, axis=1)), (3,1))
    B_centroid = np.reshape(1/N * (np.sum(B, axis=1)), (3,1))

    # calculate the vectors from centroids
    A_prime = A - A_centroid
    B_prime = B - B_centroid

    # rotation estimation
    H = np.zeros([3, 3])
    for i in range(N):
        ai = A_prime[:, i]
        bi = B_prime[:, i]
        H = H + np.outer(ai, bi)
    U, S, V_transpose = np.linalg.svd(H)
    V = np.transpose(V_transpose)
    U_transpose = np.transpose(U)
    R = V @ np.diag([1, 1, np.linalg.det(V) * np.linalg.det(U_transpose)]) @ U_transpose

    # translation estimation
    t = B_centroid - R @ A_centroid
    
    T = np.identity(4)
    T[0:3,0:3] = R
    T[0][-1] = t[0]
    T[1][-1] = t[1]
    T[2][-1] = t[2]

    return R, t, T




Obj_points = [[0,0,0],[0,-0.09,-0.025], [0,-0.045,-0.05],[0,-0.14,-0.025]]
Measured_points = [[-0.01956693, -0.09714836,  0.2534  ],[ 0.01630225, -0.14291672,  0.2654  ],[ 0.0818298,  -0.18074061,  0.2854    ],[ 0.06994404, -0.26792893,  0.3094   ]]
permuted_list = list(itertools.permutations(Measured_points))

X = np.transpose(np.array(Measured_points))
Y = np.transpose(np.array(Obj_points))
min_err = 1000



for iter in range(len(permuted_list)):

    P = np.transpose(np.array(permuted_list[iter]))
    #print(P)
    Rot, Transl, T= arun (P,Y)
    
    error = np.linalg.norm(Rot @ P + Transl - Y, 'fro')
    #print(error)
    
    if error < min_err:
        min_err = error
        match = P
        match_R = Rot
        match_t = Transl
        match_T = T

#result = KF_based_registration(match,Y,match_R)

#print(match)
#print(result)
#print(match_R)
#print(min_err)
print(match)
C = match_R @ match + match_t
squared_diff = np.square(C - Y)
mean_squared_error = np.mean(squared_diff)
rmse_error = np.sqrt(mean_squared_error)
print("RMSE_Arun",rmse_error)




def compute_R(theta_x, theta_y, theta_z):
        
        
        Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])

        Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
        ])

        R = np.dot(Rx, np.dot(Ry, Rz))

        return R



U = np.random.uniform(-0.250, 0.250, size=(3, 4))

rotation_angles = np.random.uniform(-90, 90, size=(3,))


translation = np.random.uniform(-0.090, 0.090, size=(3,))


R_gener = compute_R(rotation_angles[0], rotation_angles[1], rotation_angles[2])
t_gener = translation


E = R_gener @ U + t_gener.reshape(-1,1)

covariance = 0.001 * np.eye(3)  # Identity matrix representing the covariance matrix
noise = np.random.normal(0, 0.004, E.shape)


U_noisy = U + noise



def compute_sigma_points(x, P, alpha, beta, lambda_):
    """
    Compute sigma points for a 6x1 vector.

    Parameters:
    - x: 6x1 mean vector
    - P: 6x6 covariance matrix
    - alpha, beta, kappa: Parameters for controlling the spread of sigma points

    Returns:
    - sigma_points: Array of sigma points, each column represents a sigma point
    """
    n = len(x)  # Dimension of the state vector 

    # Compute the sigma points
    sigma_points = np.zeros((n, 2 * n + 1))
    # Compute sigma points as columns of matrix X
    #sqrt_P = np.linalg.cholesky(P)
    V,S,V_t = np.linalg.svd(P)
    sqrt_S = np.diag(np.sqrt(S))
    sqrt_P = V @ sqrt_S @ V_t


    sigma_points[:, 0] = x  # Mean as the first sigma point

    for i in range(n):
    
        sigma_points[:, i + 1] = x + np.sqrt((n + lambda_))*sqrt_P[:, i]
        sigma_points[:, n + i + 1] = x - np.sqrt((n + lambda_))*sqrt_P[:, i]

    
    
    return sigma_points


def find_closest_point(matrix1, matrix2):
    
    N = matrix2.shape[1]

    # Calculate the Euclidean distance between each point in matrix2 and all points in matrix1
    distances = cdist(matrix2.T, matrix1.T, metric='euclidean')

    # Initialize arrays to store results
    assigned_indices = set()
    closest_points_matrix1 = np.zeros((3, N))
    indices = []
    # Find unique closest points
    for i in range(N):
        min_distance_index = np.argmin(distances[i])
        
        # Check if the closest point has already been assigned
        while min_distance_index in assigned_indices:
            # Set the distance of the already assigned point to infinity
            distances[i, min_distance_index] = np.inf
            min_distance_index = np.argmin(distances[i])
        
        # Assign the closest point to the set of assigned indices
        assigned_indices.add(min_distance_index)
        
        # Update the closest points matrix
        closest_points_matrix1[:, i] = matrix1[:, min_distance_index]
        indices.append(min_distance_index)
    #print(indices)
    return closest_points_matrix1



def UKF(U_init, Y):
        
        x_k_posterior = np.array([0,0,0,0,0,0])
        P_k_posterior = np.identity(6)
        sigma_x = 0.001  # Example value, adjust as needed
        sigma_y = 0.001  # Example value, adjust as needed
        sigma_z = 0.001  # Example value, adjust as needed
        # Create covariance matrix
        covariance_matrix_noise = np.diag([sigma_x, sigma_y, sigma_z])
        variances_fixed_points = np.var(Y, axis=1)
        w_mean = np.zeros(13)
        Wc = np.zeros(13)
        alpha = 1
        k = 3
        lambda_ = alpha**2 * (6 + k) - 6
        beta = 2
        w_mean[0] = lambda_ / (6 + lambda_)
        w_mean[1:13] = 1 / (2 * (6 + lambda_))
        Wc[0] = w_mean[0] + (1 - alpha**2 + beta)
        Wc[1:13] = w_mean[1:13]    
        covariance_of_process_model = np.diag([sigma_x,sigma_y,sigma_z,180/((np.sqrt(variances_fixed_points[1]/sigma_z)+(np.sqrt(variances_fixed_points[2]/sigma_y)))),180/((np.sqrt(variances_fixed_points[0]/sigma_z)+(np.sqrt(variances_fixed_points[2]/sigma_x)))),180/((np.sqrt(variances_fixed_points[0]/sigma_y)+(np.sqrt(variances_fixed_points[1]/sigma_x))))]) 
        U = U_init
        treshold = 0.001
        fre = 1
        
        '''
        print("lambda", lambda_)
        print("w_mean",w_mean)
        print("Wc",Wc)
        print(np.sum(w_mean))
        '''

        

        while(fre > treshold):
            print(fre)
            #previously_selected_points = np.zeros((0, 1))
            estimated_points = np.zeros((0, 1))

            # PREDICTION
                        
            x_k_prior = x_k_posterior

            P_k_prior = P_k_posterior + covariance_of_process_model

            sigma_x_points = compute_sigma_points(x_k_prior, P_k_prior, alpha, beta, lambda_)

        
            R = compute_R(x_k_prior[3], x_k_prior[4], x_k_prior[5])
            
            y_real = R @ U + x_k_prior[:3].reshape(-1, 1)
            y_real = y_real.reshape((3*U.shape[1],),order='F')

            for i in range(U.shape[1]):

                #previously_selected_points = np.vstack((previously_selected_points,U[:,i].reshape(-1,1)))

                y_k_prior = R @ U[:,i].reshape(-1, 1) + x_k_prior[:3].reshape(-1, 1)
                estimated_points = np.vstack([estimated_points,y_k_prior])
            
            #Compute the propagated sigma points sigma_y
            
            sigma_y_points = np.zeros((3*U.shape[1],13))


            for i in range(13):
                
                state_k = sigma_x_points[:,i]
                R_sigma = compute_R(state_k[3], state_k[4], state_k[5])
                t_sigma = np.array([state_k[0],state_k[1],state_k[2]]).reshape(-1,1)
                state_y = R_sigma @ U + t_sigma
                sigma_y_points[:, i] = state_y.reshape((3*U.shape[1],),order='F')
            
            # Compute Pxy Py
            Pxy = 0
            Py = 0
            
            y_mean = 0
            
            for i in range(13):
                y_mean = y_mean + w_mean[i] * sigma_y_points[:, i]
            
            
            for i in range(13): 
                Py = Py + Wc[i] * (sigma_y_points[:, i] - y_mean.reshape(-1,1)) @ (sigma_y_points[:, i] - y_mean.reshape(-1,1)).T
                Pxy = Pxy + Wc[i] * (sigma_x_points[:,i].reshape(-1,1) - x_k_prior.reshape(-1,1)) @ (sigma_y_points[:, i].reshape(-1,1) - y_mean.reshape(-1,1)).T

               
            K_k = Pxy @ np.linalg.pinv(Py)
        
            #find the closest point in Y to the estimated y

            y_estimated = R @ U + x_k_prior[:3].reshape(-1, 1)
            closest_points_matrix1 = find_closest_point(Y,y_estimated)
            closest_points_matrix1 = closest_points_matrix1.reshape((3*closest_points_matrix1.shape[1],1),order='F')
            y_estimated = y_estimated.reshape((3*y_estimated.shape[1],1),order='F')
            
            
            
            #CORRECTION
            x_k_posterior = x_k_prior.reshape(-1,1) + K_k @ (closest_points_matrix1 - y_estimated)
            x_k_posterior = x_k_posterior.reshape(6)
            P_k_posterior = P_k_prior - K_k @ Py @ K_k.T
            #print(P_k_posterior)

            R = compute_R(x_k_posterior[3], x_k_posterior[4], x_k_posterior[5])
            t = np.array([x_k_posterior[0],x_k_posterior[1],x_k_posterior[2]]).reshape(-1,1)

            U = R @ U + t
            
            fre = compute_fre(Y,U,R,t)
            
        


        T = np.identity(4)

        T[0:3,0:3] = R
        T[0,3] = x_k_posterior[0]
        T[1,3] = x_k_posterior[1]
        T[2,3] = x_k_posterior[2]


        return T, R, t

#T, R, t = UKF(U_init,E)
#fre1 = compute_fre(E,U_init,R,t)

#print(fre1)
#print(fre2)

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


R_pca,t_pca,T_pca = PCA_registration(X,Y)









#import the first mesh
#mesh = pv.read("C:/Users/Alessandro/Desktop/Neuro/face_3t_mWtextr.obj")
#mesh = mesh.decimate(0.1)
mesh1 = o3d.io.read_triangle_mesh("C:/Users/Alessandro/Desktop/Neuro/face_3t_mWtextr.obj")
filtered_pca = hiddenPointRemoval(mesh1)
vertices = np.array(filtered_pca.points)  # Transpose for a 3xN matrix
reduction_factor = 1 # Adjust as needed
downsampled_points = vertices[np.random.choice(vertices.shape[0], int(reduction_factor * vertices.shape[0]), replace=False)]
downsampled_points_1 = downsampled_points.T
print(downsampled_points_1)

#import the second mesh
mesh = pv.read("C:/Users/Alessandro/Desktop/Neuro/pcl_holo.ply")
#mesh = mesh.decimate(1)
vertices = np.array(mesh.points)  # Transpose for a 3xN matrix
reduction_factor = 1  # Adjust as needed
downsampled_points = vertices[np.random.choice(vertices.shape[0], int(reduction_factor * vertices.shape[0]), replace=False)]
downsampled_points_2 = downsampled_points.T
downsampled_points_2 = downsampled_points_2 * 0.6
print(downsampled_points_2)
#points_markers = np.array([[30.401019818373427, 99.34523712416595, 148.0], [21.267892448087032, 88.72368428594672, 144.0], [-8.474137969845257, 74.62251139666022, 156.0], [61.15348910659576, 61.00730867742305, 148.0], [3.148342685477857, 53.440942614801045, 152.0], [61.15348910659576, 30.826386283178625, 148.0], [-1.107926506677632, 17.833071500787707, 136.0], [2.8169381922696615, 5.84006800738452, 136.0], [43.28512929947284, 12.551242532481423, 128.0], [2.8997893155717107, 1.3821826296335704, 140.0], [-1.8928994464670907, 8.838318880735317, 136.0], [2.0917289363766796, -3.2474694542169242, 140.0], [31.860936964479635, -3.1546846126678694, 136.0], [15.020695003497178, -3.2474694542169242, 140.0], [29.506018145111256, -4.653810049343267, 136.0], [22.293238416252457, -12.506773621917914, 140.0], [14.212634624302146, -12.506773621917914, 140.0], [37.35574754300585, 0.5931289790206264, 136.0]]).T
#points_markers = points_markers/1000

R_pca,t_pca,T_pca = PCA_registration(downsampled_points_2,downsampled_points_1)
registered_pca = R_pca @ downsampled_points_2 + t_pca

# Create a plotter


cloud_registered = pv.PolyData(registered_pca.T)
cloud_target = pv.PolyData(downsampled_points_1.T)
plotter = pv.Plotter()



source_cloud = o3d.geometry.PointCloud()
source_cloud.points = o3d.utility.Vector3dVector(downsampled_points_2.T)  # Transpose for correct shape

target_cloud = o3d.geometry.PointCloud()
target_cloud.points = o3d.utility.Vector3dVector(downsampled_points_1.T)
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
points_patient = pv.PolyData(np.asarray(mesh1.vertices))

plotter.add_mesh(points_patient, color="green", point_size=1)
#plotter.add_mesh(icp_regist, color="blue", point_size=5)
#plotter.add_mesh(cloud_registered, color="red", point_size=5)
#plotter.add_mesh(cloud_target, color="green", point_size=5)
plotter.show()



T_world_rig = np.array([[ 0.13602746, -0.9899998,  -0.03737512,  0.44044223],
 [-0.07037723, -0.04728663,  0.99639904, -0.20265168],
 [-0.9882022,  -0.1329073,  -0.07610571, -0.12694663],
 [ 0,          0,          0,          1,        ]])




T_to_send = T_world_rig @ np.linalg.inv(refined_transform)
print(T_to_send)
#T, R, t = UKF(registered_pca,downsampled_points_1)

matrixString = '\n'.join([','.join(map(str, row)) for row in T_to_send])
print(matrixString)
#sock.sendall(matrixString.encode("UTF-8"))






tranformed_vertices = R_gener @ downsampled_points + t_gener.reshape(-1,1)
noise = np.random.normal(0, 0.002, tranformed_vertices.shape)
tranformed_vertices = tranformed_vertices + noise

noise = np.random.normal(0, 0.001, downsampled_points.shape)
#downsampled_points = downsampled_points + noise


permutation = np.random.permutation(downsampled_points.shape[1])
downsampled_points = downsampled_points[:, permutation]

R_pca,t_pca,T_pca = PCA_registration(downsampled_points,tranformed_vertices)
registered_pca = R_pca @ downsampled_points + t_pca


R_arun,t_arun,T_arun = arun(downsampled_points,tranformed_vertices)
registered_arun = R_arun @ downsampled_points + t_arun

#T, R, t = UKF(downsampled_points,tranformed_vertices)

source_cloud = o3d.geometry.PointCloud()
source_cloud.points = o3d.utility.Vector3dVector(downsampled_points.T)  # Transpose for correct shape

target_cloud = o3d.geometry.PointCloud()
target_cloud.points = o3d.utility.Vector3dVector(tranformed_vertices.T)


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh



voxel_size = 0.1
source_down, source_fpfh = preprocess_point_cloud(source_cloud, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target_cloud, voxel_size)


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.3),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
#print(result_ransac.transformation)


result = o3d.pipelines.registration.registration_icp(
    source_cloud, target_cloud,  # Source and target point clouds
    1,  # Maximum correspondence distance (increase if points are far apart)
    T_pca,  # Initial transformation guess
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),  # Point-to-point ICP
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)  # Max iterations
)

refined_transform = result.transformation

icp_regist = source_cloud.transform(refined_transform)

# Compute the RMSE
evaluation = o3d.pipelines.registration.evaluate_registration(
    icp_regist, target_cloud,
    max_correspondence_distance=0.1  # Adjust as needed
)

rmse_icp = evaluation.inlier_rmse
print("RMSE_icp:", rmse_icp)


icp_regist = np.asarray(icp_regist.points).T

#cloud1 = pv.PolyData(downsampled_points.T)
cloud2 = pv.PolyData(tranformed_vertices.T)
cloud3 = pv.PolyData(registered_pca.T)
cloud4 = pv.PolyData(registered_arun.T)
cloud5 = pv.PolyData(icp_regist.T)


# Create a plotter
plotter = pv.Plotter()

# Add each point cloud to the plot
#plotter.add_mesh(cloud1, color="red", point_size=1, opacity=0.8, render_points_as_spheres=True)
plotter.add_mesh(cloud2, color="green", point_size=2, opacity=1, render_points_as_spheres=True)
plotter.add_mesh(cloud3, color="blue", point_size=2, opacity=1, render_points_as_spheres=True)
#plotter.add_mesh(cloud4, color="yellow", point_size=1, opacity=0.8, render_points_as_spheres=True)
plotter.add_mesh(cloud5, color="red", point_size=2, opacity=1, render_points_as_spheres=True)

plotter.show()


R_pca,t_pca,T_pca = PCA_registration(U_noisy,E)
#registered_pca = R_pca @ downsampled_points + t_pca
U_init = R_pca @ U + t_pca
#T, R, t = UKF(U_init,E)







