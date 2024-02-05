
import numpy as np
import open3d as o3d
from scipy.spatial import procrustes
import copy

def Kabsch_Algorithm (A,B):

    
    N = A.shape[1]
    
   
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

    return R, t



def ICP(source, target):
        print("Apply point-to-point ICP")
        point_cloud1 = o3d.geometry.PointCloud()
        point_cloud2 = o3d.geometry.PointCloud()

        point_cloud1.points = o3d.utility.Vector3dVector(source)
        point_cloud2.points = o3d.utility.Vector3dVector(target)

        threshold = 1
        trans_init = np.identity(4)                        
        reg_p2p = o3d.pipelines.registration.registration_icp(
            point_cloud1, point_cloud2, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        return reg_p2p.transformation
  


if __name__ == "__main__":

    A = [[1.0, 2.0, 3.0],[4.0, 5.0, 6.0],[7.0, 8.0, 9.0],[10.0, 11.0, 12.0]]
    B = [[2.0, 3.0, 4.0],[5.0, 6.0, 7.0],[8.0, 9.0, 10.0],[11.0, 12.0, 13.0]]

    A = np.transpose(np.array(A))
    B = np.transpose(np.array(B))
    print(A)
    
    R,t = Kabsch_Algorithm (A,B)

    C = R @ A + t
   
    print(C)
