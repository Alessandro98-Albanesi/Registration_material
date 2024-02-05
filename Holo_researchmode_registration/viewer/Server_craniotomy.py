import numpy as np
import socket
import struct
import array
import pickle
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pyvista as pv
import open3d as o3d
from scipy.spatial import KDTree
import time
import math
import datetime
from pynput import keyboard
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
import threading

# HoloLens address
host = "192.168.0.102"

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
    
   
    def computeTumorProj(Avg_nearbyPts,CoM_tumor,tumor, growValue, face, update ):
        dir_COM_skin=Avg_nearbyPts-CoM_tumor
        direction= dir_COM_skin / np.linalg.norm(dir_COM_skin)
        #vector_dir=np.ones((len(tumor.points),3))*direction

        pts_moved=tumor.points+(direction*2)
        #trova punti di intersezione per ogni punto del tumore

        #rateofExpansion to add tot cm= growValue to each point
        bounds=tumor.bounds
        Exp_perc=((bounds[1]-bounds[0])+growValue)/(bounds[1]-bounds[0])

        #pts_expandedMoved=(tumor.points*Exp_perc)+(direction*100)
        #pts_expanded=(tumor.points*Exp_perc)
        pts_expanded=(tumor.points*Exp_perc)
        com_expanded=pv.PolyData(pts_expanded).center_of_mass()
        #recentering the data
        pts_expanded=pv.PolyData(pts_expanded-(com_expanded-CoM_tumor), tumor.faces)
        pts_expandedMoved=pts_expanded.points+(direction*2)
        i=0
        projectedPts=[]
        Prj_expandedPts=[]
        
        '''
        p = pv.Plotter()
        color_Nexp = "aliceblue"
        p.add_mesh(face, color=lightpink2, opacity=0.35, smooth_shading=True)  # face
        p.add_mesh(tumor, color=color_Nexp, opacity=0.95, smooth_shading=True)  # tumor
        p.add_mesh(CoM_tumor, color="black", smooth_shading=True)  # com tumor
        p.add_mesh(Avg_nearbyPts, color="blue", smooth_shading=True )
        p.add_mesh(nearby_points, color=True, smooth_shading=True )
        p.add_mesh(ray, color="dimgrey", line_width=5, label="Ray Segment")  # traj
        p.camera.roll += 0
        p.camera.azimuth += 20
        p.camera.elevation -= 5
        p.set_background("white")
        p.show()
        '''
        for pts in tumor.points:
            str=pts.tolist()
            end=pts_moved[i].tolist()
            prjPt, prj_ind = face.ray_trace(str, end)
            # Create geometry to represent ray trace
            projectedPts.append(prjPt)

            # #create the expanded model by 2cm
            expandedPt=pts_expanded.points[i].tolist()
            expandedPt_moved = pts_expandedMoved[i].tolist()
            prjPtExp, prj_indExp = face.ray_trace(expandedPt, expandedPt_moved)
            Prj_expandedPts.append(prjPtExp)
            i += 1

        # for pts in pts_expanded:
        #     str = pts.tolist()
        #     end = pts_expandedMoved[i].tolist()
        #     prjPt, prj_ind = face.ray_trace(str, end)
        #     # Create geometry to represent ray trace
        #     Prj_expandedPts.append(prjPt)
        #     i+=1


        projectedPtsPD=pv.PolyData(np.asarray(projectedPts).squeeze())
        projectedPtsExpPD=pv.PolyData(np.asarray(Prj_expandedPts).squeeze())

        proj_surf = projectedPtsPD.delaunay_2d(alpha=1)
        projExp_surf = projectedPtsExpPD.delaunay_2d(alpha=1)



        #do the same but with a tumor expanded by tot cm about 2cm (20mm)

        color_exp="dodgerblue"
        color_Nexp="aliceblue"
        #color_Nexp="blue"
        show=0
        if show==1:
            p = pv.Plotter()
            #p.add_mesh(face,scalars="distances", opacity=0.95, smooth_shading=True, cmap=my_colormap)
            p.add_mesh(face, color=lightpink2, opacity=0.35, smooth_shading=True)  #face
            p.add_mesh(tumor, color=color_Nexp, opacity=0.95, smooth_shading=True)        #tumor
            p.add_mesh(CoM_tumor, color="black", smooth_shading=True )         #com tumor
            #p.add_mesh(Avg_nearbyPts, color=True, smooth_shading=True )
            #p.add_mesh(nearby_points, color=True, smooth_shading=True )
            p.add_mesh(ray, color="dimgrey", line_width=5, label="Ray Segment")    #traj
            #p.add_mesh(intersection, color="blue", point_size=10, label="Intersection Points")
            #p.add_mesh(pts_moved, color="blue", line_width=5, label="Ray Segment")
            #p.add_mesh(pts_expanded, color=color_exp, opacity=0.45, label="Ray Segment")   #tumore expanded
            #p.add_mesh(projectedPtsPD, color="blue", point_size=2, label="Intersection Points")

            #p.add_mesh(projectedPtsExpPD, color="white", point_size=2, label="Intersection Points")
            p.add_mesh(proj_surf, color="cornflowerblue", opacity=0.99, smooth_shading=True)          #projection tumore
            p.add_mesh(projExp_surf, color="hotpink", opacity=0.99, smooth_shading=True)        #projection expanded tum
            p.camera.roll+= 0
            p.camera.azimuth+=20
            p.camera.elevation-=5
            p.set_background("white")
            p.show()
        #from mm to m
        proj_surf.points=proj_surf.points/units_obj
        projExp_surf.points = projExp_surf.points / units_obj
        decimation=1
        nPt=400
        reduction=0.96
        # proj_surf = proj_surf.decimate(reduction)
        # projExp_surf = projExp_surf.decimate(reduction)
        decimProj_surf = proj_surf
        decimProj_surfExp = projExp_surf

        if decimation:
            while (len(decimProj_surf.points)>nPt and len(decimProj_surfExp.points>nPt)):
                decimProj_surf = proj_surf.decimate(reduction)
                decimProj_surfExp = projExp_surf.decimate(reduction)
                print("reduction=", reduction, "  pt proj exp =", len(decimProj_surf.points), "  pt proj  =",
                    len(decimProj_surfExp.points))
                reduction = reduction + 0.005

        proj_surf=decimProj_surf
        projExp_surf=decimProj_surfExp

        pl = pv.Plotter()
        _ = pl.add_mesh(proj_surf)
        if update==1:
            pl.export_obj('C:/Users/Alessandro/Desktop/Neuro/tumorProjUpdate.obj')
        else:
            pl.export_obj('C:/Users/Alessandro/Desktop/Neuro/tumorProj.obj')
        pl.close()
        pl = pv.Plotter()
        _ = pl.add_mesh(projExp_surf)
        if update == 1:
            pl.export_obj('tumorProjExpUpdate.obj')
        else:
            pl.export_obj('tumorProjExp.obj')
        pl.close()
        now = datetime.datetime.now()
        # Format the date and time into a string
        date_time_str = now.strftime("%Y-%m-%d_%H-%M")

        proj_surf.save("C:/Users/Alessandro/Desktop/Neuro"+TumorSavedName+date_time_str+".stl")
        projExp_surf.save("C:/Users/Alessandro/Desktop/Neuro"+TumorExpSavedName+date_time_str+".stl")

        return proj_surf, projExp_surf, Avg_nearbyPts, CoM_tumor
    
    def ObjLoader(fileName):
        obj=[]
        faces=[]
        vertices=[]
        obj_type=0
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2])*(-1), float(line[index2:index3]), float(line[index3:-1]))
                    # vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    vertices.append(vertex)



                elif line[0] == "f":
                    try:
                        string = line.replace("//", "/")
                        ##
                        i = 0
                        # face = []
                        for k in range(3):
                            i = string.find("/", i + 1)
                            face=(int(string[i + 1:string.find(" ", i + 1)])-1)
                            faces.append(face)
                        obj_type=1
                    except:
                        index1 = line.find(" ") + 1
                        index2 = line.find(" ", index1 + 1)
                        index3 = line.find(" ", index2 + 1)
                        face = (int(line[index1:index2])-1 , int(line[index2:index3])-1, int(line[index3:-1])-1)
                        faces.append(face)

            if obj_type==1:
                nFace_tumorPrj = len(faces)
                nPt_tumorPrj = len(vertices)
                obj.extend([nPt_tumorPrj, nFace_tumorPrj])
                obj.extend(np.asarray(vertices).flatten().tolist())
                obj.extend(faces)
            else:
                nFace_tumorPrj = len(faces) * 3
                nPt_tumorPrj = len(vertices)
                nPt_tumorPrj = len(vertices)
                obj.extend([nPt_tumorPrj, nFace_tumorPrj])
                obj.extend(np.asarray(vertices).flatten().tolist())
                obj.extend(np.asarray(faces).flatten().tolist())


            f.close()
            return obj
        except IOError:
           print(".obj file not found.")
    
    TumorSavedName="tumorProj"
    TumorExpSavedName="tumorProjExp"

    units_obj=1 #1 if obj are already in meters, 1000 if they are in mm
    face = pv.read("C:/Users/Alessandro/Desktop/Modelli_neuro_obj/face_3t_mWtextr.obj")
    tumor = pv.read("C:/Users/Alessandro/Desktop/Modelli_neuro_obj/tumor2_m2.obj")



    tumorDec=1
    reduction=0.26
    nPt_Tumor=5000
    decimTumor=tumor
    if tumorDec==1:
        while (len(decimTumor.points) > nPt_Tumor  ):
            decimTumor = tumor.decimate(reduction)
            print("reduction=", reduction, "  pt tumor =", len(decimTumor.points), )
            reduction = reduction + 0.1
    tumor=decimTumor
    #identify the projection of the tumor on the skin
    fileNameTP="C:/Users/Alessandro/Desktop/Neuro/tumorProj.obj"
    fileNameTPE="C:/Users/Alessandro/Desktop/Neuro/tumorProjExp.obj"

    fileNameTP_Update="C:/Users/Alessandro/Desktop/Neuro/tumorProjUpdate.obj"
    fileNameTPE_Update="C:/Users/Alessandro/Desktop/Neuro/tumorProjExpUpdate.obj"    

    growValue=0.010*units_obj # intorno di 1cm oltre alla lesione #20 mm

    #calcolo la distanza tra il tumore e la faccia
    tree = KDTree(tumor.points)
    d_kdtree, idx = tree.query(face.points) #d_kdtree ha stessa dim di Face, e ogni punto ha la distanza rispetto al tumore
    face["distances"] = d_kdtree
    np.mean(d_kdtree)

    treshold=face["distances"].min()
    pts_belowDist=np.argwhere(face["distances"]<=treshold)
    nearbySurf=face.extract_points(face["distances"]<=treshold) #estraggo i punti della faccia che hanno una distanza dal tumore inferiore ad una certa soglia


    nearby_points=nearbySurf.points
    Avg_nearbyPts=nearbySurf.center_of_mass() #prendo il centro di questi punti vicini
    CoM_tumor=tumor.center_of_mass() #centro del tumore


    # Define the colors we want to use
    blue = np.array([12 / 256, 238 / 256, 246 / 256, 1.0])
    green = np.array([0.0, 1.0, 0.0, 1.0])
    grey = np.array([189 / 256, 189 / 256, 189 / 256, 1.0])
    yellow = np.array([255 / 256, 247 / 256, 0 / 256, 1.0])
    red = np.array([1.0, 0.0, 0.0, 1.0])
    lightpink2=np.array((247/256, 188/256, 196/256, 0.5))
    mapping = np.linspace(face["distances"].min(), face["distances"].max(), 256)
    newcolors = np.empty((256, 4))
    newcolors[mapping > treshold] = lightpink2
    newcolors[mapping <= treshold] = green
    my_colormap = ListedColormap(newcolors)

    #start and stop punti per tracciare la traiettoria
    start = CoM_tumor.tolist()
    dist=math.sqrt((Avg_nearbyPts[0]-CoM_tumor[0])*2*units_obj+(Avg_nearbyPts[1]-CoM_tumor[1])*2*units_obj+(Avg_nearbyPts[2]-CoM_tumor[2])*2*units_obj)
    Vdir=(Avg_nearbyPts-CoM_tumor)/dist
    stop = (Avg_nearbyPts+ Vdir*0.30*units_obj).tolist()

    # Perform ray trace between the center of the tumor and the stop point outside
    points, ind = face.ray_trace(start, stop)

    # Create geometry to represent ray trace
    ray = pv.Line(start, stop)
    intersection = pv.PolyData(points)

    computeProjections=0
    if computeProjections==1:
        startTime=time.time()
        update=1
        (proj_surf, projExp_surf, Avg_nearbyPts, CoM_tumor)=computeTumorProj(Avg_nearbyPts,CoM_tumor,tumor, growValue, face, update )
        print("exec time  took %.3f sec.\n" % (time.time() - startTime))
        #salva file non è necessario se poi mando solo vertici e connettività a unity
        
    def handle_client(client_socket, client_address, server_number):
            print(f"Accepted connection from {client_address} on Server {server_number}")

            while True:

                if(server_number==1):

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
                    max_bound = np.array([math.inf,-0.05,0.8])
                    
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
                    

                    mesh = o3d.io.read_triangle_mesh("C:/Users/Alessandro/Desktop/Modelli_neuro_obj/face_3t_mWtextr.obj")
                    filtered_pca = hiddenPointRemoval(mesh)
                    vertices = np.array(filtered_pca.points)  # Transpose for a 3xN matrix
                    reduction_factor = 1 # Adjust as needed
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
                    
                    # Convert Open3D mesh to PyVista mesh
                    
                    #pyvista_mesh = pv.add_mesh_from_open3d(mesh)


                    plotter.add_mesh(icp_regist, color="blue", point_size=5)
                    #plotter.add_mesh(cloud_registered, color="red", point_size=5)
                    #plotter.add_mesh(pyvista_mesh, color="pink")
                    plotter.add_mesh(cloud_target, color="pink", point_size=5)
                    
                    plotter.show(interactive=True)
                    
                    
                    T_CT_to_world = np.linalg.inv(refined_transform)

                    matrixString = '\n'.join([','.join(map(str, row)) for row in T_CT_to_world ])
                    print(matrixString)

                    
                    with client_socket:
                        client_socket.sendall(matrixString.encode("UTF-8"))
                        print("Matrix sent to client")
                        
                    client_socket.close()
                    break
                
                if(server_number==2):
                    data = client_socket.recv(400000)
                    try:
                        arr = array.array('f', data)
                        if data:
                            print(f"Received {data!r}")
                            ptRcv = arr.tolist()
                            id_button = int(ptRcv[0])
                            if id_button==5:
                                print("id_button5")
                                stopUp = [ptRcv[1]*-units_obj, ptRcv[2]*units_obj, ptRcv[3]*units_obj]
                                startUp = [ptRcv[4]*-units_obj, ptRcv[5]*units_obj, ptRcv[6]*units_obj]
                                pickle.dump(stopUp, open("stopUp.p", "wb"))
                                pickle.dump(startUp, open("startUp.p", "wb"))
                                update=1
                                (proj_surf, projExp_surf, Avg_nearbyPts, CoM_tumor) = computeTumorProj(np.asarray(stopUp), np.asarray(startUp), tumor, growValue, face,update)
                                update = 0
                                client_socket.sendall(b"done")




                    except:
                        if data:
                            print(f"Received {data!r}") #da 2 in poi str(data)[2:...] i primi due caratteri sono -> b/
                            #
                            if str(data)[2:5]=="tpr":
                                obj = ObjLoader(fileNameTP)
                                tumorPrj2Send = np.asarray(obj, dtype=float)
                                info2send=struct.pack('f' * len(tumorPrj2Send), *tumorPrj2Send)
                                client_socket.sendall(info2send)
                                print("sent tpr")
                            if str(data)[2:5]=="tpe":
                                obj_TPE = ObjLoader(fileNameTPE)
                                tumorPrjExp2Send = np.asarray(obj_TPE, dtype=float)
                                info2send=struct.pack('f' * len(tumorPrjExp2Send), *tumorPrjExp2Send)
                                client_socket.sendall(info2send)
                                print("sent tpe")
                            if str(data)[2:5]=="trj":
                                trj2send=[]
                                trj2send.extend(start)
                                trj2send.extend(stop)
                                print("start is", start)
                                print("stop is", stop)
                                trj2send = np.asarray(trj2send, dtype=float)
                                info2send = struct.pack('f' * len(trj2send), *trj2send)
                                client_socket.sendall(info2send)
                            if str(data)[2:5]=="utp": #updated tumor projection
                                obj = ObjLoader(fileNameTP_Update)
                                tumorPrj2Send = np.asarray(obj, dtype=float)
                                info2send = struct.pack('f' * len(tumorPrj2Send), *tumorPrj2Send)
                                client_socket.sendall(info2send)

                            if str(data)[2:5]=="ute": #updated tumor projection expanded
                                obj_TPE = ObjLoader(fileNameTPE_Update)
                                tumorPrjExp2Send = np.asarray(obj_TPE, dtype=float)
                                info2send = struct.pack('f' * len(tumorPrjExp2Send), *tumorPrjExp2Send)
                                client_socket.sendall(info2send)

                            if str(data)[2:5]=="utr":
                                trj2send=[]
                                stopUp=pickle.load(open("stopUp.p", "rb"))
                                startUp=pickle.load(open("startUp.p", "rb"))
                                trj2send.extend(startUp)
                                trj2send.extend(stopUp)
                                print("start Up is", startUp)
                                print("stop Up is", stopUp)
                                trj2send = np.asarray(trj2send, dtype=float)
                                info2send = struct.pack('f' * len(trj2send), *trj2send)
                                client_socket.sendall(info2send)



                        if not data:
                            break


    def server_thread(server_socket, server_number):
        while True:
            print(f"Waiting for the next client to connect on Server {server_number}...")
            client_socket, client_address = server_socket.accept()
            threading.Thread(target=handle_client, args=(client_socket, client_address, server_number)).start()

    while True:
        HOST = "192.168.0.101"
        PORT1 = 1000  
        PORT2 = 2000

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket1, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket2:
                server_socket1.bind((HOST, PORT1))
                server_socket2.bind((HOST, PORT2))
                server_socket1.listen()
                server_socket2.listen()



                print(f"Server1 listening on {HOST}:{PORT1}")
                print(f"Server2 listening on {HOST}:{PORT2}")

                while True:
                    
                    threading.Thread(target=server_thread, args=(server_socket1, 1)).start()
                    threading.Thread(target=server_thread, args=(server_socket2, 2)).start()
               
                    threading.Event().wait()

                   
                   
                    
                    
                    
                    

                