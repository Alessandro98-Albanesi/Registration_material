import numpy as np
import cv2
import open3d as o3d
import time
import copy
from matplotlib import pyplot as plt

class Open3dVisualizer():
    def __init__(self):

        self.point_cloud = o3d.geometry.PointCloud()
        self.o3d_started = False

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="AK Stream")

        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.ply_filename = f".\saved_pcd\AK_3D_recon_{self.timestr}.ply"
      

    def __call__(self, points_3d, rgb_image=None):
        self.AK_stream_update(points_3d, rgb_image)


    def AK_stream_update(self, points_3d, rgb_image=None):
        # self.vis.create_window(window_name="AK Stream")
        vis = self.vis
        def save_ply(vis):
            self.timestr = time.strftime("%Y%m%d-%H%M%S")
            self.ply_filename = f".\saved_pcd\saved_pcd_{self.timestr}.ply"
            o3d.io.write_point_cloud(self.ply_filename, self.point_cloud)
            print("saved pcd")

        self.vis.register_key_callback(ord("S"), save_ply)
        blue_color = np.array([0.0, 0.0, 1.0])  # Blue color
        colors = np.tile(blue_color, (len(self.point_cloud.points), 1))
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        # Add values to vectors
        self.point_cloud.points = o3d.utility.Vector3dVector(points_3d)
        if rgb_image is not None:
            colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB).reshape(-1, 3) / 255
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

        self.point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # flip y and z (o3d frame)

        # Add geometries if it is the first time
        if not self.o3d_started:
            self.vis.add_geometry(self.point_cloud, )
            self.o3d_started = True

        else:
            self.vis.update_geometry(self.point_cloud)

        self.vis.poll_events()
        self.vis.update_renderer()


class Opend3dPCD():
    def __int__(self):
        self.point_cloud = o3d.geometry.PointCloud()
        self.o3d_started = False

        self.workflow = r"pcd processing"

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        # self.vis.create_window(window_name="PCD")

    def readPCD_fromFile(self, filename):
        # reading from file created from a o3d data visualization: no need to flip on y and z the pcd
        self.point_cloud = o3d.io.read_point_cloud(filename, remove_nan_points=True, remove_infinite_points=True)


    def voxel_downsampling(self, pcd_to_process, voxel_size):
        self.workflow = r"voxel downsamplig"
        self.point_cloud_downsampled = copy.deepcopy(pcd_to_process)
        self.point_cloud_downsampled.voxel_down_sample(voxel_size= voxel_size)

    def segment_plane(self, pcd_to_process, distance_threshold, ransac_n, iterations):
        self.point_cloud_plane_segm = copy.deepcopy(pcd_to_process)
        plane_model, inliers = self.point_cloud_plane_segm.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=iterations)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        print("Displaying pointcloud with planar points in red ...")
        inlier_cloud = self.point_cloud_plane_segm.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = self.point_cloud_plane_segm.select_by_index(inliers, invert=True)
        o3d.visualization.draw([inlier_cloud, outlier_cloud])

    def dbScan_clustering(self, pcd_to_process):
        self.point_cloud_dbscan = copy.deepcopy(pcd_to_process)

        # o3d debug
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                self.point_cloud_dbscan.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        self.point_cloud_dbscan.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([self.point_cloud_dbscan])


    def visualize_staticPCD(self):
        self.vis.create_window(window_name=self.workflow)
        self.vis.add_geometry([self.point_cloud])
        self.vis.run()
        self.vis.destroy_window()

