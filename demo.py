import cv2
import open3d as o3d
from glob import glob
import numpy as np
from pathlib import Path
from time import sleep
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

class BinDataset:
    bin_files =[]
    def __init__(self, folder_name):
        folder = Path(folder_name)
        self.bin_files = list(folder.glob('*.bin'))

    def __getitem__(self, index):
        return np.fromfile(self.bin_files[index], dtype=np.float32).reshape(-1,5)[..., :4]

class PointCloudViewer:
    app = o3d.visualization.VisualizerWithKeyCallback()
    app.create_window(window_name = "LiDAR")
    cntrl = app.get_view_control()
    render = app.get_render_option()
    render.point_size = 5
    index = 0
    pcd = o3d.geometry.PointCloud()
    dataset = []
    __app_close = True
    intrinsic = np.array([[605.331,   0.,    527.805],
                        [  0.,    604.306, 410.819],
                        [  0.,      0.,      1.   ]])
    width = 1024
    height = 768
    lidar_data = None
    def __init__(self, dataset):
        self.dataset = dataset
        self.app.register_key_callback(ord('Q'), self.__close_app)

    def __close_app(self, app):
        print(f"Window {app.get_window_name()} Shuting Down!") 
        self.__app_close = False
    
    def get_frame(self):
        self.pcd = o3d.geometry.PointCloud()
        points = self.dataset[self.index]        
        reflection = points[:, 3]
        points = points[:, :3]
        colors = np.c_[reflection, reflection, reflection]
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors/255)        

    def get_projected(self, extrinsic):
        pcd_transformed = np.asarray(deepcopy(self.pcd).transform(extrinsic).points)
        depth = pcd_transformed[:, 2]
        points = np.asarray(self.pcd.points)
        reflection = np.asarray(self.pcd.colors)[:,0] * 255
        rvec = R.from_matrix(extrinsic[:3, :3]).as_rotvec()
        tvec = extrinsic[:3, 3]                            
        img_points, joc = cv2.projectPoints(points.T, rvec, tvec, self.intrinsic, np.array([[0,0,0,0,0]], dtype=np.float32))
        flag = (img_points[:,0, 0] < self.width) & (img_points[:,0, 0] > 0) & \
            (img_points[:,0, 1] < self.height) & (img_points[:,0, 1] > 0) & (depth > 0)
        img_points_flag = img_points[:,0,:][flag].astype(int)
        self.lidar_data = np.zeros([2, self.height, self.width])
        reflection_flag = reflection[flag]
        depth_flag = depth[flag]
        if len(img_points_flag):
            self.lidar_data[0, img_points_flag[:,1], img_points_flag[:,0]] = depth_flag
            self.lidar_data[1, img_points_flag[:,1], img_points_flag[:,0]] = reflection_flag

    def start(self):
        self.get_frame()
        arrow = o3d.geometry.TriangleMesh.create_coordinate_frame(0.3)
        while self.__app_close:
            cntrl = self.app.get_view_control()
            param = cntrl.convert_to_pinhole_camera_parameters()
            self.app.clear_geometries()
            self.app.add_geometry(self.pcd)
            self.app.add_geometry(arrow)
            cntrl.convert_from_pinhole_camera_parameters(param)
            # sleep(0.1)
            self.get_projected(deepcopy(param.extrinsic))
            cv2.imshow("Projected Image", self.lidar_data[1])
            key = cv2.waitKey(10)
            self.app.poll_events()
            self.app.update_renderer()
            

        self.app.destroy_window()
        cv2.destroyAllWindows()

dataset = BinDataset('./samples')
PointCloudViewer(dataset).start()