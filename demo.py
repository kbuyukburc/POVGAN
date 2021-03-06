import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

import cv2
import open3d as o3d
from glob import glob
import numpy as np
from pathlib import Path
from time import sleep
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

model = create_model(opt)
nuscenes_lidar_extrinsic = {
    "LIDAR_TOP":{
    "translation": [0.943713,0.0,1.84023],
    "rotation": [0.7077955119163518,-0.006492242056004365,0.010646214713995808,-0.7063073142877817]
    }
}

nuscenes_camera_extrinsic = {
    1:{
    "translation": [1.70079118954,0.0159456324149,1.51095763913],
    "rotation": [0.4998015430569128,-0.5030316162024876,0.4997798114386805,-0.49737083824542755]
    },
    2:{
    "translation": [0.0283260309358,0.00345136761476,1.57910346144],
    "rotation": [0.5037872666382278,-0.49740249788611096,-0.4941850223835201,0.5045496097725578]
    },
    3:{"translation": [1.5508477543,-0.493404796419,1.49574800619],
    "rotation": [0.2060347966337182,-0.2026940577919598,0.6824507824531167,-0.6713610884174485]
    },
    4:{"translation": [1.52387798135,0.494631336551,1.50932822144],
    "rotation": [0.6757265034669446,-0.6736266522251881,0.21214015046209478,-0.21122827103904068],
    },
    5:{"translation": [1.0148780988,-0.480568219723,1.56239545128],
    "rotation": [0.12280980120078765,-0.132400842670559,-0.7004305821388234,0.690496031265798],
    },
    6:{"translation": [1.03569100218,0.484795032713,1.59097014818],
    "rotation": [0.6924185592174665,-0.7031619420114925,-0.11648342771943819,0.11203317912370753],
    },
}

def calibDictToMatrix(calib_dict : dict) -> np.ndarray:
    quat = Quaternion(calib_dict['rotation'])
    matrix = np.eye(4)
    matrix[:3,:3] = quat.rotation_matrix
    matrix[:3, 3] = calib_dict['translation']
    return matrix


class BinDataset:
    bin_files =[]
    def __init__(self, folder_name):
        folder = Path(folder_name)
        self.bin_files = list(folder.glob('*.bin'))

    def __getitem__(self, index):
        index = index % len(self.bin_files)
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
    intrinsic = np.array([[1260.0,0.0,800.0],
                        [0.0,1260.0,450.0],
                        [0.0,0.0,1.0]])
    width = 1600
    height = 900
    lidar_data = None
    generated_data = None
    def __init__(self, dataset):
        self.dataset = dataset        
        self.app.register_key_callback(ord('Q'), self.__close_app)
        self.app.register_key_callback(ord('D'), self.__next_frame)
        self.app.register_key_callback(ord('A'), self.__back_frame)
        for i in range(1,7):
            self.app.register_key_callback(ord(f'{i}'), self.__get_camera(i))

    def __next_frame(self, app):
        self.index += 1
        self.get_frame()

    def __back_frame(self, app):
        self.index -= 1
        self.get_frame()

    def __get_camera(self, number):
        def get_camera(app):
            cntrl = app.get_view_control()
            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic = cntrl.convert_to_pinhole_camera_parameters().intrinsic
            camera_extrinsic = calibDictToMatrix(nuscenes_camera_extrinsic[number])
            lidar_extrinsic = calibDictToMatrix(nuscenes_lidar_extrinsic["LIDAR_TOP"])
            param.extrinsic = np.linalg.inv(camera_extrinsic) @ lidar_extrinsic
            cntrl.convert_from_pinhole_camera_parameters(param)
        return get_camera

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
        self.lidar_data = np.zeros([2, 512, 512])
        reflection_flag = reflection[flag]
        depth_flag = depth[flag]
        if len(img_points_flag):
            img_points_flag = img_points_flag.copy()
            img_points_flag[:, 0] = img_points_flag[:, 0] * 512 / self.width
            img_points_flag[:, 1] = img_points_flag[:, 1] * 512 / self.height
            img_points_flag = img_points_flag.astype(int)
            self.lidar_data[0, img_points_flag[:,1], img_points_flag[:,0]] = depth_flag
            self.lidar_data[1, img_points_flag[:,1], img_points_flag[:,0]] = reflection_flag        
        with torch.no_grad():
            label_ts = torch.FloatTensor(self.lidar_data[None, ...]/255).cuda()
            generated = model.inference(label_ts, None)
        self.generated_data = cv2.cvtColor(util.tensor2im(generated[0]), cv2.COLOR_RGB2BGR)
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
            cv2.imshow("Generated Image", self.generated_data)
            key = cv2.waitKey(1)
            self.app.poll_events()
            self.app.update_renderer()
            

        self.app.destroy_window()
        cv2.destroyAllWindows()

dataset = BinDataset('./samples')
PointCloudViewer(dataset).start()